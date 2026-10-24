// Lure TTS C API over the qwen3_tts session of audio.cpp (Apache-2.0, see LICENSE and NOTICE).
#include "lure_tts.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/prompt_segment.h"
#include "engine/models/qwen3_tts/session.h"
#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"

#include <ggml-backend.h>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace q3 = engine::models::qwen3_tts;
namespace rt = engine::runtime;

// What the core's model inspection (core/lure_tts/model.py) derives from the folder.
struct lure_model_defaults {
    std::string family = "qwen3_tts";   // "qwen3_tts" | "lure"
    std::string variant;                // "base" | "custom_voice" | "voice_design"
    bool non_streaming = true;          // default prompt layout
    std::string language = "auto";      // default language
    std::string instruct = "none";      // "none" | "optional" | "required"
    std::vector<std::pair<std::string, int32_t>> speakers;  // sorted by name
    std::vector<std::string> languages; // "auto" + non-dialect codec language keys
};

struct lure_tts {
    std::shared_ptr<const q3::Qwen3TTSAssets> assets;
    std::unique_ptr<q3::Qwen3TTSSession> session;
    std::unique_ptr<q3::Qwen3TextTokenizer> tokenizer;
    lure_model_defaults defaults;
    uint64_t fingerprint = 0;
    std::mutex mutex;
};

struct lure_voice {
    q3::Qwen3VoiceClonePrompt prompt;
    std::string ref_text;
    uint64_t fingerprint = 0;
};

namespace {

thread_local std::string g_last_error;

void set_error(const std::string & message) { g_last_error = message; }

uint64_t fnv1a(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

constexpr uint64_t kFnvOffset = 1469598103934665603ull;

uint64_t hash_file(uint64_t hash, const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return fnv1a(hash, data.data(), data.size());
}

// Voices depend on the speech tokenizer encoder, the speaker encoder and the text tokenizer. The
// fingerprint covers the configs; a fine-tune that changes encoder weights must also change a config
// field, or the voice cache must be cleared.
uint64_t model_fingerprint(const q3::Qwen3TTSAssets & assets) {
    uint64_t hash = kFnvOffset;
    hash = hash_file(hash, assets.resources.require_file("config"));
    hash = hash_file(hash, assets.resources.require_file("speech_tokenizer_config"));
    hash = hash_file(hash, assets.resources.require_file("vocab"));
    return hash;
}

engine::core::BackendType parse_backend(const char * name) {
    const std::string value = name == nullptr ? "auto" : name;
    if (value.empty() || value == "auto" || value == "best") return engine::core::BackendType::BestAvailable;
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "hip" || value == "rocm") return engine::core::BackendType::Hip;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "metal") return engine::core::BackendType::Metal;
    throw std::runtime_error("unknown backend '" + value + "' (auto|metal|cuda|vulkan|cpu)");
}

std::unordered_map<std::string, std::string> parse_options(const char * text) {
    std::unordered_map<std::string, std::string> out;
    if (text == nullptr) return out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (item.empty()) continue;
        const auto eq = item.find('=');
        if (eq == std::string::npos) throw std::runtime_error("option without '=': " + item);
        out[item.substr(0, eq)] = item.substr(eq + 1);
    }
    return out;
}

q3::Qwen3TTSGenerationOptions to_generation(const lure_params & p, const q3::Qwen3TTSConfig & config) {
    q3::Qwen3TTSGenerationOptions o;
    o.temperature = p.temperature;
    o.top_k = p.top_k;
    o.top_p = p.top_p;
    o.repetition_penalty = p.repetition_penalty;
    o.subtalker_temperature = p.subtalker_temperature;
    o.subtalker_top_k = p.subtalker_top_k;
    o.subtalker_top_p = p.subtalker_top_p;
    o.do_sample = p.do_sample != 0;
    o.subtalker_do_sample = p.subtalker_do_sample != 0;
    o.max_new_tokens = p.max_new_tokens > 0 ? std::min<int64_t>(p.max_new_tokens, config.max_new_tokens)
                                            : config.max_new_tokens;
    o.seed = p.has_seed ? p.seed : rt::random_u32_seed();
    if (o.do_sample && o.temperature <= 0.0F) throw std::runtime_error("temperature must be > 0 when sampling");
    if (o.subtalker_do_sample && o.subtalker_temperature <= 0.0F) {
        throw std::runtime_error("subtalker_temperature must be > 0 when sampling");
    }
    if (o.top_p < 0.0F || o.top_p > 1.0F || o.subtalker_top_p < 0.0F || o.subtalker_top_p > 1.0F) {
        throw std::runtime_error("top_p must be in [0, 1]");
    }
    if (o.top_k < 0 || o.subtalker_top_k < 0) throw std::runtime_error("top_k must be >= 0");
    if (o.repetition_penalty <= 0.0F) throw std::runtime_error("repetition_penalty must be > 0");
    return o;
}

lure_params resolve_params(const lure_params * params) {
    lure_params p;
    lure_default_params(&p);
    if (params != nullptr) {
        if (params->struct_size != sizeof(lure_params)) {
            throw std::runtime_error("lure_params.struct_size mismatch (header/library version skew)");
        }
        p = *params;
    }
    return p;
}

const char * variant_name(q3::Qwen3TTSVariant variant) {
    switch (variant) {
        case q3::Qwen3TTSVariant::Base: return "base";
        case q3::Qwen3TTSVariant::CustomVoice: return "custom_voice";
        case q3::Qwen3TTSVariant::VoiceDesign: return "voice_design";
    }
    return "unknown";
}

// lure.json next to the model (or embedded in a GGUF), as core/lure_tts/model.py reads it.
std::optional<engine::io::json::Value> read_lure_json(const q3::Qwen3TTSAssets & assets) {
    for (const auto & dir : {assets.resources.model_root(), assets.resources.model_directory()}) {
        if (dir.empty()) continue;
        const auto path = dir / "lure.json";
        if (std::filesystem::is_regular_file(path)) return engine::io::json::parse_file(path);
    }
    return std::nullopt;
}

lure_model_defaults model_defaults(const q3::Qwen3TTSAssets & assets) {
    const auto & config = assets.config;
    lure_model_defaults out;
    out.variant = variant_name(config.variant);
    const auto lure_json = read_lure_json(assets);
    const bool lure = lure_json.has_value() || assets.resources.find_file("turbo_config") != nullptr;
    if (lure && config.variant != q3::Qwen3TTSVariant::Base) {
        throw std::runtime_error("lure.json / turbo.json on a checkpoint that is not Base (" + out.variant + ")");
    }
    if (lure) {
        out.family = "lure";
        out.language = "english";
        out.instruct = "optional";
        std::string layout = "non_streaming";
        if (lure_json) {
            if (const auto * v = lure_json->find("prompt_layout"); v != nullptr && v->is_string()) layout = v->as_string();
        }
        if (layout != "non_streaming" && layout != "streaming") {
            throw std::runtime_error("lure.json prompt_layout must be non_streaming or streaming, got " + layout);
        }
        out.non_streaming = layout == "non_streaming";
    } else {
        out.non_streaming = config.variant != q3::Qwen3TTSVariant::Base;
        if (config.variant == q3::Qwen3TTSVariant::CustomVoice) out.instruct = config.tts_model_size == "0b6" ? "none" : "optional";
        if (config.variant == q3::Qwen3TTSVariant::VoiceDesign) out.instruct = "required";
    }
    for (const auto & [name, id] : config.talker.speaker_id) out.speakers.emplace_back(name, static_cast<int32_t>(id));
    std::sort(out.speakers.begin(), out.speakers.end());
    std::vector<std::string> dialects;
    for (const auto & entry : config.talker.speaker_dialect) {
        if (entry.second) dialects.push_back(*entry.second);
    }
    for (const auto & entry : config.talker.codec_language_id) {
        if (std::find(dialects.begin(), dialects.end(), entry.first) == dialects.end()) out.languages.push_back(entry.first);
    }
    std::sort(out.languages.begin(), out.languages.end());
    out.languages.insert(out.languages.begin(), "auto");
    return out;
}

std::string model_info_json(const q3::Qwen3TTSAssets & assets, const lure_model_defaults & d) {
    namespace json = engine::io::json;
    const auto & config = assets.config;
    std::string out = "{\"variant\":" + json::stringify_string(d.variant);
    out += ",\"size\":" + json::stringify_string(config.tts_model_size);
    out += ",\"family\":" + json::stringify_string(d.family);
    out += ",\"talker_hidden_size\":" + std::to_string(config.talker.hidden_size);
    out += ",\"code_predictor_hidden_size\":" + std::to_string(config.code_predictor.hidden_size);
    out += ",\"speaker_encoder_dim\":" +
           (config.variant == q3::Qwen3TTSVariant::Base ? std::to_string(config.speaker_encoder.embedding_dim) : "null");
    out += ",\"speakers\":{";
    for (size_t i = 0; i < d.speakers.size(); ++i) {
        out += (i ? "," : "") + json::stringify_string(d.speakers[i].first) + ":" + std::to_string(d.speakers[i].second);
    }
    out += "},\"speaker_dialects\":{";
    bool first = true;
    for (const auto & [name, id] : d.speakers) {
        const auto it = config.talker.speaker_dialect.find(name);
        if (it == config.talker.speaker_dialect.end() || !it->second) continue;
        out += (first ? "" : ",") + json::stringify_string(name) + ":" + json::stringify_string(*it->second);
        first = false;
    }
    out += "},\"languages\":[";
    for (size_t i = 0; i < d.languages.size(); ++i) out += (i ? "," : "") + json::stringify_string(d.languages[i]);
    out += "],\"default_language\":" + json::stringify_string(d.language);
    out += ",\"default_layout\":" + json::stringify_string(d.non_streaming ? "non_streaming" : "streaming");
    out += ",\"modes\":";
    out += config.variant == q3::Qwen3TTSVariant::Base ? "[\"icl\",\"xvec\"]"
         : config.variant == q3::Qwen3TTSVariant::CustomVoice ? "[\"custom_voice\"]" : "[\"voice_design\"]";
    out += ",\"instruct\":" + json::stringify_string(d.instruct);
    out += ",\"turbo\":" + std::string(assets.turbo ? "true" : "false") + "}";
    return out;
}

lure_request resolve_request(const lure_request * request) {
    if (request == nullptr) throw std::runtime_error("request is NULL");
    if (request->struct_size != sizeof(lure_request)) {
        throw std::runtime_error("lure_request.struct_size mismatch (header/library version skew)");
    }
    return *request;
}

// lure_request -> segment prompt (the voice is filled in by the caller).
q3::Qwen3SegmentPrompt to_segment(const lure_request & r, const q3::Qwen3TTSConfig & config,
                                  const lure_model_defaults & defaults) {
    if (r.text == nullptr) throw std::runtime_error("request text is NULL");
    q3::Qwen3SegmentPrompt segment;
    switch (r.mode) {
        case LURE_MODE_ICL: segment.mode = q3::Qwen3SegmentMode::Icl; break;
        case LURE_MODE_XVEC: segment.mode = q3::Qwen3SegmentMode::XVector; break;
        case LURE_MODE_CUSTOM_VOICE: segment.mode = q3::Qwen3SegmentMode::CustomVoice; break;
        case LURE_MODE_VOICE_DESIGN: segment.mode = q3::Qwen3SegmentMode::VoiceDesign; break;
        default: throw std::runtime_error("unknown mode " + std::to_string(r.mode));
    }
    if (r.layout == LURE_LAYOUT_DEFAULT) {
        segment.non_streaming_mode = defaults.non_streaming;
    } else if (r.layout == LURE_LAYOUT_NON_STREAMING || r.layout == LURE_LAYOUT_STREAMING) {
        segment.non_streaming_mode = r.layout == LURE_LAYOUT_NON_STREAMING;
    } else {
        throw std::runtime_error("unknown layout " + std::to_string(r.layout));
    }
    segment.text = r.text;
    segment.instruct = r.instruct != nullptr ? r.instruct : "";
    segment.language = r.language != nullptr && r.language[0] != '\0' ? r.language : defaults.language;
    if (segment.mode == q3::Qwen3SegmentMode::CustomVoice) {
        if (r.speaker_id >= 0) {
            for (const auto & [name, id] : config.talker.speaker_id) {
                if (id == r.speaker_id) segment.speaker = name;
            }
            if (segment.speaker.empty()) throw std::runtime_error("no preset speaker has codec id " + std::to_string(r.speaker_id));
            if (r.speaker != nullptr && r.speaker[0] != '\0') {
                const auto it = config.talker.speaker_id.find(segment.speaker);
                std::string named(r.speaker);
                std::transform(named.begin(), named.end(), named.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (config.talker.speaker_id.find(named) == config.talker.speaker_id.end() ||
                    config.talker.speaker_id.at(named) != it->second) {
                    throw std::runtime_error("speaker '" + std::string(r.speaker) + "' does not have codec id " +
                                             std::to_string(r.speaker_id));
                }
                segment.speaker = named;
            }
        } else if (r.speaker != nullptr && r.speaker[0] != '\0') {
            segment.speaker = r.speaker;
        } else {
            throw std::runtime_error("custom_voice needs a speaker name or speaker_id");
        }
    } else if ((r.speaker != nullptr && r.speaker[0] != '\0') || r.speaker_id >= 0) {
        throw std::runtime_error("a preset speaker is only used in custom_voice mode");
    }
    return segment;
}

// Prefill for a loaded model; checks the voice for the clone modes.
q3::Qwen3TalkerPrefill build_prefill(lure_tts & tts, const lure_request & r) {
    auto segment = to_segment(r, tts.assets->config, tts.defaults);
    const bool clone = segment.mode == q3::Qwen3SegmentMode::Icl || segment.mode == q3::Qwen3SegmentMode::XVector;
    if (clone) {
        if (r.voice == nullptr) throw std::runtime_error("icl and xvec need a voice");
        if (r.voice->fingerprint != tts.fingerprint) throw std::runtime_error("voice belongs to a different model");
        if (segment.mode == q3::Qwen3SegmentMode::Icl && r.voice->ref_text.empty()) {
            throw std::runtime_error("this voice has no reference transcript: use xvec, or make it with ref_text");
        }
        segment.voice = &r.voice->prompt;
    } else if (r.voice != nullptr) {
        throw std::runtime_error("a voice is only used in icl and xvec modes");
    }
    return tts.session->lure_build_prefill(segment);
}

// ---- voice file (.alv), little-endian; layout in docs/backends/cpp.md ----
constexpr char kVoiceMagic[8] = {'A', 'L', 'L', 'U', 'R', 'E', 'V', '1'};
constexpr uint32_t kVoiceVersion = 1;

template <class T> void put(std::ostream & out, const T & value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}
template <class T> T get(std::istream & in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated voice file");
    return value;
}

void write_voice(const lure_voice & voice, const std::filesystem::path & path) {
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write voice file: " + tmp);
        out.write(kVoiceMagic, sizeof(kVoiceMagic));
        put(out, kVoiceVersion);
        put(out, voice.fingerprint);
        const auto & p = voice.prompt;
        const q3::Qwen3SpeechCodes codes = p.reference_codes.value_or(q3::Qwen3SpeechCodes{{}, 0, 16});
        put<int64_t>(out, p.speaker_embedding.dims);
        put<int64_t>(out, codes.frames);
        put<int64_t>(out, codes.code_groups);
        put<int64_t>(out, static_cast<int64_t>(voice.ref_text.size()));
        out.write(reinterpret_cast<const char *>(p.speaker_embedding.values.data()),
                  static_cast<std::streamsize>(p.speaker_embedding.values.size() * sizeof(float)));
        out.write(reinterpret_cast<const char *>(codes.codes.data()),
                  static_cast<std::streamsize>(codes.codes.size() * sizeof(int32_t)));
        out.write(voice.ref_text.data(), static_cast<std::streamsize>(voice.ref_text.size()));
        if (!out) throw std::runtime_error("failed writing voice file: " + tmp);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("cannot move voice file into place: " + ec.message());
}

std::unique_ptr<lure_voice> read_voice(lure_tts & tts, const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open voice file: " + path.string());
    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kVoiceMagic, sizeof(magic)) != 0) throw std::runtime_error("not a Lure TTS voice file");
    if (get<uint32_t>(in) != kVoiceVersion) throw std::runtime_error("unsupported voice file version");
    auto voice = std::make_unique<lure_voice>();
    voice->fingerprint = get<uint64_t>(in);
    if (voice->fingerprint != tts.fingerprint) {
        throw std::runtime_error("voice file was built for a different model (fingerprint mismatch); rebuild it");
    }
    const auto dims = get<int64_t>(in);
    const auto frames = get<int64_t>(in);
    const auto groups = get<int64_t>(in);
    const auto text_len = get<int64_t>(in);
    if (dims != tts.assets->config.talker.hidden_size || groups != tts.assets->config.talker.num_code_groups ||
        frames < 0 || frames > 1000000 || (frames == 0 && text_len != 0) || text_len < 0 || text_len > (1 << 24)) {
        throw std::runtime_error("voice file shape does not match the model");
    }
    auto & p = voice->prompt;
    p.icl_mode = text_len > 0;
    p.speaker_embedding.dims = dims;
    p.speaker_embedding.values.resize(static_cast<size_t>(dims));
    in.read(reinterpret_cast<char *>(p.speaker_embedding.values.data()), static_cast<std::streamsize>(dims * sizeof(float)));
    q3::Qwen3SpeechCodes codes;
    codes.frames = frames;
    codes.code_groups = groups;
    codes.codes.resize(static_cast<size_t>(frames * groups));
    in.read(reinterpret_cast<char *>(codes.codes.data()), static_cast<std::streamsize>(codes.codes.size() * sizeof(int32_t)));
    voice->ref_text.resize(static_cast<size_t>(text_len));
    in.read(voice->ref_text.data(), static_cast<std::streamsize>(text_len));
    if (!in) throw std::runtime_error("truncated voice file");
    if (frames > 0) p.reference_codes = std::move(codes);
    // Re-tokenize with the loaded tokenizer instead of trusting stored ids.
    if (p.icl_mode) p.reference_text_ids = tts.tokenizer->encode(tts.tokenizer->build_reference_prompt(voice->ref_text));
    return voice;
}

uint64_t audio_key(const float * pcm, size_t n, int32_t sample_rate, const char * ref_text) {
    uint64_t hash = kFnvOffset;
    hash = fnv1a(hash, &sample_rate, sizeof(sample_rate));
    hash = fnv1a(hash, pcm, n * sizeof(float));
    hash = fnv1a(hash, ref_text, std::strlen(ref_text));
    return hash;
}

// With GGML_BACKEND_DL (CPU_ALL_VARIANTS presets) the ggml backends are separate shared libraries.
// ggml_backend_load_all() searches the executable's directory, which for a ctypes host is Python's;
// load them from this library's own directory instead.
#ifdef GGML_BACKEND_DL
std::once_flag g_backend_load_once;
#endif

void load_library_backends() {
#ifdef GGML_BACKEND_DL
    std::call_once(g_backend_load_once, [] {
        std::filesystem::path library_path;
#ifdef _WIN32
        HMODULE module = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&g_backend_load_once), &module)) {
            std::vector<wchar_t> path(32768);
            const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
            if (length > 0 && length < path.size()) library_path = std::wstring(path.data(), length);
        }
#else
        Dl_info info{};
        if (dladdr(&g_backend_load_once, &info) != 0 && info.dli_fname != nullptr) library_path = info.dli_fname;
#endif
        if (library_path.empty()) {
            ggml_backend_load_all();
            return;
        }
        const auto directory = std::filesystem::canonical(library_path).parent_path().u8string();
        ggml_backend_load_all_from_path(directory.c_str());
    });
#endif
}

char * dup_string(const std::string & value) {
    auto * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out == nullptr) throw std::bad_alloc();
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

template <class F> auto guarded(F && fn, decltype(fn()) failure) -> decltype(fn()) {
    try {
        g_last_error.clear();
        return fn();
    } catch (const std::exception & error) {
        set_error(error.what());
    } catch (...) {
        set_error("unknown error");
    }
    return failure;
}

}  // namespace

extern "C" {

uint32_t lure_abi_version(void) { return LURE_ABI_VERSION; }

const char * lure_build_info(void) {
    load_library_backends();
    static std::string info = [] {
        std::string out = "lure-cpp (audio.cpp fork) backends:";
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            out += " ";
            out += ggml_backend_reg_name(ggml_backend_reg_get(i));
        }
        return out;
    }();
    return info.c_str();
}

const char * lure_last_error(void) { return g_last_error.c_str(); }

void lure_default_params(lure_params * out) {
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(lure_params);
    out->temperature = 0.9F;
    out->top_k = 50;
    out->top_p = 1.0F;
    out->repetition_penalty = 1.05F;
    out->subtalker_temperature = 0.9F;
    out->subtalker_top_k = 50;
    out->subtalker_top_p = 1.0F;
    out->do_sample = 1;
    out->subtalker_do_sample = 1;
    out->max_new_tokens = 0;
    out->has_seed = 0;
    out->seed = 0;
    out->chunk_frames = 4;
    out->chunk_grow = 1;
}

void lure_default_request(lure_request * out) {
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(lure_request);
    out->mode = LURE_MODE_ICL;
    out->layout = LURE_LAYOUT_DEFAULT;
    out->speaker_id = -1;
}

lure_tts * lure_load(const char * model_dir, const char * backend, int32_t device, int32_t threads,
                         const char * options) {
    return guarded([&]() -> lure_tts * {
        if (model_dir == nullptr) throw std::runtime_error("model_dir is NULL");
        load_library_backends();
        auto tts = std::make_unique<lure_tts>();
        rt::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend);
        session_options.backend.device = device;
        session_options.backend.threads = threads > 0
            ? threads : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        session_options.options = parse_options(options);
        tts->assets = q3::load_qwen3_tts_assets(std::filesystem::u8path(model_dir));
        tts->defaults = model_defaults(*tts->assets);
        // Base sessions load the speech-tokenizer encoder and the speaker encoder for voices;
        // CustomVoice and VoiceDesign checkpoints have neither.
        const auto task = tts->assets->config.variant == q3::Qwen3TTSVariant::Base
            ? rt::VoiceTaskKind::VoiceCloning : rt::VoiceTaskKind::Tts;
        tts->session = std::make_unique<q3::Qwen3TTSSession>(
            rt::TaskSpec{task, rt::RunMode::Offline}, session_options, tts->assets);
        tts->session->prepare({});
        tts->tokenizer = std::make_unique<q3::Qwen3TextTokenizer>(tts->assets);
        tts->fingerprint = model_fingerprint(*tts->assets);
        return tts.release();
    }, nullptr);
}

void lure_free(lure_tts * tts) { delete tts; }

int32_t lure_sample_rate(const lure_tts * tts) {
    return tts == nullptr ? 24000 : tts->assets->config.speech_tokenizer.output_sample_rate;
}

const char * lure_variant(const lure_tts * tts) {
    return tts == nullptr ? nullptr : tts->defaults.variant.c_str();
}

int32_t lure_speaker_count(const lure_tts * tts) {
    return tts == nullptr ? 0 : static_cast<int32_t>(tts->defaults.speakers.size());
}

const char * lure_speaker_name(const lure_tts * tts, int32_t index) {
    if (tts == nullptr || index < 0 || index >= lure_speaker_count(tts)) return nullptr;
    return tts->defaults.speakers[static_cast<size_t>(index)].first.c_str();
}

int32_t lure_speaker_id(const lure_tts * tts, int32_t index) {
    if (tts == nullptr || index < 0 || index >= lure_speaker_count(tts)) return -1;
    return tts->defaults.speakers[static_cast<size_t>(index)].second;
}

char * lure_model_info(const lure_tts * tts) {
    return guarded([&]() -> char * {
        if (tts == nullptr) throw std::runtime_error("tts is NULL");
        return dup_string(model_info_json(*tts->assets, tts->defaults));
    }, nullptr);
}

char * lure_model_info_offline(const char * model_dir) {
    return guarded([&]() -> char * {
        if (model_dir == nullptr) throw std::runtime_error("model_dir is NULL");
        const auto assets = q3::load_qwen3_tts_assets_metadata(std::filesystem::u8path(model_dir));
        return dup_string(model_info_json(*assets, model_defaults(*assets)));
    }, nullptr);
}

lure_voice * lure_voice_create(lure_tts * tts, const float * pcm, size_t n_samples, int32_t sample_rate,
                                   const char * ref_text) {
    return guarded([&]() -> lure_voice * {
        if (tts == nullptr || pcm == nullptr || n_samples == 0) {
            throw std::runtime_error("lure_voice_create needs a model and audio");
        }
        std::lock_guard<std::mutex> lock(tts->mutex);
        rt::AudioBuffer audio;
        audio.sample_rate = sample_rate;
        audio.channels = 1;
        audio.samples.assign(pcm, pcm + n_samples);
        auto voice = std::make_unique<lure_voice>();
        voice->ref_text = ref_text != nullptr ? ref_text : "";
        voice->prompt = tts->session->lure_make_voice(audio, voice->ref_text);
        voice->fingerprint = tts->fingerprint;
        return voice.release();
    }, nullptr);
}

lure_voice * lure_voice_create_from_wav(lure_tts * tts, const char * wav_path, const char * ref_text) {
    return guarded([&]() -> lure_voice * {
        if (wav_path == nullptr) throw std::runtime_error("wav_path is NULL");
        const auto wav = engine::audio::read_wav_f32(std::filesystem::u8path(wav_path));
        std::vector<float> mono(wav.samples.size() / static_cast<size_t>(std::max(1, wav.channels)));
        for (size_t i = 0; i < mono.size(); ++i) {
            float sum = 0.0F;
            for (int c = 0; c < wav.channels; ++c) sum += wav.samples[i * static_cast<size_t>(wav.channels) + static_cast<size_t>(c)];
            mono[i] = sum / static_cast<float>(std::max(1, wav.channels));
        }
        auto * voice = lure_voice_create(tts, mono.data(), mono.size(), wav.sample_rate, ref_text);
        if (voice == nullptr) throw std::runtime_error(g_last_error);
        return voice;
    }, nullptr);
}

lure_voice * lure_voice_cached(lure_tts * tts, const float * pcm, size_t n_samples, int32_t sample_rate,
                                   const char * ref_text, const char * cache_dir) {
    return guarded([&]() -> lure_voice * {
        if (tts == nullptr || pcm == nullptr) throw std::runtime_error("invalid arguments");
        const char * text = ref_text != nullptr ? ref_text : "";
        if (cache_dir == nullptr || cache_dir[0] == '\0') {
            auto * voice = lure_voice_create(tts, pcm, n_samples, sample_rate, text);
            if (voice == nullptr) throw std::runtime_error(g_last_error);
            return voice;
        }
        const std::filesystem::path dir = std::filesystem::u8path(cache_dir);
        std::filesystem::create_directories(dir);
        char name[64];
        std::snprintf(name, sizeof(name), "%016llx-%016llx.alv",
                      static_cast<unsigned long long>(audio_key(pcm, n_samples, sample_rate, text)),
                      static_cast<unsigned long long>(tts->fingerprint));
        const auto path = dir / name;
        if (std::filesystem::exists(path)) {
            try {
                return read_voice(*tts, path).release();
            } catch (const std::exception &) {
                std::filesystem::remove(path);  // stale or corrupt: rebuild below
            }
        }
        auto * voice = lure_voice_create(tts, pcm, n_samples, sample_rate, text);
        if (voice == nullptr) throw std::runtime_error(g_last_error);
        write_voice(*voice, path);
        return voice;
    }, nullptr);
}

int32_t lure_voice_save(const lure_voice * voice, const char * path) {
    return guarded([&]() -> int32_t {
        if (voice == nullptr || path == nullptr) throw std::runtime_error("invalid arguments");
        write_voice(*voice, std::filesystem::u8path(path));
        return 0;
    }, -1);
}

lure_voice * lure_voice_load(lure_tts * tts, const char * path) {
    return guarded([&]() -> lure_voice * {
        if (tts == nullptr || path == nullptr) throw std::runtime_error("invalid arguments");
        return read_voice(*tts, std::filesystem::u8path(path)).release();
    }, nullptr);
}

int32_t lure_voice_frames(const lure_voice * voice) {
    return voice == nullptr || !voice->prompt.reference_codes ? 0 : static_cast<int32_t>(voice->prompt.reference_codes->frames);
}

int32_t lure_voice_has_transcript(const lure_voice * voice) {
    return voice != nullptr && !voice->ref_text.empty() ? 1 : 0;
}

lure_voice * lure_voice_from_codes(lure_tts * tts, const int32_t * codes, int32_t frames, int32_t groups,
                                       const float * speaker, int32_t dims, const char * ref_text) {
    return guarded([&]() -> lure_voice * {
        const std::string text = ref_text != nullptr ? ref_text : "";
        if (tts == nullptr || speaker == nullptr || frames < 0 || (frames > 0 && codes == nullptr)) {
            throw std::runtime_error("lure_voice_from_codes needs a model, a speaker embedding and codes for frames > 0");
        }
        if (frames == 0 && !text.empty()) throw std::runtime_error("a voice with a transcript needs reference codes");
        if (tts->assets->config.variant != q3::Qwen3TTSVariant::Base) {
            throw std::runtime_error("voices need a Qwen3-TTS Base checkpoint");
        }
        const auto & talker = tts->assets->config.talker;
        if ((frames > 0 && groups != talker.num_code_groups) || dims != talker.hidden_size) {
            throw std::runtime_error("voice shape does not match the model: groups " + std::to_string(groups) + " vs " +
                                     std::to_string(talker.num_code_groups) + ", dims " + std::to_string(dims) + " vs " +
                                     std::to_string(talker.hidden_size));
        }
        auto voice = std::make_unique<lure_voice>();
        voice->fingerprint = tts->fingerprint;
        voice->ref_text = text;
        auto & p = voice->prompt;
        p.icl_mode = !text.empty();
        p.speaker_embedding.dims = dims;
        p.speaker_embedding.values.assign(speaker, speaker + dims);
        if (frames > 0) {
            q3::Qwen3SpeechCodes c;
            c.frames = frames;
            c.code_groups = groups;
            c.codes.assign(codes, codes + static_cast<size_t>(frames) * static_cast<size_t>(groups));
            p.reference_codes = std::move(c);
        }
        if (p.icl_mode) p.reference_text_ids = tts->tokenizer->encode(tts->tokenizer->build_reference_prompt(text));
        return voice.release();
    }, nullptr);
}

int64_t lure_voice_codes(const lure_voice * voice, int32_t * out, size_t capacity) {
    if (voice == nullptr) return -1;
    if (!voice->prompt.reference_codes) return 0;
    const auto & codes = voice->prompt.reference_codes->codes;
    if (out != nullptr) std::memcpy(out, codes.data(), std::min(capacity, codes.size()) * sizeof(int32_t));
    return static_cast<int64_t>(codes.size());
}

int64_t lure_voice_speaker(const lure_voice * voice, float * out, size_t capacity) {
    if (voice == nullptr) return -1;
    const auto & values = voice->prompt.speaker_embedding.values;
    if (out != nullptr) std::memcpy(out, values.data(), std::min(capacity, values.size()) * sizeof(float));
    return static_cast<int64_t>(values.size());
}

void lure_voice_free(lure_voice * voice) { delete voice; }

int32_t lure_generate(lure_tts * tts, const lure_request * request, const lure_params * params,
                        lure_pcm_callback callback, void * user, lure_gen_info * info) {
    return guarded([&]() -> int32_t {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        if (tts == nullptr) throw std::runtime_error("tts is NULL");
        const auto r = resolve_request(request);
        if (r.text == nullptr || r.text[0] == '\0') throw std::runtime_error("lure_generate needs non-empty text");
        const auto p = resolve_params(params);
        std::lock_guard<std::mutex> lock(tts->mutex);
        const auto & config = tts->assets->config;
        const auto options = to_generation(p, config);
        const auto prefill = build_prefill(*tts, r);
        // icl decodes [ref ; generated] and cuts the reference; the other modes decode the generated codes alone.
        const q3::Qwen3SpeechCodes * reference = prefill.icl_mode && prefill.reference_codes ? &*prefill.reference_codes : nullptr;
        double first_chunk_ms = -1.0;
        const auto result = tts->session->lure_generate_segment(
            reference, prefill, options, p.chunk_frames,
            p.chunk_grow ? rt::StreamingAudioChunkPolicy::Grow : rt::StreamingAudioChunkPolicy::Fixed,
            [&](const float * samples, size_t count, int sample_rate) {
                if (first_chunk_ms < 0.0) {
                    first_chunk_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                }
                return callback == nullptr || callback(samples, count, sample_rate, user) != 0;
            });
        if (info != nullptr) {
            info->frames = result.frames;
            info->ended_by_eos = result.ended_by_eos ? 1 : 0;
            info->cancelled = result.cancelled ? 1 : 0;
            info->samples = result.samples;
            info->sample_rate = config.speech_tokenizer.output_sample_rate;
            info->talker_ms = result.talker_ms;
            info->decoder_ms = result.decoder_ms;
            info->first_chunk_ms = first_chunk_ms;
            info->total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        return 0;
    }, -1);
}

char * lure_dump_prefill(lure_tts * tts, const lure_request * request) {
    return guarded([&]() -> char * {
        if (tts == nullptr) throw std::runtime_error("tts is NULL");
        const auto r = resolve_request(request);
        std::lock_guard<std::mutex> lock(tts->mutex);
        return dup_string(q3::qwen3_talker_prompt_recipe_json(build_prefill(*tts, r), tts->assets->config));
    }, nullptr);
}

char * lure_dump_prefill_offline(const char * model_dir, const lure_request * request, const char * ref_text,
                                   int32_t ref_frames) {
    return guarded([&]() -> char * {
        if (model_dir == nullptr) throw std::runtime_error("model_dir is NULL");
        const auto r = resolve_request(request);
        const auto assets = q3::load_qwen3_tts_assets_metadata(std::filesystem::u8path(model_dir));
        const q3::Qwen3TextTokenizer tokenizer(assets);
        const auto & config = assets->config;
        auto segment = to_segment(r, config, model_defaults(*assets));
        // A symbolic voice: zero codes and a zero x-vector (only the layout is dumped).
        q3::Qwen3VoiceClonePrompt voice;
        voice.speaker_embedding.dims = config.talker.hidden_size;
        voice.speaker_embedding.values.assign(static_cast<size_t>(config.talker.hidden_size), 0.0F);
        voice.icl_mode = segment.mode == q3::Qwen3SegmentMode::Icl;
        if (voice.icl_mode) {
            if (ref_text == nullptr || ref_frames <= 0) throw std::runtime_error("icl needs ref_text and ref_frames > 0");
            q3::Qwen3SpeechCodes codes;
            codes.frames = ref_frames;
            codes.code_groups = config.talker.num_code_groups;
            codes.codes.assign(static_cast<size_t>(codes.frames * codes.code_groups), 0);
            voice.reference_codes = std::move(codes);
            voice.reference_text_ids = tokenizer.encode(tokenizer.build_reference_prompt(ref_text));
        }
        if (segment.mode == q3::Qwen3SegmentMode::Icl || segment.mode == q3::Qwen3SegmentMode::XVector) {
            segment.voice = &voice;
        }
        const auto prefill = q3::build_qwen3_segment_prefill(tokenizer, config, segment);
        return dup_string(q3::qwen3_talker_prompt_recipe_json(prefill, config, false));
    }, nullptr);
}

char * lure_tokenize_offline(const char * model_dir, const char * text) {
    return guarded([&]() -> char * {
        if (model_dir == nullptr || text == nullptr) throw std::runtime_error("lure_tokenize_offline needs model_dir and text");
        const auto assets = q3::load_qwen3_tts_assets_metadata(std::filesystem::u8path(model_dir));
        const q3::Qwen3TextTokenizer tokenizer(assets);
        const auto ids = tokenizer.encode(text);
        std::string out = "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) out += ",";
            out += std::to_string(ids[i]);
        }
        out += "]";
        return dup_string(out);
    }, nullptr);
}

void lure_string_free(char * value) { std::free(value); }

}  // extern "C"
