// lure_cli: CLI + HTTP server over the Lure TTS C API (lure_tts.h).
//   lure_cli info         --model DIR                       (config + tokenizer only, no weights)
//   lure_cli dump-prefill --model DIR --text T [mode flags] [--ref-frames N] [--full] [--out F]
//   lure_cli make-voice   --model DIR --ref-wav W [--ref-text R] --out V.alv
//   lure_cli generate     --model DIR --text T [mode flags] [--out out.wav] [sampling flags]
//   lure_cli serve        --model DIR [--host 127.0.0.1] [--port 8088] [--voice-cache DIR]
// Mode flags (Base: icl / xvec, CustomVoice: custom_voice, VoiceDesign: voice_design):
//   --voice V.alv | --ref-wav W [--ref-text R]   clone; icl with a transcript, xvec without one or with --x-vector-only
//   --speaker NAME | --speaker-id N              custom_voice preset speaker
//   (neither)                                    voice_design: --instruct describes the voice
//   --instruct I, --language L (default: the checkpoint's), --layout default|non_streaming|streaming
// Common: --backend auto|metal|cuda|vulkan|cpu, --device N, --threads N, --options "k=v;k=v".
#include "lure_tts.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/json.h"
#include "httplib.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace json = engine::io::json;

namespace {

struct Args {
    std::string command;
    std::map<std::string, std::string> values;
    bool has(const std::string & key) const { return values.count(key) != 0; }
    std::string get(const std::string & key, const std::string & fallback = "") const {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }
    const char * cstr(const std::string & key) const {
        const auto it = values.find(key);
        return it == values.end() ? nullptr : it->second.c_str();
    }
};

Args parse_args(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("usage: lure_cli <info|dump-prefill|make-voice|generate|serve> [--flags]");
    Args args;
    args.command = argv[1];
    static const char * kFlags[] = {"--full", "--greedy", "--fixed-chunks", "--x-vector-only"};
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + key);
        bool flag = false;
        for (const char * f : kFlags) flag = flag || key == f;
        if (flag) {
            args.values[key.substr(2)] = "1";
            continue;
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        args.values[key.substr(2)] = argv[++i];
    }
    return args;
}

void check(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + lure_last_error());
}

struct TtsDeleter { void operator()(lure_tts * t) const { lure_free(t); } };
struct VoiceDeleter { void operator()(lure_voice * v) const { lure_voice_free(v); } };
struct StringDeleter { void operator()(char * s) const { lure_string_free(s); } };
using TtsPtr = std::unique_ptr<lure_tts, TtsDeleter>;
using VoicePtr = std::unique_ptr<lure_voice, VoiceDeleter>;
using StringPtr = std::unique_ptr<char, StringDeleter>;

TtsPtr load_tts(const Args & args) {
    if (!args.has("model")) throw std::runtime_error("--model is required");
    TtsPtr tts(lure_load(args.cstr("model"), args.get("backend", "auto").c_str(),
                           std::stoi(args.get("device", "0")), std::stoi(args.get("threads", "0")),
                           args.cstr("options")));
    check(tts != nullptr, "lure_load");
    return tts;
}

std::vector<float> read_mono(const std::string & path, int32_t & sample_rate) {
    const auto wav = engine::audio::read_wav_f32(std::filesystem::u8path(path));
    const size_t channels = static_cast<size_t>(std::max(1, wav.channels));
    std::vector<float> mono(wav.samples.size() / channels);
    for (size_t i = 0; i < mono.size(); ++i) {
        float sum = 0.0F;
        for (size_t c = 0; c < channels; ++c) sum += wav.samples[i * channels + c];
        mono[i] = sum / static_cast<float>(channels);
    }
    sample_rate = wav.sample_rate;
    return mono;
}

bool wants_voice(const Args & args) { return args.has("voice") || args.has("ref-wav"); }

// --voice V.alv, or --ref-wav W with an optional --ref-text (no transcript = x-vector-only voice).
VoicePtr load_voice(lure_tts * tts, const Args & args) {
    if (args.has("voice")) {
        VoicePtr voice(lure_voice_load(tts, args.cstr("voice")));
        check(voice != nullptr, "lure_voice_load");
        return voice;
    }
    if (!args.has("ref-wav")) throw std::runtime_error("--voice or --ref-wav is required");
    int32_t sample_rate = 24000;
    const auto mono = read_mono(args.get("ref-wav"), sample_rate);
    const std::string ref_text = args.has("x-vector-only") ? "" : args.get("ref-text");
    VoicePtr voice(lure_voice_cached(tts, mono.data(), mono.size(), sample_rate, ref_text.c_str(),
                                       args.cstr("voice-cache")));
    check(voice != nullptr, "lure_voice_cached");
    return voice;
}

int32_t parse_layout(const std::string & value) {
    if (value.empty() || value == "default") return LURE_LAYOUT_DEFAULT;
    if (value == "non_streaming" || value == "non-streaming") return LURE_LAYOUT_NON_STREAMING;
    if (value == "streaming") return LURE_LAYOUT_STREAMING;
    throw std::runtime_error("layout must be default, non_streaming or streaming, got " + value);
}

// Mode from the flags: a speaker -> custom_voice; a voice -> icl (xvec without a transcript or with
// --x-vector-only); neither -> voice_design. The strings in args must outlive the request.
lure_request request_from_args(const Args & args, const lure_voice * voice, bool has_voice) {
    lure_request r;
    lure_default_request(&r);
    r.text = args.cstr("text");
    r.instruct = args.cstr("instruct");
    r.language = args.cstr("language");
    r.layout = parse_layout(args.get("layout"));
    if (args.has("speaker") || args.has("speaker-id")) {
        if (has_voice) throw std::runtime_error("pass a voice (clone) or a speaker (preset), not both");
        r.mode = LURE_MODE_CUSTOM_VOICE;
        r.speaker = args.cstr("speaker");
        if (args.has("speaker-id")) r.speaker_id = std::stoi(args.get("speaker-id"));
    } else if (has_voice) {
        const bool transcript = voice != nullptr ? lure_voice_has_transcript(voice) != 0 : args.has("ref-text");
        r.mode = args.has("x-vector-only") || !transcript ? LURE_MODE_XVEC : LURE_MODE_ICL;
        r.voice = voice;
    } else {
        if (!args.has("instruct")) {
            throw std::runtime_error("no voice and no speaker means voice_design: pass --instruct describing the voice "
                                     "(or --ref-wav / --voice for a Base model, --speaker for a CustomVoice model)");
        }
        r.mode = LURE_MODE_VOICE_DESIGN;
    }
    return r;
}

lure_params params_from_args(const Args & args) {
    lure_params p;
    lure_default_params(&p);
    if (args.has("temperature")) p.temperature = std::stof(args.get("temperature"));
    if (args.has("top-k")) p.top_k = std::stoi(args.get("top-k"));
    if (args.has("top-p")) p.top_p = std::stof(args.get("top-p"));
    if (args.has("repetition-penalty")) p.repetition_penalty = std::stof(args.get("repetition-penalty"));
    if (args.has("subtalker-temperature")) p.subtalker_temperature = std::stof(args.get("subtalker-temperature"));
    if (args.has("subtalker-top-k")) p.subtalker_top_k = std::stoi(args.get("subtalker-top-k"));
    if (args.has("subtalker-top-p")) p.subtalker_top_p = std::stof(args.get("subtalker-top-p"));
    if (args.has("max-new-tokens")) p.max_new_tokens = std::stoi(args.get("max-new-tokens"));
    if (args.has("chunk-frames")) p.chunk_frames = std::stoi(args.get("chunk-frames"));
    if (args.has("seed")) {
        p.has_seed = 1;
        p.seed = static_cast<uint32_t>(std::stoul(args.get("seed")));
    }
    if (args.has("greedy")) p.do_sample = p.subtalker_do_sample = 0;
    if (args.has("fixed-chunks")) p.chunk_grow = 0;
    return p;
}

void write_text(const Args & args, const std::string & text) {
    if (args.has("out")) {
        std::ofstream(std::filesystem::u8path(args.get("out"))) << text << "\n";
    } else {
        std::cout << text << std::endl;
    }
}

int cmd_info(const Args & args) {
    if (!args.has("model")) throw std::runtime_error("--model is required");
    StringPtr out(lure_model_info_offline(args.cstr("model")));
    check(out != nullptr, "lure_model_info_offline");
    write_text(args, out.get());
    return 0;
}

int cmd_dump_prefill(const Args & args) {
    if (!args.has("text") || !args.has("model")) throw std::runtime_error("--model and --text are required");
    StringPtr out;
    if (args.has("full")) {
        auto tts = load_tts(args);
        VoicePtr voice = wants_voice(args) ? load_voice(tts.get(), args) : VoicePtr();
        const auto request = request_from_args(args, voice.get(), wants_voice(args));
        out.reset(lure_dump_prefill(tts.get(), &request));
    } else {
        // Offline: a symbolic voice; --ref-text (icl) or --x-vector-only stand for the clone modes.
        const bool clone = args.has("ref-text") || args.has("x-vector-only");
        const auto request = request_from_args(args, nullptr, clone);
        out.reset(lure_dump_prefill_offline(args.cstr("model"), &request, args.cstr("ref-text"),
                                              std::stoi(args.get("ref-frames", "1"))));
    }
    check(out != nullptr, "dump_prefill");
    write_text(args, out.get());
    return 0;
}

int cmd_make_voice(const Args & args) {
    if (!args.has("out")) throw std::runtime_error("--out is required");
    auto tts = load_tts(args);
    auto voice = load_voice(tts.get(), args);
    check(lure_voice_save(voice.get(), args.cstr("out")) == 0, "lure_voice_save");
    std::cerr << "voice: " << lure_voice_frames(voice.get()) << " reference frames"
              << (lure_voice_has_transcript(voice.get()) ? "" : ", x-vector only") << " -> " << args.get("out") << "\n";
    return 0;
}

int cmd_generate(const Args & args) {
    if (!args.has("text")) throw std::runtime_error("--text is required");
    auto tts = load_tts(args);
    VoicePtr voice = wants_voice(args) ? load_voice(tts.get(), args) : VoicePtr();
    const auto request = request_from_args(args, voice.get(), wants_voice(args));
    const auto params = params_from_args(args);
    std::vector<float> audio;
    lure_gen_info info{};
    const int32_t rc = lure_generate(
        tts.get(), &request, &params,
        [](const float * pcm, size_t n, int32_t, void * user) -> int {
            auto * out = static_cast<std::vector<float> *>(user);
            out->insert(out->end(), pcm, pcm + n);
            return 1;
        },
        &audio, &info);
    check(rc == 0, "lure_generate");
    if (args.has("out")) {
        engine::audio::write_pcm16_wav(std::filesystem::u8path(args.get("out")), info.sample_rate, 1, audio);
    }
    std::cerr << "{\"frames\":" << info.frames << ",\"ended_by_eos\":" << info.ended_by_eos
              << ",\"audio_s\":" << static_cast<double>(info.samples) / std::max(1, info.sample_rate)
              << ",\"first_chunk_ms\":" << info.first_chunk_ms << ",\"talker_ms\":" << info.talker_ms
              << ",\"decoder_ms\":" << info.decoder_ms << ",\"total_ms\":" << info.total_ms << "}\n";
    return 0;
}

// ---- HTTP server ----
// GET  /health       -> {"ok": true, "sample_rate": 24000, "build": "..."}
// GET  /v1/model     -> lure_model_info JSON (variant, speakers, languages, modes, ...)
// POST /v1/voices    {"ref_wav": path | "ref_pcm_f32_b64": b64, "sample_rate": 24000, "ref_text"?: "..."}
//                    -> {"voice_id": "...", "frames": N, "transcript": bool}  (Base; no ref_text = x-vector only)
// POST /v1/generate  {"text", "voice"?, "speaker"?, "speaker_id"?, "instruct"?, "language"?, "layout"?,
//                     "x_vector_only"?, "seed"?, "temperature"?, "top_k"?, "top_p"?, "repetition_penalty"?,
//                     "subtalker_*"?, "max_new_tokens"?, "chunk_frames"?, "chunk_grow"?}
//                    "voice" (alias "voice_id") names a voice from /v1/voices (icl, or xvec without a transcript
//                    or with x_vector_only) or a preset speaker of a CustomVoice model (custom_voice). Neither
//                    runs voice_design, where "instruct" describes the voice.
//                    -> chunked application/x-lure-pcm: repeated [u32 LE n][n x f32 LE PCM], then [u32 0]
//                       followed by one JSON metadata object (frames, ended_by_eos, timings) or {"error"}.
// POST /v1/prefill   same body as generate -> prefill dump JSON.

std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    uint32_t buffer = 0;
    int bits = 0;
    for (const char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        const auto pos = chars.find(c);
        if (pos == std::string::npos) throw std::runtime_error("invalid base64");
        buffer = (buffer << 6) | static_cast<uint32_t>(pos);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

std::string json_str(const json::Value & body, const char * key, const std::string & fallback = "") {
    const auto * v = body.find(key);
    return v != nullptr && v->is_string() ? v->as_string() : fallback;
}

lure_params params_from_json(const json::Value & body) {
    lure_params p;
    lure_default_params(&p);
    auto f = [&](const char * key, float & dst) { if (const auto * v = body.find(key); v && v->is_number()) dst = v->as_f32(); };
    auto i = [&](const char * key, int32_t & dst) { if (const auto * v = body.find(key); v && v->is_number()) dst = static_cast<int32_t>(v->as_i64()); };
    f("temperature", p.temperature);
    i("top_k", p.top_k);
    f("top_p", p.top_p);
    f("repetition_penalty", p.repetition_penalty);
    f("subtalker_temperature", p.subtalker_temperature);
    i("subtalker_top_k", p.subtalker_top_k);
    f("subtalker_top_p", p.subtalker_top_p);
    i("max_new_tokens", p.max_new_tokens);
    i("chunk_frames", p.chunk_frames);
    if (const auto * v = body.find("do_sample"); v && v->is_bool()) p.do_sample = v->as_bool();
    if (const auto * v = body.find("subtalker_do_sample"); v && v->is_bool()) p.subtalker_do_sample = v->as_bool();
    if (const auto * v = body.find("chunk_grow"); v && v->is_bool()) p.chunk_grow = v->as_bool();
    if (const auto * v = body.find("seed"); v && v->is_number()) {
        p.has_seed = 1;
        p.seed = static_cast<uint32_t>(v->as_i64());
    }
    return p;
}

bool json_bool(const json::Value & body, const char * key) {
    const auto * v = body.find(key);
    return v != nullptr && v->is_bool() && v->as_bool();
}

// A lure_request plus the strings it points into.
struct ServerRequest {
    std::string text, instruct, language, speaker;
    lure_request request{};
};

int cmd_serve(const Args & args) {
    auto tts = load_tts(args);
    const std::string cache_dir = args.get("voice-cache", "");
    std::vector<std::string> speakers;
    for (int32_t k = 0; k < lure_speaker_count(tts.get()); ++k) speakers.emplace_back(lure_speaker_name(tts.get(), k));
    std::mutex voices_mutex;
    std::map<std::string, VoicePtr> voices;
    httplib::Server server;
    auto error = [](httplib::Response & res, int status, const std::string & message) {
        res.status = status;
        res.set_content("{\"error\":" + json::stringify_string(message) + "}", "application/json");
    };
    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"ok\":true,\"sample_rate\":" + std::to_string(lure_sample_rate(tts.get())) +
                        ",\"build\":" + json::stringify_string(lure_build_info()) + "}", "application/json");
    });
    server.Get("/v1/model", [&](const httplib::Request &, httplib::Response & res) {
        StringPtr info(lure_model_info(tts.get()));
        if (!info) return error(res, 500, lure_last_error());
        res.set_content(info.get(), "application/json");
    });
    server.Post("/v1/voices", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const auto body = json::parse(req.body);
            const auto ref_text = json_str(body, "ref_text");
            std::vector<float> pcm;
            int32_t sr = 24000;
            if (const auto path = json_str(body, "ref_wav"); !path.empty()) {
                pcm = read_mono(path, sr);
            } else {
                const auto bytes = base64_decode(json_str(body, "ref_pcm_f32_b64"));
                pcm.resize(bytes.size() / sizeof(float));
                std::memcpy(pcm.data(), bytes.data(), pcm.size() * sizeof(float));
                if (const auto * v = body.find("sample_rate"); v && v->is_number()) sr = static_cast<int32_t>(v->as_i64());
            }
            if (pcm.empty()) return error(res, 400, "ref audio is required");
            VoicePtr voice(lure_voice_cached(tts.get(), pcm.data(), pcm.size(), sr, ref_text.c_str(),
                                               cache_dir.empty() ? nullptr : cache_dir.c_str()));
            if (!voice) return error(res, 400, lure_last_error());
            const int32_t frames = lure_voice_frames(voice.get());
            const bool transcript = lure_voice_has_transcript(voice.get()) != 0;
            std::string id;
            {
                std::lock_guard<std::mutex> lock(voices_mutex);
                id = "v" + std::to_string(voices.size() + 1);
                voices[id] = std::move(voice);
            }
            res.set_content("{\"voice_id\":" + json::stringify_string(id) + ",\"frames\":" + std::to_string(frames) +
                            ",\"transcript\":" + (transcript ? "true" : "false") + "}", "application/json");
        } catch (const std::exception & e) {
            error(res, 400, e.what());
        }
    });
    auto find_voice = [&](const std::string & id) -> lure_voice * {
        std::lock_guard<std::mutex> lock(voices_mutex);
        const auto it = voices.find(id);
        return it == voices.end() ? nullptr : it->second.get();
    };
    auto is_speaker = [&](std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        return std::find(speakers.begin(), speakers.end(), name) != speakers.end();
    };
    // Throws std::invalid_argument (-> 404) for a voice that is neither a cached voice nor a preset speaker.
    auto request_from_json = [&](const json::Value & body, ServerRequest & out) {
        out.text = json_str(body, "text");
        out.instruct = json_str(body, "instruct");
        out.language = json_str(body, "language");
        out.speaker = json_str(body, "speaker");
        lure_request & r = out.request;
        lure_default_request(&r);
        r.text = out.text.c_str();
        r.instruct = out.instruct.empty() ? nullptr : out.instruct.c_str();
        r.language = out.language.empty() ? nullptr : out.language.c_str();
        r.layout = parse_layout(json_str(body, "layout"));
        const std::string voice_name = json_str(body, "voice", json_str(body, "voice_id"));
        if (const auto * v = body.find("speaker_id"); v && v->is_number()) r.speaker_id = static_cast<int32_t>(v->as_i64());
        if (!voice_name.empty() && out.speaker.empty() && r.speaker_id < 0) {
            if (auto * voice = find_voice(voice_name)) {
                const bool transcript = lure_voice_has_transcript(voice) != 0;
                r.mode = transcript && !json_bool(body, "x_vector_only") ? LURE_MODE_ICL : LURE_MODE_XVEC;
                r.voice = voice;
                return;
            }
            if (!is_speaker(voice_name)) throw std::invalid_argument("unknown voice: " + voice_name);
            out.speaker = voice_name;
        } else if (!voice_name.empty()) {
            throw std::runtime_error("pass voice (a cached voice or a preset speaker) or speaker, not both");
        }
        if (!out.speaker.empty() || r.speaker_id >= 0) {
            r.mode = LURE_MODE_CUSTOM_VOICE;
            r.speaker = out.speaker.empty() ? nullptr : out.speaker.c_str();
        } else {
            r.mode = LURE_MODE_VOICE_DESIGN;
        }
    };
    server.Post("/v1/prefill", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            const auto body = json::parse(req.body);
            ServerRequest r;
            request_from_json(body, r);
            StringPtr out(lure_dump_prefill(tts.get(), &r.request));
            if (!out) return error(res, 400, lure_last_error());
            res.set_content(out.get(), "application/json");
        } catch (const std::invalid_argument & e) {
            error(res, 404, e.what());
        } catch (const std::exception & e) {
            error(res, 400, e.what());
        }
    });
    server.Post("/v1/generate", [&](const httplib::Request & req, httplib::Response & res) {
        auto r = std::make_shared<ServerRequest>();
        lure_params p{};
        try {
            const auto body = json::parse(req.body);
            request_from_json(body, *r);
            p = params_from_json(body);
        } catch (const std::invalid_argument & e) {
            return error(res, 404, e.what());
        } catch (const std::exception & e) {
            return error(res, 400, e.what());
        }
        if (r->text.empty()) return error(res, 400, "text is required");
        res.set_chunked_content_provider("application/x-lure-pcm", [&, r, p](size_t, httplib::DataSink & sink) {
            lure_gen_info info{};
            const int32_t rc = lure_generate(
                tts.get(), &r->request, &p,
                [](const float * pcm, size_t n, int32_t, void * user) -> int {
                    auto * out = static_cast<httplib::DataSink *>(user);
                    const uint32_t count = static_cast<uint32_t>(n);
                    if (!out->is_writable()) return 0;
                    out->write(reinterpret_cast<const char *>(&count), sizeof(count));
                    return out->write(reinterpret_cast<const char *>(pcm), n * sizeof(float)) ? 1 : 0;
                },
                &sink, &info);
            const uint32_t zero = 0;
            sink.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
            const std::string meta = rc == 0
                ? "{\"frames\":" + std::to_string(info.frames) + ",\"ended_by_eos\":" + (info.ended_by_eos ? "true" : "false") +
                  ",\"cancelled\":" + (info.cancelled ? "true" : "false") + ",\"samples\":" + std::to_string(info.samples) +
                  ",\"sample_rate\":" + std::to_string(info.sample_rate) + ",\"first_chunk_ms\":" + json::stringify_number(info.first_chunk_ms) +
                  ",\"talker_ms\":" + json::stringify_number(info.talker_ms) + ",\"decoder_ms\":" + json::stringify_number(info.decoder_ms) +
                  ",\"total_ms\":" + json::stringify_number(info.total_ms) + "}"
                : "{\"error\":" + json::stringify_string(lure_last_error()) + "}";
            sink.write(meta.data(), meta.size());
            sink.done();
            return true;
        });
    });
    const std::string host = args.get("host", "127.0.0.1");
    const int port = std::stoi(args.get("port", "8088"));
    std::cerr << "lure_cli serve on http://" << host << ":" << port << " (" << lure_variant(tts.get()) << ", "
              << lure_build_info() << ")\n";
    // One generation at a time: the handle serializes internally; keep the pool small.
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    return server.listen(host, port) ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (args.command == "info") return cmd_info(args);
        if (args.command == "dump-prefill") return cmd_dump_prefill(args);
        if (args.command == "make-voice") return cmd_make_voice(args);
        if (args.command == "generate") return cmd_generate(args);
        if (args.command == "serve") return cmd_serve(args);
        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
