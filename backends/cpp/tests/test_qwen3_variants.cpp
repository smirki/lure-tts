// Loader checks for every stock Qwen3-TTS 12Hz config, without weights.
//
//   lure_qwen3_variants_test <conformance/reference>
//
// For each configs/*.json: build a prompt-only folder (config + reference tokenizer), load its metadata, then run
// check_qwen3_tts_weights against a fake tensor source holding only the shapes that config implies. A correct set
// must pass; dropping or reshaping a variant-specific tensor (small_to_mtp_projection on 1.7B, the speaker encoder
// on Base) must fail.
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/qwen3_tts/assets.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using engine::assets::RawTensorData;
using engine::assets::TensorMetadata;
using engine::models::qwen3_tts::Qwen3TTSConfig;
using engine::models::qwen3_tts::Qwen3TTSVariant;

namespace {

class ShapeOnlySource final : public engine::assets::TensorSource {
public:
    std::map<std::string, std::vector<int64_t>> shapes;
    fs::path path = "shape-only";

    const fs::path & source_path() const noexcept override { return path; }
    bool has_tensor(std::string_view name) const noexcept override { return shapes.count(std::string(name)) != 0; }
    TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = shapes.find(std::string(name));
        if (it == shapes.end()) throw std::runtime_error("missing " + std::string(name));
        return {it->first, "F32", it->second};
    }
    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        for (const auto & [name, shape] : shapes) out.push_back({name, "F32", shape});
        return out;
    }
    RawTensorData require_tensor_data(std::string_view) const override { throw std::runtime_error("no data"); }
    std::vector<float> require_f32(std::string_view, const std::optional<std::vector<int64_t>> &) const override {
        throw std::runtime_error("no data");
    }
    std::optional<std::vector<float>> optional_f32(std::string_view, const std::optional<std::vector<int64_t>> &) const override {
        return std::nullopt;
    }
    int64_t require_i64_scalar(std::string_view) const override { throw std::runtime_error("no data"); }
};

ShapeOnlySource shapes_for(const Qwen3TTSConfig & c) {
    ShapeOnlySource s;
    const auto & t = c.talker;
    const auto & cp = c.code_predictor;
    s.shapes["talker.model.text_embedding.weight"] = {t.text_vocab_size, t.text_hidden_size};
    s.shapes["talker.text_projection.linear_fc2.weight"] = {t.hidden_size, t.text_hidden_size};
    s.shapes["talker.model.codec_embedding.weight"] = {t.vocab_size, t.hidden_size};
    s.shapes["talker.codec_head.weight"] = {t.vocab_size, t.hidden_size};
    s.shapes["talker.model.layers.0.mlp.gate_proj.weight"] = {t.intermediate_size, t.hidden_size};
    for (int64_t g = 0; g + 1 < t.num_code_groups; ++g) {
        s.shapes["talker.code_predictor.model.codec_embedding." + std::to_string(g) + ".weight"] = {cp.vocab_size, t.hidden_size};
    }
    s.shapes["talker.code_predictor.model.layers.0.mlp.gate_proj.weight"] = {cp.intermediate_size, cp.hidden_size};
    if (cp.hidden_size != t.hidden_size) {
        s.shapes["talker.code_predictor.small_to_mtp_projection.weight"] = {cp.hidden_size, t.hidden_size};
        s.shapes["talker.code_predictor.small_to_mtp_projection.bias"] = {cp.hidden_size};
    }
    if (c.variant == Qwen3TTSVariant::Base) {
        s.shapes["speaker_encoder.fc.weight"] = {c.speaker_encoder.embedding_dim, 1536, 1};
    }
    return s;
}

int failures = 0;

void expect(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <conformance/reference>\n", argv[0]);
        return 2;
    }
    const fs::path reference = fs::absolute(argv[1]);
    const fs::path root = fs::temp_directory_path() / "lure_qwen3_variants_test";
    fs::remove_all(root);
    try {
        for (const auto & entry : fs::directory_iterator(reference / "configs")) {
            const std::string name = entry.path().stem().string();
            const fs::path dir = root / name;
            fs::create_directories(dir);
            for (const char * f : {"vocab.json", "merges.txt", "tokenizer_config.json"}) {
                fs::create_symlink(reference / "tokenizer" / f, dir / f);
            }
            fs::create_symlink(entry.path(), dir / "config.json");
            const auto assets = engine::models::qwen3_tts::load_qwen3_tts_assets_metadata(dir);
            const auto & c = assets->config;
            const bool big = name.find("1.7B") != std::string::npos;
            expect(c.talker.hidden_size == (big ? 2048 : 1024), name + ": talker hidden size");
            expect(c.code_predictor.hidden_size == 1024, name + ": code predictor hidden size");
            const bool base = name.find("Base") != std::string::npos;
            expect((c.variant == Qwen3TTSVariant::Base) == base, name + ": variant");
            expect(base || !c.talker.speaker_id.empty() || name.find("VoiceDesign") != std::string::npos,
                   name + ": preset speakers match the variant");

            auto good = shapes_for(c);
            const auto errors = engine::models::qwen3_tts::check_qwen3_tts_weights(c, good);
            for (const auto & e : errors) std::printf("     %s\n", e.c_str());
            expect(errors.empty(), name + ": complete weights pass");

            auto bad = good;
            bad.shapes["talker.model.codec_embedding.weight"] = {c.talker.vocab_size, big ? 1024 : 2048};
            expect(!engine::models::qwen3_tts::check_qwen3_tts_weights(c, bad).empty(), name + ": wrong talker width fails");
            if (big) {
                auto no_proj = good;
                no_proj.shapes.erase("talker.code_predictor.small_to_mtp_projection.weight");
                expect(!engine::models::qwen3_tts::check_qwen3_tts_weights(c, no_proj).empty(),
                       name + ": missing small_to_mtp_projection fails");
            }
            if (base) {
                auto no_spk = good;
                no_spk.shapes.erase("speaker_encoder.fc.weight");
                expect(!engine::models::qwen3_tts::check_qwen3_tts_weights(c, no_spk).empty(),
                       name + ": Base without a speaker encoder fails");
            }
        }
    } catch (const std::exception & e) {
        std::printf("FAIL %s\n", e.what());
        ++failures;
    }
    fs::remove_all(root);
    std::printf("%s\n", failures == 0 ? "all passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
