// Turbo head vs the shared reference fixture (conformance/turbo/fixtures/tiny_turbo.safetensors: tiny random
// weights plus expected logits and greedy codes, made by conformance/turbo/make_fixture.py from ref_fastcp.py).
// Checks forced-path logits and greedy codes.
//
//   lure_turbo_head_test <tiny_turbo.safetensors> [cpu|metal|cuda|vulkan] [head.gguf]
//     with head.gguf: head tensors from its "turbo_head_weights" namespace (lure_gguf --hf-dir output) instead
//   lure_turbo_head_test --resolve <model dir or .gguf> [k=v;...]
//     prints what apply_qwen3_tts_turbo selects (config + tensor headers only, no weights are read)

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/turbo_head.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace engine;
using engine::models::qwen3_tts::Qwen3TurboHead;
using engine::models::qwen3_tts::Qwen3TurboHeadInit;

namespace {

std::vector<int32_t> read_i32(const assets::TensorSource & src, const std::string & name) {
    const auto raw = src.require_tensor_data(name);
    std::vector<int32_t> out(raw.bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), raw.bytes.data(), out.size() * sizeof(int32_t));
    return out;
}

int argmax(const std::vector<float> & v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

int resolve(const char * model_path, const char * options_text) {
    std::unordered_map<std::string, std::string> options;
    std::stringstream stream(options_text != nullptr ? options_text : "");
    std::string item;
    while (std::getline(stream, item, ';')) {
        const auto eq = item.find('=');
        if (eq != std::string::npos) options[item.substr(0, eq)] = item.substr(eq + 1);
    }
    try {
        const auto assets = models::qwen3_tts::apply_qwen3_tts_turbo(
            models::qwen3_tts::load_qwen3_tts_assets(std::filesystem::u8path(model_path)), options);
        if (!assets->turbo) {
            std::printf("turbo: off\n");
            return 0;
        }
        const auto & t = *assets->turbo;
        std::printf("turbo: on head=%s groups=%zu addon_hidden=%lld has_mask=%d n_tensors=%zu\n", t.head_path.c_str(),
                    t.groups.size(), static_cast<long long>(t.addon_hidden),
                    assets->turbo_head_weights->has_tensor("mask") ? 1 : 0, assets->turbo_head_weights->tensors().size());
        return 0;
    } catch (const std::exception & e) {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--resolve") {
        return resolve(argv[2], argc > 3 ? argv[3] : nullptr);
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tiny_turbo.safetensors> [cpu|metal|cuda|vulkan]\n", argv[0]);
        return 2;
    }
    const std::string backend_name = argc > 2 ? argv[2] : "cpu";
    try {
        constexpr int64_t H = 64, V = 512, B = 3, K = 15;
        const auto src = assets::open_tensor_source(argv[1]);
        std::vector<assets::TensorData> cemb;
        for (int i = 0; i < 15; ++i) {
            cemb.push_back(src->require_tensor("cemb." + std::to_string(i) + ".weight", assets::TensorStorageType::F32, {V, H}));
        }
        const auto x2 = src->require_f32("x2", {B, 2, H});
        const auto greedy_codes = read_i32(*src, "greedy_codes");
        const auto forced_codes = read_i32(*src, "forced_codes");
        const auto greedy_logits = src->require_f32("greedy_logits", {B, K, V});
        const auto forced_logits = src->require_f32("forced_logits", {B, K, V});

        core::BackendConfig bc;
        bc.type = backend_name == "metal" ? core::BackendType::Metal
            : backend_name == "cuda"      ? core::BackendType::Cuda
            : backend_name == "vulkan"    ? core::BackendType::Vulkan
                                          : core::BackendType::Cpu;
        bc.threads = 2;
        ggml_backend_t backend = core::init_backend(bc);

        Qwen3TurboHeadInit init;
        init.dims = {H, 96, 2, 4, 2, 16, V, 1e-6F, 1000000.0F};
        init.turbo.hidden_size = H;
        init.turbo.num_layers = 2;
        init.turbo.addon_hidden = 512;
        init.head = src;
        init.key_prefix = "head.";
        if (argc > 3) {
            init.head = assets::open_tensor_source(argv[3], "turbo_head_weights");
            init.key_prefix.clear();
            std::printf("[%s] head tensors from %s:turbo_head_weights\n", backend_name.c_str(), argv[3]);
        }
        init.code_embeddings = &cemb;
        init.backend = backend;
        init.backend_type = bc.type;
        init.threads = 2;
        init.weight_storage_type = assets::TensorStorageType::F32;

        int failures = 0;
        {
            Qwen3TurboHead head(init);
            double worst_forced = 0.0, worst_greedy = 0.0;
            for (int64_t b = 0; b < B; ++b) {
                const std::vector<float> xb(x2.begin() + b * 2 * H, x2.begin() + (b + 1) * 2 * H);
                const auto check = [&](const std::vector<float> & ref, const std::vector<float> & got, int k, double & worst) {
                    double max_ref = 0.0, max_err = 0.0;
                    for (int64_t j = 0; j < V; ++j) {
                        const double r = ref[static_cast<size_t>((b * K + (k - 1)) * V + j)];
                        max_ref = std::max(max_ref, std::fabs(r));
                        max_err = std::max(max_err, std::fabs(r - got[static_cast<size_t>(j)]));
                    }
                    worst = std::max(worst, max_err / std::max(max_ref, 1e-6));
                };
                // forced path: feed the fixture's codes, compare every position's logits
                const auto fc = head.run(xb, [&](const std::vector<float> & logits, int k) {
                    check(forced_logits, logits, k, worst_forced);
                    return forced_codes[static_cast<size_t>(b * K + (k - 1))];
                });
                for (int64_t k = 0; k < K; ++k) {
                    if (fc[static_cast<size_t>(k)] != forced_codes[static_cast<size_t>(b * K + k)]) ++failures;
                }
                // greedy path: argmax everywhere, codes must match exactly
                const auto gc = head.run(xb, [&](const std::vector<float> & logits, int k) {
                    check(greedy_logits, logits, k, worst_greedy);
                    return argmax(logits);
                });
                for (int64_t k = 0; k < K; ++k) {
                    if (gc[static_cast<size_t>(k)] != greedy_codes[static_cast<size_t>(b * K + k)]) {
                        std::fprintf(stderr, "greedy mismatch b=%lld cb%lld: got %d want %d\n", static_cast<long long>(b),
                                     static_cast<long long>(k + 1), gc[static_cast<size_t>(k)],
                                     greedy_codes[static_cast<size_t>(b * K + k)]);
                        ++failures;
                    }
                }
            }
            // ggml-metal's kernel_mul_mm_f32_f32 stages tiles through half (multi-row matmuls: the group passes), so
            // GPU backends get a looser bound; greedy codes must still match exactly everywhere.
            const double tol = bc.type == core::BackendType::Cpu ? 1e-3 : 3e-3;
            std::printf("[%s] forced logits max rel err %.3g, greedy logits max rel err %.3g (tol %.0e)\n", backend_name.c_str(),
                        worst_forced, worst_greedy, tol);
            if (worst_forced > tol || worst_greedy > tol) ++failures;
            const auto & t = head.timing();
            std::printf("[%s] last frame: upload %.3f ms, compute %.3f ms, read %.3f ms\n", backend_name.c_str(), t.input_upload_ms,
                        t.graph_compute_ms, t.output_read_ms);
        }
        ggml_backend_free(backend);
        std::printf("[%s] %s\n", backend_name.c_str(), failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
