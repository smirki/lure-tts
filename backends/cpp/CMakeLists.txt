cmake_minimum_required(VERSION 3.21)

project(LureCpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Everything ends up in the shared liblure_tts, so all static code must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(ENGINE_DEFAULT_ENABLE_METAL OFF)
if (APPLE)
    set(ENGINE_DEFAULT_ENABLE_METAL ON)
endif()
option(ENGINE_ENABLE_CUDA "Build ggml with the CUDA backend" OFF)
option(ENGINE_ENABLE_VULKAN "Build ggml with the Vulkan backend" OFF)
option(ENGINE_ENABLE_METAL "Build ggml with the Metal backend" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM" ON)
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs" ON)
option(ENGINE_ENABLE_NATIVE_CPU "Build ggml CPU kernels for the host ISA (-march=native)" ON)
option(ENGINE_ENABLE_CPU_ALL_VARIANTS
    "Build ggml CPU kernels as per-ISA shared libraries selected at runtime (portable binaries)" OFF)
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP" ON)
# Rockchip NPU: runs a precompiled .rknn speech decoder through librknnrt instead of ggml.
option(ENGINE_ENABLE_RKNN "Build the Rockchip NPU (librknnrt) speech decoder path" OFF)
set(ENGINE_RKNN_INCLUDE_DIR "" CACHE PATH "Directory containing rknn_api.h (empty: system include path)")
set(ENGINE_RKNN_LIBRARY "rknnrt" CACHE STRING "librknnrt to link (path or library name)")
option(AUDIOCPP_DEPLOYMENT_BUILD "Compile the model spec into the runtime" ON)
option(LURE_METAL_AOT "Precompile Metal libraries next to the binaries (Apple only)" OFF)
option(LURE_BUILD_TESTS "Build the Turbo head unit test" ON)
set(AUDIOCPP_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml" CACHE PATH "ggml source tree")

set(LURE_GIT_SHA "unknown")
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE LURE_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (LURE_GIT_SHA STREQUAL "")
        set(LURE_GIT_SHA "unknown")
    endif()
endif()

# ---------------------------------------------------------------------------
# Model spec: embedded as a C++ string literal
# ---------------------------------------------------------------------------
# MSVC caps one string literal at 16380 bytes (C2026) and qwen3_tts.json is larger,
# so the spec is split into adjacent raw literals at newline boundaries.
function(audiocpp_model_spec_literal out json)
    set(limit 16000)
    set(rest "${json}")
    set(literals "")
    while(TRUE)
        string(LENGTH "${rest}" len)
        if (len LESS_EQUAL limit)
            string(APPEND literals "R\"AUDIOCPP_SPEC(${rest})AUDIOCPP_SPEC\"")
            break()
        endif()
        string(SUBSTRING "${rest}" 0 ${limit} head)
        string(FIND "${head}" "\n" cut REVERSE)
        if (cut LESS 1)
            set(cut ${limit})
        endif()
        string(SUBSTRING "${rest}" 0 ${cut} chunk)
        string(SUBSTRING "${rest}" ${cut} -1 rest)
        string(APPEND literals "R\"AUDIOCPP_SPEC(${chunk})AUDIOCPP_SPEC\" ")
    endwhile()
    set(${out} "${literals}" PARENT_SCOPE)
endfunction()

set(LURE_MODEL_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/model_specs/qwen3_tts.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LURE_MODEL_SPEC}")
file(READ "${LURE_MODEL_SPEC}" LURE_MODEL_SPEC_JSON)
audiocpp_model_spec_literal(LURE_MODEL_SPEC_LITERAL "${LURE_MODEL_SPEC_JSON}")
set(LURE_MODEL_SPEC_ENTRY "    {\"qwen3_tts\", ${LURE_MODEL_SPEC_LITERAL}},\n")
set(LURE_BUILTIN_SPEC_ENTRY "")
if (AUDIOCPP_DEPLOYMENT_BUILD)
    set(LURE_BUILTIN_SPEC_ENTRY "${LURE_MODEL_SPEC_ENTRY}")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_specs.inc"
    CONTENT "${LURE_BUILTIN_SPEC_ENTRY}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gguf_converter_model_specs.inc"
    CONTENT "${LURE_MODEL_SPEC_ENTRY}")

# ---------------------------------------------------------------------------
# CUDA architectures
# ---------------------------------------------------------------------------
if (ENGINE_ENABLE_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    # Must be set before enable_language(CUDA), which otherwise seeds nvcc's single default arch.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=... (or "native" for the local GPU only).
    if ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "" AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-virtual 86-real 89-real 90-virtual)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            # 12Xa, not 12X: Blackwell FP4 tensor cores are arch-specific.
            list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
        endif()
    endif()
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# ggml and vendored libraries
# ---------------------------------------------------------------------------
if (ENGINE_ENABLE_CPU_ALL_VARIANTS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Native host ISA flags" FORCE)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE ${ENGINE_ENABLE_NATIVE_CPU} CACHE BOOL "Native host ISA flags" FORCE)
endif()
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${ENGINE_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${ENGINE_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_METAL ${ENGINE_ENABLE_METAL} CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ${ENGINE_ENABLE_LLAMAFILE} CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "" FORCE)

if (NOT EXISTS "${AUDIOCPP_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AUDIOCPP_GGML_SOURCE_DIR does not point to a ggml source tree: ${AUDIOCPP_GGML_SOURCE_DIR}")
endif()
add_subdirectory("${AUDIOCPP_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

set(AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING=\"0.2.5\"
)

add_library(cjson_vendor STATIC external/cJSON/cJSON.c)
target_include_directories(cjson_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON)
# Without this, cJSON marks every function dllexport on Windows and lure_tts.dll re-exports them.
target_compile_definitions(cjson_vendor PUBLIC CJSON_HIDE_SYMBOLS)

add_library(yaml_vendor STATIC
    external/libyaml/src/api.c
    external/libyaml/src/reader.c
    external/libyaml/src/scanner.c
    external/libyaml/src/parser.c
    external/libyaml/src/loader.c
)
target_include_directories(yaml_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include)
target_compile_definitions(yaml_vendor PUBLIC ${AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS})

add_library(utf8proc_vendor STATIC external/utf8proc/utf8proc.c)
target_include_directories(utf8proc_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/utf8proc)
target_compile_definitions(utf8proc_vendor PUBLIC UTF8PROC_STATIC)

# Plain HTTP only: compile cpp-httplib directly (its own CMakeLists fetches a TLS library).
add_library(lure_httplib STATIC external/cpp-httplib/httplib.cpp)
target_include_directories(lure_httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib)
target_link_libraries(lure_httplib PUBLIC Threads::Threads)
if (WIN32)
    target_link_libraries(lure_httplib PUBLIC ws2_32)
endif()
target_compile_options(lure_httplib PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
)

# ---------------------------------------------------------------------------
# engine_runtime: framework + Qwen3-TTS model
# ---------------------------------------------------------------------------
add_library(engine_runtime STATIC
    src/framework/assets/resource_bundle.cpp
    src/framework/model_spec/package.cpp
    src/framework/model_spec/options.cpp
    src/framework/model_spec/schema.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/weight_norm_folding.cpp
    src/framework/assets/weight_adapter.cpp
    src/framework/assets/weight_adapter_source.cpp
    src/framework/assets/torch_bin.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/attention_fallback.cpp
    src/framework/core/execution_context.cpp
    src/framework/core/host_memory.cpp
    src/framework/core/load_progress.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/task_vocabulary.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session_base.cpp
    src/framework/io/filesystem.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chunking.cpp
    src/framework/io/yaml.cpp
    src/framework/io/binary.cpp
    src/framework/io/safetensors.cpp
    src/framework/debug/trace.cpp
    src/framework/debug/profiler.cpp
    src/framework/rknn/engine.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/packed_linear_weights.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/attention/scaled_dot_product_attention.cpp
    src/framework/modules/attention/streaming_frame_attention_cache.cpp
    src/framework/modules/transformers/decoder.cpp
    src/framework/modules/transformers/causal_decoder.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/llama_bpe.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/audio/fft.cpp
    src/framework/audio/conversion.cpp
    src/framework/audio/dsp.cpp
    src/framework/audio/mel_spectrogram_frontend.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/waveform_ops.cpp
    src/framework/audio/wav_reader.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/sampling/hf_sampler.cpp
    src/framework/sampling/torch_random.cpp
    src/models/qwen3_tts/assets.cpp
    src/models/qwen3_tts/tokenizer_text.cpp
    src/models/qwen3_tts/tokenizer_speech_encoder.cpp
    src/models/qwen3_tts/tokenizer_speech_decoder.cpp
    src/models/qwen3_tts/rknn_speech_decoder.cpp
    src/models/qwen3_tts/speaker_encoder.cpp
    src/models/qwen3_tts/talker.cpp
    src/models/qwen3_tts/turbo_head.cpp
    src/models/qwen3_tts/prompt_tts_voice_clone.cpp
    src/models/qwen3_tts/prompt_tts_voice_design.cpp
    src/models/qwen3_tts/prompt_tts_custom_voice.cpp
    src/models/qwen3_tts/prompt_segment.cpp
    src/models/qwen3_tts/session.cpp
)
target_include_directories(engine_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(engine_runtime PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
)
target_link_libraries(engine_runtime PUBLIC ggml)
target_link_libraries(engine_runtime PRIVATE
    Threads::Threads ${CMAKE_DL_LIBS} cjson_vendor yaml_vendor utf8proc_vendor)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(engine_runtime PRIVATE OpenMP::OpenMP_CXX)
    if (MSVC)
        # MSVC's default /openmp is OpenMP 2.0 and rejects '#pragma omp simd'.
        target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
    endif()
endif()
if (ENGINE_ENABLE_CUDA)
    # GPU-side sampling helpers used by the talker's CUDA sampling policy.
    target_sources(engine_runtime PRIVATE src/framework/sampling/torch_random_cuda_runtime.cu)
    set_source_files_properties(src/framework/sampling/torch_random_cuda_runtime.cu PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart)
endif()
if (ENGINE_ENABLE_RKNN)
    target_compile_definitions(engine_runtime PRIVATE ENGINE_HAVE_RKNN=1)
    if (ENGINE_RKNN_INCLUDE_DIR)
        target_include_directories(engine_runtime PUBLIC ${ENGINE_RKNN_INCLUDE_DIR})
    endif()
    target_link_libraries(engine_runtime PUBLIC ${ENGINE_RKNN_LIBRARY})
endif()

function(lure_link_openmp target)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# Windows decodes narrow strings with the legacy code page unless the process asks for UTF-8.
function(lure_use_utf8_codepage target)
    if (MSVC)
        target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/audiocpp_utf8.manifest")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
add_library(lure_tts SHARED src/lure/lure_tts.cpp)
target_link_libraries(lure_tts PRIVATE engine_runtime ggml)
target_include_directories(lure_tts PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(lure_tts PRIVATE LURE_BUILD_DLL LURE_VERSION_STRING="${LURE_GIT_SHA}")
set_target_properties(lure_tts PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if (APPLE)
    set_target_properties(lure_tts PROPERTIES INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH ON)
    target_link_options(lure_tts PRIVATE
        "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/lure/lure_tts.symbols")
elseif (UNIX)
    set_target_properties(lure_tts PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
    target_link_options(lure_tts PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/lure/lure_tts.map"
        "-Wl,--exclude-libs,ALL")
endif()
lure_link_openmp(lure_tts)

# The CLI compiles the C API in statically: the WAV and JSON helpers it uses are not exported.
add_executable(lure_cli app/lure/lure_cli.cpp src/lure/lure_tts.cpp)
target_compile_definitions(lure_cli PRIVATE LURE_VERSION_STRING="${LURE_GIT_SHA}")
target_link_libraries(lure_cli PRIVATE engine_runtime ggml lure_httplib)
lure_use_utf8_codepage(lure_cli)
lure_link_openmp(lure_cli)

add_executable(lure_gguf app/gguf/main.cpp)
target_include_directories(lure_gguf PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(lure_gguf PRIVATE engine_runtime ggml)
lure_use_utf8_codepage(lure_gguf)
lure_link_openmp(lure_gguf)

if (LURE_BUILD_TESTS)
    add_executable(lure_turbo_head_test tests/test_turbo_head.cpp)
    target_link_libraries(lure_turbo_head_test PRIVATE engine_runtime ggml)
    lure_link_openmp(lure_turbo_head_test)
    add_executable(lure_qwen3_variants_test tests/test_qwen3_variants.cpp)
    target_link_libraries(lure_qwen3_variants_test PRIVATE engine_runtime ggml)
    lure_link_openmp(lure_qwen3_variants_test)
endif()

# ---------------------------------------------------------------------------
# Precompiled Metal libraries
# ---------------------------------------------------------------------------
# The embedded-source Metal build compiles about 8 MB of shader source on the first launch of
# each new binary. These metallibs are built from the same generated source with the macros the
# runtime would set; ggml-metal-device.m loads the matching one from GGML_METAL_LIBRARY_DIR or
# the directory holding the ggml image, and falls back to the embedded source on any failure.
if (LURE_METAL_AOT AND APPLE AND ENGINE_ENABLE_METAL AND GGML_METAL_EMBED_LIBRARY)
    execute_process(COMMAND xcrun -sdk macosx -f metal RESULT_VARIABLE xcrun_metal_result
                    OUTPUT_QUIET ERROR_QUIET)
    if (NOT xcrun_metal_result EQUAL 0)
        message(WARNING "LURE_METAL_AOT: xcrun metal not found (install the Metal toolchain); skipping")
    else()
        set(metal_embed_src "${CMAKE_BINARY_DIR}/ggml/src/ggml-metal/autogenerated/ggml-metal-embed.metal")
        set(metal_outputs "")
        foreach(variant IN ITEMS bf16 bf16-tensor)
            set(flags -O3 -DGGML_METAL_EMBED_LIBRARY=1 -DGGML_METAL_HAS_BF16=1)
            if (variant STREQUAL "bf16-tensor")
                list(APPEND flags -DGGML_METAL_HAS_TENSOR=1)
            endif()
            set(out "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal-${variant}.metallib")
            add_custom_command(
                OUTPUT "${out}"
                COMMAND xcrun -sdk macosx metal ${flags} -c "${metal_embed_src}" -o "${CMAKE_BINARY_DIR}/ggml-metal-${variant}.air"
                COMMAND xcrun -sdk macosx metallib "${CMAKE_BINARY_DIR}/ggml-metal-${variant}.air" -o "${out}"
                DEPENDS ggml-metal "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml/src/ggml-metal/ggml-metal.metal"
                COMMENT "Precompiling Metal library (${variant})")
            list(APPEND metal_outputs "${out}")
        endforeach()
        add_custom_target(lure_metal_aot DEPENDS ${metal_outputs})
        add_dependencies(lure_tts lure_metal_aot)
    endif()
endif()
