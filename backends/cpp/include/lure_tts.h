/*
 * lure_tts.h: C API of the Lure TTS C++ backend (built on audio.cpp and ggml).
 *
 *   lure_load(model_dir, ...)                      -> lure_tts
 *   lure_voice_create(tts, ref_pcm, ref_text)      -> lure_voice (Base checkpoints)
 *   lure_generate(tts, request, params, callback)  -> float32 PCM chunks at 24 kHz + lure_gen_info
 *
 * Runs every Qwen3-TTS 12Hz checkpoint (0.6B and 1.7B; Base, CustomVoice, VoiceDesign) with the prompt
 * of the Lure TTS core (core/lure_tts/prompt.py, docs/architecture.md):
 *
 *   mode          checkpoint    needs
 *   icl           Base          a voice with a reference transcript
 *   xvec          Base          a voice (only its x-vector is used)
 *   custom_voice  CustomVoice   a preset speaker (name or codec id); instruct optional on 1.7B
 *   voice_design  VoiceDesign   the instruct, which describes the voice
 *
 * and the non_streaming or streaming prompt layout. The instruct goes in as
 * `<|im_start|>user\n{instruct}<|im_end|>\n` before the role rows.
 *
 * Built for ctypes: plain C types, opaque handles, one callback type. Functions that can fail
 * return NULL or a negative value and set a thread-local message (lure_last_error()).
 * A handle is not thread-safe; serialize calls per lure_tts.
 */
#ifndef LURE_TTS_H
#define LURE_TTS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(LURE_BUILD_DLL)
#    define LURE_API __declspec(dllexport)
#  elif defined(LURE_USE_DLL)
#    define LURE_API __declspec(dllimport)
#  else
#    define LURE_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define LURE_API __attribute__((visibility("default")))
#else
#  define LURE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LURE_ABI_VERSION 2u

typedef struct lure_tts   lure_tts;
typedef struct lure_voice lure_voice;

/* Prompt modes. */
#define LURE_MODE_ICL          0
#define LURE_MODE_XVEC         1
#define LURE_MODE_CUSTOM_VOICE 2
#define LURE_MODE_VOICE_DESIGN 3

/* Prompt layouts. DEFAULT = the checkpoint's default: lure.json prompt_layout (non_streaming) for the
 * Lure family, streaming for stock Base, non_streaming for CustomVoice and VoiceDesign. */
#define LURE_LAYOUT_DEFAULT       (-1)
#define LURE_LAYOUT_NON_STREAMING 0
#define LURE_LAYOUT_STREAMING     1

/* What to say and how to prompt it. Always initialise with lure_default_request(). */
typedef struct lure_request {
    uint32_t struct_size;        /* = sizeof(lure_request) */
    int32_t  mode;               /* LURE_MODE_*; default ICL */
    int32_t  layout;             /* LURE_LAYOUT_*; default LURE_LAYOUT_DEFAULT */
    int32_t  speaker_id;         /* custom_voice: codec id from the config's spk_id, or -1 to use `speaker` */
    const char * text;           /* the segment, exactly as given (tags stay plain text) */
    const char * instruct;       /* NULL or "" = no instruct */
    const char * language;       /* codec language key ("english", ...) or "auto"; NULL = checkpoint default */
    const char * speaker;        /* custom_voice: preset name, case-insensitive */
    const lure_voice * voice;    /* icl and xvec */
} lure_request;

/* Sampling and streaming. Always initialise with lure_default_params(); struct_size guards ABI. */
typedef struct lure_params {
    uint32_t struct_size;            /* = sizeof(lure_params) */
    float    temperature;            /* 0.9  */
    int32_t  top_k;                  /* 50   */
    float    top_p;                  /* 1.0  */
    float    repetition_penalty;     /* 1.05 (talker codebook 0 only, as qwen-tts) */
    float    subtalker_temperature;  /* 0.9  (code predictor, qwen-tts defaults) */
    int32_t  subtalker_top_k;        /* 50   */
    float    subtalker_top_p;        /* 1.0  */
    int32_t  do_sample;              /* 1    */
    int32_t  subtalker_do_sample;    /* 1    */
    int32_t  max_new_tokens;         /* 0 = checkpoint generation_config.json (2048 when absent) */
    int32_t  has_seed;               /* 0 = random seed */
    uint32_t seed;
    int32_t  chunk_frames;           /* streaming chunk in 80 ms codec frames; 4 = 0.32 s. 0 = one chunk at the end */
    int32_t  chunk_grow;             /* 1 = 1,2,4,4,... frames (first audio ASAP); 0 = fixed chunk_frames */
} lure_params;

typedef struct lure_gen_info {
    int64_t frames;          /* generated codec frames (12.5 Hz) */
    int32_t ended_by_eos;    /* 1 = codec EOS, 0 = hit max_new_tokens or cancelled */
    int32_t cancelled;       /* 1 = callback returned 0 */
    int64_t samples;         /* PCM samples delivered through the callback */
    int32_t sample_rate;     /* 24000 */
    double  talker_ms;       /* talker + code predictor (excludes streamed decode) */
    double  decoder_ms;      /* codec decode */
    double  first_chunk_ms;  /* call start -> first PCM chunk handed to the callback */
    double  total_ms;
} lure_gen_info;

/* Receives mono float32 PCM. Return non-zero to continue, 0 to cancel. Runs on the calling thread. */
typedef int (*lure_pcm_callback)(const float * pcm, size_t n_samples, int32_t sample_rate, void * user);

LURE_API uint32_t     lure_abi_version(void);
LURE_API const char * lure_build_info(void);   /* backends compiled in, version */
LURE_API const char * lure_last_error(void);   /* thread-local, valid until the next call on this thread */
LURE_API void         lure_default_params(lure_params * out);
LURE_API void         lure_default_request(lure_request * out);

/*
 * model_dir: HF checkpoint directory (model.safetensors + speech_tokenizer/) or a converted
 *            GGUF from lure_gguf (file or directory containing model.gguf + sidecars).
 * backend:   "auto" | "metal" | "cuda" | "vulkan" | "cpu" (NULL = auto).
 * device:    device index within the backend.
 * threads:   CPU threads (<= 0: hardware concurrency).
 * options:   optional "key=value;key=value" session options, e.g.
 *            "qwen3_tts.weight_type=q8_0;turbo=0". NULL = defaults.
 *            quant: pass qwen3_tts.weight_type / talker_weight_type (native|f32|f16|bf16|q8_0),
 *            or point model_dir at a pre-quantized GGUF.
 */
LURE_API lure_tts * lure_load(const char * model_dir, const char * backend, int32_t device,
                                    int32_t threads, const char * options);
LURE_API void         lure_free(lure_tts * tts);
LURE_API int32_t      lure_sample_rate(const lure_tts * tts);

/* The loaded checkpoint. variant: "base" | "custom_voice" | "voice_design". Preset speakers are sorted by
 * lowercase name; lure_speaker_name returns NULL and lure_speaker_id -1 outside 0..count-1. */
LURE_API const char * lure_variant(const lure_tts * tts);
LURE_API int32_t      lure_speaker_count(const lure_tts * tts);
LURE_API const char * lure_speaker_name(const lure_tts * tts, int32_t index);
LURE_API int32_t      lure_speaker_id(const lure_tts * tts, int32_t index);
/* JSON: variant, size, family, talker_hidden_size, code_predictor_hidden_size, speaker_encoder_dim, speakers
 * (name -> id), speaker_dialects, languages, default_language, default_layout, modes, instruct, turbo.
 * The offline form reads config.json and the tokenizer only (no weights). Free with lure_string_free. */
LURE_API char *       lure_model_info(const lure_tts * tts);
LURE_API char *       lure_model_info_offline(const char * model_dir);

/* Voice (Base checkpoints) = reference codes + x-vector + reference transcript. An empty or NULL ref_text
 * makes an x-vector-only voice, usable in xvec mode only. Pass 24 kHz mono for exact parity with qwen-tts
 * (other rates are linearly resampled inside the encoders). */
LURE_API lure_voice * lure_voice_create(lure_tts * tts, const float * pcm, size_t n_samples,
                                              int32_t sample_rate, const char * ref_text);
LURE_API lure_voice * lure_voice_create_from_wav(lure_tts * tts, const char * wav_path,
                                                       const char * ref_text);
/* Disk cache: <cache_dir>/<fnv(audio,ref_text)>-<model>.alv, created on miss. cache_dir NULL = no disk cache. */
LURE_API lure_voice * lure_voice_cached(lure_tts * tts, const float * pcm, size_t n_samples,
                                              int32_t sample_rate, const char * ref_text, const char * cache_dir);
LURE_API int32_t        lure_voice_save(const lure_voice * voice, const char * path);
LURE_API lure_voice * lure_voice_load(lure_tts * tts, const char * path);
LURE_API int32_t        lure_voice_frames(const lure_voice * voice);
/* 1 = the voice has a reference transcript (icl and xvec), 0 = x-vector only (xvec). */
LURE_API int32_t        lure_voice_has_transcript(const lure_voice * voice);
/* Portable voice exchange with the core's voice cache (core Voice = ref_codes int32 [T,16] frame-major
 * + spk_emb float32 [dims], dims = talker hidden size: 1024 on 0.6B, 2048 on 1.7B). from_codes skips the
 * encoders entirely; codes may be NULL with frames 0 for an x-vector-only voice. The getters copy up to
 * capacity values and return the total count (frames*groups / dims); pass out=NULL to query the size. */
LURE_API lure_voice * lure_voice_from_codes(lure_tts * tts, const int32_t * codes, int32_t frames,
                                                  int32_t groups, const float * speaker, int32_t dims,
                                                  const char * ref_text);
LURE_API int64_t        lure_voice_codes(const lure_voice * voice, int32_t * out, size_t capacity);
LURE_API int64_t        lure_voice_speaker(const lure_voice * voice, float * out, size_t capacity);
LURE_API void           lure_voice_free(lure_voice * voice);

/*
 * Synthesize ONE segment (the core does segmentation). params NULL = defaults. Returns 0 on success
 * (also when cancelled), negative on error: a mode the checkpoint does not run, a missing voice or
 * speaker, an unknown language, or an instruct on the 0.6B CustomVoice (qwen-tts drops it). info may be NULL.
 * icl decodes the reference codes in front of the generated codes and cuts them; the other modes decode
 * the generated codes alone.
 */
LURE_API int32_t lure_generate(lure_tts * tts, const lure_request * request, const lure_params * params,
                                   lure_pcm_callback callback, void * user, lure_gen_info * info);

/* Prefill dump (token ids + per-row embedding recipe + trailing rows) as JSON; no generation.
 * Free with lure_string_free. */
LURE_API char * lure_dump_prefill(lure_tts * tts, const lure_request * request);
/* Same without loading weights: reads config.json and the tokenizer files only (a folder with just those
 * works). request->voice is ignored; icl uses ref_text and ref_frames symbolic reference frames (only the
 * count matters for the layout), xvec a zero x-vector. */
LURE_API char * lure_dump_prefill_offline(const char * model_dir, const lure_request * request,
                                              const char * ref_text, int32_t ref_frames);
/* Raw tokenizer (no template, added tokens as single ids): JSON int array. Tokenizer files only. */
LURE_API char * lure_tokenize_offline(const char * model_dir, const char * text);
LURE_API void   lure_string_free(char * value);

#ifdef __cplusplus
}
#endif

#endif /* LURE_TTS_H */
