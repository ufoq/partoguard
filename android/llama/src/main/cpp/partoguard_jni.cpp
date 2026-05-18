#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "PG_LLAMA"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static void log_callback(enum ggml_log_level level, const char * text, void *) {
    int prio = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_ERROR) prio = ANDROID_LOG_ERROR;
    else if (level == GGML_LOG_LEVEL_WARN) prio = ANDROID_LOG_WARN;
    else if (level == GGML_LOG_LEVEL_DEBUG) prio = ANDROID_LOG_DEBUG;
    __android_log_print(prio, "LLAMA_CPP", "%s", text);
}

struct pg_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    mtmd_context * mtmd = nullptr;
    common_sampler * sampler = nullptr;
    std::atomic<bool> stop_flag{false};
};

static pg_context * g_ctx = nullptr;
static std::mutex g_mutex;

static jstring to_jstring(JNIEnv * env, const std::string & s) {
    return env->NewStringUTF(s.c_str());
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_partoguard_llama_LlamaEngine_nativeInit(JNIEnv * env, jobject, jstring jLibDir) {
    llama_log_set(log_callback, nullptr);

    const char * lib_dir = env->GetStringUTFChars(jLibDir, nullptr);
    LOGI("Loading backends from: %s", lib_dir);
    ggml_backend_load_all_from_path(lib_dir);
    env->ReleaseStringUTFChars(jLibDir, lib_dir);

    llama_backend_init();
    LOGI("Backend initialized, %zu backends registered", ggml_backend_reg_count());
}

JNIEXPORT jboolean JNICALL
Java_com_partoguard_llama_LlamaEngine_nativeLoadModel(
        JNIEnv * env, jobject,
        jstring jModelPath, jstring jMmprojPath,
        jint nCtx, jint nThreads) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ctx) {
        LOGW("Model already loaded, releasing first");
        if (g_ctx->sampler) common_sampler_free(g_ctx->sampler);
        if (g_ctx->mtmd) mtmd_free(g_ctx->mtmd);
        if (g_ctx->ctx) llama_free(g_ctx->ctx);
        if (g_ctx->model) llama_model_free(g_ctx->model);
        delete g_ctx;
        g_ctx = nullptr;
    }

    const char * model_path = env->GetStringUTFChars(jModelPath, nullptr);
    const char * mmproj_path = env->GetStringUTFChars(jMmprojPath, nullptr);

    LOGI("Loading model: %s", model_path);
    LOGI("Loading mmproj: %s", mmproj_path);

    g_ctx = new pg_context();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    g_ctx->model = llama_model_load_from_file(model_path, model_params);
    if (!g_ctx->model) {
        LOGE("Failed to load model");
        env->ReleaseStringUTFChars(jModelPath, model_path);
        env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);
        delete g_ctx;
        g_ctx = nullptr;
        return JNI_FALSE;
    }

    // Create context
    int n_threads_actual = nThreads > 0 ? nThreads : 4;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = nCtx > 0 ? nCtx : 4096;
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 2048;
    ctx_params.n_threads = n_threads_actual;
    ctx_params.n_threads_batch = n_threads_actual;

    g_ctx->ctx = llama_init_from_model(g_ctx->model, ctx_params);
    if (!g_ctx->ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_ctx->model);
        env->ReleaseStringUTFChars(jModelPath, model_path);
        env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);
        delete g_ctx;
        g_ctx = nullptr;
        return JNI_FALSE;
    }

    // Initialize mtmd (multimodal)
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.n_threads = n_threads_actual;
    mtmd_params.warmup = false;

    g_ctx->mtmd = mtmd_init_from_file(mmproj_path, g_ctx->model, mtmd_params);
    if (!g_ctx->mtmd) {
        LOGE("Failed to init mtmd (mmproj)");
        llama_free(g_ctx->ctx);
        llama_model_free(g_ctx->model);
        env->ReleaseStringUTFChars(jModelPath, model_path);
        env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);
        delete g_ctx;
        g_ctx = nullptr;
        return JNI_FALSE;
    }

    env->ReleaseStringUTFChars(jModelPath, model_path);
    env->ReleaseStringUTFChars(jMmprojPath, mmproj_path);

    LOGI("Model loaded successfully (ctx=%d, threads=%d, vision=%s)",
         ctx_params.n_ctx, n_threads_actual,
         mtmd_support_vision(g_ctx->mtmd) ? "yes" : "no");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_partoguard_llama_LlamaEngine_nativeComplete(
        JNIEnv * env, jobject,
        jstring jPrompt, jbyteArray jImageRgb, jint imgWidth, jint imgHeight,
        jfloat temperature, jint maxTokens) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_ctx || !g_ctx->model || !g_ctx->ctx || !g_ctx->mtmd) {
        LOGE("Context not initialized");
        return to_jstring(env, "");
    }

    g_ctx->stop_flag = false;

    const char * prompt_cstr = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(prompt_cstr);
    env->ReleaseStringUTFChars(jPrompt, prompt_cstr);

    // Prepare bitmap from image bytes
    mtmd_bitmap * bitmap = nullptr;
    if (jImageRgb != nullptr && imgWidth > 0 && imgHeight > 0) {
        jsize len = env->GetArrayLength(jImageRgb);
        jbyte * data = env->GetByteArrayElements(jImageRgb, nullptr);
        bitmap = mtmd_bitmap_init((uint32_t)imgWidth, (uint32_t)imgHeight, (const unsigned char *)data);
        env->ReleaseByteArrayElements(jImageRgb, data, JNI_ABORT);

        if (!bitmap) {
            LOGE("Failed to create bitmap");
            return to_jstring(env, "");
        }
    }

    // Tokenize with media
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps_ptr = bitmap;

    mtmd_input_text input_text;
    input_text.text = prompt.c_str();
    input_text.add_special = true;
    input_text.parse_special = true;

    int32_t ret = mtmd_tokenize(g_ctx->mtmd, chunks, &input_text,
                                bitmap ? &bitmaps_ptr : nullptr,
                                bitmap ? 1 : 0);
    if (ret != 0) {
        LOGE("mtmd_tokenize failed: %d", ret);
        mtmd_input_chunks_free(chunks);
        if (bitmap) mtmd_bitmap_free(bitmap);
        return to_jstring(env, "");
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(g_ctx->ctx), true);

    // Decode all chunks using the helper (handles text + image automatically)
    llama_pos n_past = 0;
    int32_t eval_ret = mtmd_helper_eval_chunks(g_ctx->mtmd, g_ctx->ctx, chunks,
                                               n_past, 0, 2048, true, &n_past);
    mtmd_input_chunks_free(chunks);
    if (bitmap) mtmd_bitmap_free(bitmap);

    if (eval_ret != 0) {
        LOGE("mtmd_helper_eval_chunks failed: %d", eval_ret);
        return to_jstring(env, "");
    }

    // Setup sampler
    if (g_ctx->sampler) {
        common_sampler_free(g_ctx->sampler);
    }
    common_params_sampling sparams;
    sparams.temp = temperature > 0 ? temperature : 0.1f;
    sparams.top_k = 1;
    g_ctx->sampler = common_sampler_init(g_ctx->model, sparams);

    // Generate tokens
    std::string result;
    int max_tok = maxTokens > 0 ? maxTokens : 512;
    const llama_vocab * vocab = llama_model_get_vocab(g_ctx->model);

    for (int i = 0; i < max_tok && !g_ctx->stop_flag; i++) {
        llama_token token = common_sampler_sample(g_ctx->sampler, g_ctx->ctx, -1);
        common_sampler_accept(g_ctx->sampler, token, true);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        std::string piece = common_token_to_piece(g_ctx->ctx, token, true);
        result += piece;

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(g_ctx->ctx, batch) != 0) {
            LOGE("llama_decode failed during generation at token %d", i);
            break;
        }
    }

    // Cleanup after generation to free memory between calls
    if (g_ctx->sampler) {
        common_sampler_free(g_ctx->sampler);
        g_ctx->sampler = nullptr;
    }
    llama_memory_clear(llama_get_memory(g_ctx->ctx), true);

    LOGI("Generated %zu chars", result.size());
    return to_jstring(env, result);
}

JNIEXPORT void JNICALL
Java_com_partoguard_llama_LlamaEngine_nativeStop(JNIEnv *, jobject) {
    if (g_ctx) {
        g_ctx->stop_flag = true;
    }
}

JNIEXPORT void JNICALL
Java_com_partoguard_llama_LlamaEngine_nativeRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) {
        if (g_ctx->sampler) common_sampler_free(g_ctx->sampler);
        if (g_ctx->mtmd) mtmd_free(g_ctx->mtmd);
        if (g_ctx->ctx) llama_free(g_ctx->ctx);
        if (g_ctx->model) llama_model_free(g_ctx->model);
        delete g_ctx;
        g_ctx = nullptr;
    }
    llama_backend_free();
    LOGI("Released all resources");
}

} // extern "C"
