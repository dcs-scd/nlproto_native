// src/jni_bridge.cpp
#include "jni_bridge.h"
#include "jvm_manager.hpp"
#include "benchmark_types.hpp"
#include <string>

double jni_run(int ticks,
               const std::string& modelPath,
               const std::string& commands,
               int seed,
               bool headless)
{
try {
    JNIEnv* env = JVMManager::attach();

    jclass cls  = env->FindClass("nl/proto/NetLogoRunner");
    if (!cls) throw std::runtime_error("NetLogoRunner class not found");

    jmethodID mid = headless
        ? env->GetStaticMethodID(cls, "headless",
                                 "(ILjava/lang/String;Ljava/lang/String;I)D")
        : env->GetStaticMethodID(cls, "gui",
                                 "(ILjava/lang/String;Ljava/lang/String;I)D");
    if (!mid) throw std::runtime_error("headless/gui static method not found");

    jstring jModel  = env->NewStringUTF(modelPath.c_str());
    jstring jParams = env->NewStringUTF(commands.c_str());

    jdouble res = env->CallStaticDoubleMethod(
        cls, mid, ticks, jModel, jParams, seed);

    env->DeleteLocalRef(jModel);
    env->DeleteLocalRef(jParams);
    return static_cast<double>(res);

} catch (const std::exception& ex) {
    throw std::runtime_error(std::string("jni_run failed: ") + ex.what());
}
}

// New function for running with metrics collection
std::vector<double> jni_run_with_metrics(int ticks,
                                        const std::string& modelPath,
                                        const std::string& commands,
                                        int seed,
                                        const std::vector<std::string>& metrics,
                                        bool headless)
{
try {
    JNIEnv* env = JVMManager::attach();

    jclass cls = env->FindClass("nl/proto/NetLogoRunner");
    if (!cls) throw std::runtime_error("NetLogoRunner class not found");

    jmethodID mid = env->GetStaticMethodID(cls, "headlessWithMetrics",
                                          "(ILjava/lang/String;Ljava/lang/String;I[Ljava/lang/String;)[D");
    if (!mid) throw std::runtime_error("headlessWithMetrics static method not found");

    // Create Java string array for metrics
    jobjectArray jMetrics = env->NewObjectArray(metrics.size(), env->FindClass("java/lang/String"), nullptr);
    for (size_t i = 0; i < metrics.size(); i++) {
        jstring jMetric = env->NewStringUTF(metrics[i].c_str());
        env->SetObjectArrayElement(jMetrics, i, jMetric);
        env->DeleteLocalRef(jMetric);
    }

    jstring jModel = env->NewStringUTF(modelPath.c_str());
    jstring jParams = env->NewStringUTF(commands.c_str());

    jdoubleArray jRes = (jdoubleArray)env->CallStaticObjectMethod(
        cls, mid, ticks, jModel, jParams, seed, jMetrics);

    // Convert Java double array to C++ vector
    std::vector<double> results;
    if (jRes) {
        jsize len = env->GetArrayLength(jRes);
        jdouble* elements = env->GetDoubleArrayElements(jRes, nullptr);
        results.assign(elements, elements + len);
        env->ReleaseDoubleArrayElements(jRes, elements, JNI_ABORT);
        env->DeleteLocalRef(jRes);
    }

    env->DeleteLocalRef(jModel);
    env->DeleteLocalRef(jParams);
    env->DeleteLocalRef(jMetrics);
    
    return results;

} catch (const std::exception& ex) {
    throw std::runtime_error(std::string("jni_run_with_metrics failed: ") + ex.what());
}
}

// Wrapper function to match the expected signature
double run_netlogo_sim(const ExperimentConfig& cfg, const std::string& paramString, int seed) {
    return jni_run(cfg.ticks, cfg.model_path, paramString, seed, true);
}

// New wrapper function for metrics collection
std::vector<double> run_netlogo_sim_with_metrics(const ExperimentConfig& cfg, const std::string& paramString, int seed) {
    return jni_run_with_metrics(cfg.ticks, cfg.model_path, paramString, seed, cfg.metrics, true);
}