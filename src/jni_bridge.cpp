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

// Wrapper function to match the expected signature
double run_netlogo_sim(const ExperimentConfig& cfg, const std::string& paramString, int seed) {
    return jni_run(cfg.ticks, cfg.model_path, paramString, seed, true);
}