#pragma once
#include <string>
#include <jni.h>

class JVMManager {
public:
    static void start(const std::string& jvm_args);
    static JNIEnv* attach();           // current-thread attach
    static void detach();              // OPTIONAL for thread-pool reuse
};