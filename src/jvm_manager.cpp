#include "jvm_manager.hpp"
#include <stdexcept>
#include <sstream>
#include <vector>

static JavaVM* g_jvm = nullptr;

void JVMManager::start(const std::string& args) {
    if (g_jvm) return;
    JavaVMInitArgs vm_args{};
    JavaVMOption  opts[16];
    
    // Parse args manually to handle quoted paths properly
    std::vector<std::string> tokens;
    std::string current;
    bool in_quotes = false;
    
    for (size_t i = 0; i < args.length(); ++i) {
        char c = args[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ' ' && !in_quotes) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) tokens.push_back(current);
    
    int n = 0;
    for (const auto& tok : tokens) {
        if (n < 16) {
            opts[n++].optionString = const_cast<char*>(tok.c_str());
        }
    }

    vm_args.version  = JNI_VERSION_1_8;
    vm_args.nOptions = n;
    vm_args.options  = opts;

    JNIEnv* env;
    jint rc = JNI_CreateJavaVM(&g_jvm, (void**)&env, &vm_args);
    if (rc != JNI_OK) throw std::runtime_error("Cannot create JVM");
}

JNIEnv* JVMManager::attach() {
    JNIEnv* env;
    if (g_jvm->GetEnv((void**)&env, JNI_VERSION_1_8) == JNI_EDETACHED)
        g_jvm->AttachCurrentThread((void**)&env, nullptr);
    return env;
}

void JVMManager::detach() {
    g_jvm->DetachCurrentThread();
}