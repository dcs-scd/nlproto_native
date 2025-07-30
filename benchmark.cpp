/*
 * benchmark.cpp
 * MIT 2024 – build fine-tuned OpenJDK 21 + NetLogo JVM tuning
 *
 * Computes sensible defaults (threads, heap, GC, JIT) for the particular
 * CPU topology and requested ticks/repetitions.
 */

#include "benchmark.h"
#include <hwloc.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>

// ------------------------------------------------------------------
static int logical_cores() {
    return static_cast<int>(std::thread::hardware_concurrency());
}

static int physical_cores() {
    hwloc_topology_t topo = nullptr;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int depth = hwloc_get_type_depth(topo, HWLOC_OBJ_CORE);
    int count = hwloc_get_nbobjs_by_depth(topo, depth);
    hwloc_topology_destroy(topo);
    return std::max(1, count);
}

static int64_t memoryKiB() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.starts_with("MemTotal:")) {
            std::stringstream ss(line.substr(8));
            int64_t k; ss >> k;
            return k;
        }
    }
    return 8 * 1024 * 1024; // ~8 GB fallback
}

// ------------------------------------------------------------------
std::string heap_default(const Experiment& exp) {
    int64_t memKiB = memoryKiB();
    double factor  = 0.25 * exp.ticks * 0.02; // coarse rule
    int64_t heapKiB = static_cast<int64_t>(memKiB * factor);
    heapKiB = std::clamp(heapKiB, 256 * 1024, static_cast<int64_t>(memKiB * 0.7));
    std::ostringstream oss;
    oss << "-Xmx" << heapKiB / 1024 << "m";
    return oss.str();
}

std::string tune_gc() {
    std::ostringstream gc;
    // EpsilonGC: blazing fast for micro-benchmarks without allocation spikes
    gc << "-XX:+UnlockExperimentalVMOptions "
       << "-XX:+UseEpsilonGC "
       << "-XX:+AlwaysPreTouch "
       << "-XX:+UseTransparentHugePages ";
    return gc.str();
}

std::string tune_threads() {
    int density = logical_cores();
    std::ostringstream th;
    th << "-XX:CompileThreshold=500 "
       << "-XX:TieredStopAtLevel=1 "
       << "-XX:ActiveProcessorCount=" << density;
    return th.str();
}

// ------------------------------------------------------------------
Benchmark benchmark(const std::string& modelPath) {
    Benchmark b;

    Experiment exp = {}; // we do NOT parse YAML here, caller passes necessary fields
    exp.ticks = 1000;    // default until true experiment parsed

    // Threads capped by physical cores up to 16, give worker granularity
    b.threads = std::min(physical_cores(), 16);

    // JVM arguments assembled in string
    std::ostringstream full;
    full << heap_default(exp) << " "
         << tune_gc() << " "
         << tune_threads() << " "
         << "-Dnetlogo.headless.codecache=64m "
         << "--add-opens=java.base/java.lang=ALL-UNNAMED "
         << "--add-opens=java.base/java.nio=ALL-UNNAMED "
         << "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED";
    b.jvm_args = full.str();
    return b;
}