#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "benchmark_types.hpp"
#include "expander.hpp"
#include "retry_policy.hpp"
#include "circuit_breaker.hpp"
#include "async_writer.hpp"
#include "jvm_manager.hpp"
#include "jni_bridge.h"
#include "thread_pool_worker.hpp"
#include "output_manager.hpp"

int main(int argc, char* argv[]) {
    CLI::App app{"nlserver – ultra-fast NetLogo batch"};
    std::string yaml;
    app.add_option("experiment", yaml)->required();
    CLI11_PARSE(app, argc, argv);

    /* -------------- YAML load & auto-tune -------------- */
    ExperimentConfig cfg = load_experiment(yaml);
    BenchmarkConfig  bench = auto_benchmark(cfg);
    if (cfg.threads) bench.threads = cfg.threads;
    cfg.threads = bench.threads;

    spdlog::info("Auto-tuned JVM: {}", bench.jvm_args);
    JVMManager::start(bench.jvm_args);

    // ---- build matrix ----
    std::vector<JobParams> matrix = build_matrix(cfg);

    // ---- output path = YAML name + timestamp ----
    std::string out_name = cfg.name + ".csv";
    OutputManager out(cfg.name + "_results");   // directory suffixed

    // ---- Thread pool
    ThreadPool pool(cfg.threads);

    CircuitBreaker cb(3);
    RetryPolicy<5> retry;

    out.open_result_file(out_name);
    
    // Create header based on whether metrics are specified
    std::string header = "job_id,rep";
    bool use_metrics = !cfg.metrics.empty();
    if (use_metrics) {
        for (const std::string& metric : cfg.metrics) {
            header += "," + metric;
        }
    } else {
        header += ",mean";  // Legacy single metric
    }
    out.write_result(header);

    std::atomic<size_t> done{0};
    for (const JobParams& jp : matrix) {
        for (int rep = 0; rep < cfg.reps; ++rep) {
            pool.submit([&] {
                try {
                    retry.invoke([&] {
                        cb.call([&] {
                            std::string result_line = fmt::format("{},{}", jp.job_id, rep);
                            
                            if (use_metrics) {
                                // Use new metrics collection
                                std::vector<double> metrics_results = run_netlogo_sim_with_metrics(cfg, jp.param_string, rep);
                                for (double metric_value : metrics_results) {
                                    result_line += "," + std::to_string(metric_value);
                                }
                            } else {
                                // Legacy single metric
                                double res = run_netlogo_sim(cfg, jp.param_string, rep);
                                result_line += "," + std::to_string(res);
                            }
                            
                            out.write_result(result_line);
                            ++done;
                        });
                    });
                } catch (...) { /* fall through */ }
            });
        }
    }

    pool.wait();
    out.flush();               // flush last lines
    spdlog::info("Finished writing {} rows → {}", done.load(), out_name);
    
    return 0;
}    