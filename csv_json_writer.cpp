```cpp
/*
 * csv_json_writer.cpp
 * MIT 2024 – produces **compressed** CSV and JSON output from the result vector
 * with minimal dependencies: only C++20 std and libzstd (`#include <zstd.h>`).
 */

#include "csv_json_writer.h"
#include <zstd.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

// ------------------------------------------------------------------
static void write_zstd(FILE* fp, std::string_view data) {
    size_t bufSize = ZSTD_compressBound(data.size());
    std::vector<char> z(bufSize);
    size_t zSize = ZSTD_compress(z.data(), z.size(),
                                 data.data(), data.size(), 3 /* level */);
    fwrite(z.data(), 1, zSize, fp);
}

// ------------------------------------------------------------------
void write_results_csv_zstd(const Experiment& exp,
                            const std::string& path,
                            const std::vector<Result>& results)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("csv open failed: " + path);

    // single header line
    std::ostringstream head, buf;
    head << "run,pa,pb";
    for (const auto& m: exp.metrics) head << ',' << m.name;
    head << '\n';

    // body
    for (const auto& r: results) {
        // reconstruct pa,pb from job_id (skip if >2 vars)
        buf << r.job_id << ',' << std::setprecision(6)
            << results[r.job_id].pa << ',' << results[r.job_id].pb;
        for (double v: r.metrics) buf << ',' << v;
        buf << '\n';
    }
    std::string s = head.str() + buf.str();
    write_zstd(fp, s);
    fclose(fp);
}

// ------------------------------------------------------------------
void write_results_json_zstd(const Experiment& exp,
                             const std::string& path,
                             const std::vector<Result>& results)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("json open failed: " + path);

    std::ostringstream json;
    json << "{\n"
         << "  \"model\": \"" << exp.model << "\",\n"
         << "  \"ticks\": " << exp.ticks << ",\n"
         << "  \"data\": [\n";

    size_t idx = 0;
    for (const auto& r: results) {
        json << "    {\n";
        json << "      \"run\": " << r.job_id << ",\n";
        json << "      \"pa\": " << results[r.job_id].pa << ",\n";
        json << "      \"pb\": " << results[r.job_id].pb << ",\n";
        json << "      \"metrics\": {";
        bool first = true;
        for (size_t m=0; m<exp.metrics.size(); ++m) {
            if (!first) json << ", ";
            json << '"' << exp.metrics[m].name << "\":" << r.metrics[m];
            first = false;
        }
        json << "}";
        json << (++idx == results.size() ? "\n    }" : "\n    },");
        json << "\n";
    }
    json << "  ]\n}\n";

    std::string s = json.str();
    write_zstd(fp, s);
    fclose(fp);
}
```