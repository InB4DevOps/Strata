// Opt-in prefill stream intervals. No synchronization is introduced by this profiler.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace strata::prefill::profile {
inline std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += c;
    }
    return out + '"';
}

struct Context {
    int64_t chunk = -1, tokens = 0, layer = -1;
    std::string implementation = "default";
    int gu_type = -1, down_type = -1;
    auto key() const { return std::tie(chunk, tokens, layer, implementation, gu_type, down_type); }
    bool operator<(const Context& other) const { return key() < other.key(); }
};

class Output {
    std::FILE* file_ = nullptr;
    std::string run_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    inline static std::mutex mutex_;
    inline static std::atomic<unsigned long long> sequence_{0};
public:
    bool complete = false, valid = true;
    explicit Output(int device, int64_t pos, int64_t tokens, int64_t lb, int64_t le) {
        const char* path = std::getenv("STRATA_PREFILL_PROFILE");
        if (!path || !*path) return;
        file_ = std::fopen(path, "ab");
        if (!file_) { std::fprintf(stderr, "strata prefill profile: cannot open %s\n", path); return; }
        run_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
               std::to_string(sequence_++);
        cudaDeviceProp prop{};
        const bool have_prop = cudaGetDeviceProperties(&prop, device) == cudaSuccess;
        write("\"type\":\"run\",\"device\":" + std::to_string(device) +
              ",\"gpu\":" + quote(have_prop ? prop.name : "unknown") +
              ",\"position\":" + std::to_string(pos) + ",\"tokens\":" + std::to_string(tokens) +
              ",\"layer_begin\":" + std::to_string(lb) + ",\"layer_end\":" + std::to_string(le));
    }
    bool enabled() const { return file_ != nullptr; }
    void write(const std::string& fields) {
        if (!file_) return;
        const std::string line = "{\"schema\":1,\"run\":" + quote(run_) + "," + fields + "}\n";
        // Each stage has its own FILE; flushing under the lock keeps records intact within this process.
        std::lock_guard<std::mutex> lock(mutex_);
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() || std::fflush(file_) != 0) {
            std::fprintf(stderr, "strata prefill profile: output write failed\n");
            valid = false;
            std::fclose(file_);
            file_ = nullptr;
        }
    }
    void host(const char* name, double ms, int64_t chunk = -1, int64_t layer = -1) {
        if (!enabled()) return;
        std::ostringstream s;
        s << "\"type\":\"host\",\"phase\":" << quote(name) << ",\"ms\":" << ms
          << ",\"chunk\":" << chunk << ",\"layer\":" << layer;
        write(s.str());
    }
    ~Output() {
        if (!file_) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        write("\"type\":\"end\",\"complete\":" + std::string(complete ? "true" : "false") +
              ",\"valid\":" + (valid ? "true" : "false") + ",\"wall_ms\":" + std::to_string(ms));
        if (file_) std::fclose(file_);
    }
};

template <size_t N> class Timer {
    struct Mark { cudaEvent_t event = nullptr; int phase = 0; Context context; };
    struct Total { double ms = 0, max_ms = 0; size_t count = 0; };
    std::vector<Mark> marks_;
    size_t used_ = 0;
    const char* const* names_;
    Output* output_;
    const char* stream_;
    bool healthy_ = true;
    static constexpr size_t max_events = 8192;
    void fail(const char* message) {
        healthy_ = false;
        if (output_) output_->valid = false;
        std::fprintf(stderr, "strata prefill profile: %s; timing is incomplete\n", message);
    }
public:
    bool on;
    int device = -1;
    double ms[N] = {};
    size_t intervals[N] = {};
    Context context;
    Timer(const char* const* names, Output* output, const char* stream)
        : names_(names), output_(output), stream_(stream),
          on(std::getenv("STRATA_PREFILL_TIMING") != nullptr || (output && output->enabled())) {
        if (on && cudaGetDevice(&device) != cudaSuccess) fail("cannot identify device");
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    bool healthy() const { return healthy_; }
    // A dispatch may only reveal its selected implementation after it has enqueued the work.
    void label_last(const char* implementation) {
        if (output_ && output_->enabled() && used_) marks_[used_ - 1].context.implementation = implementation;
    }
    // Called on the owning device. Partial collection queries events, never waits for them.
    void fold(bool completed = true) {
        if (!on || !healthy_ || used_ < 2) return;
        std::map<std::pair<Context, int>, Total> totals;
        size_t consumed = 0;
        for (; consumed + 1 < used_; ++consumed) {
            const auto& a = marks_[consumed];
            const auto& b = marks_[consumed + 1];
            if (!completed) {
                const auto status = cudaEventQuery(b.event);
                if (status == cudaErrorNotReady) break;
                if (status != cudaSuccess) { fail("event query failed"); break; }
            }
            float elapsed = 0;
            if (cudaEventElapsedTime(&elapsed, a.event, b.event) != cudaSuccess) {
                fail("event elapsed time failed"); break;
            }
            ms[a.phase] += elapsed;
            ++intervals[a.phase];
            if (output_ && output_->enabled()) {
                auto& t = totals[{a.context, a.phase}];
                t.ms += elapsed; t.max_ms = std::max(t.max_ms, double(elapsed)); ++t.count;
            }
        }
        // Keep the last completed boundary: its outgoing interval belongs to its original context.
        std::rotate(marks_.begin(), marks_.begin() + consumed, marks_.begin() + used_);
        used_ -= consumed;
        for (const auto& entry : totals) {
            const auto& c = entry.first.first;
            const auto& t = entry.second;
            std::ostringstream s;
            s << "\"type\":\"gpu\",\"device\":" << device << ",\"stream\":" << quote(stream_)
              << ",\"phase\":" << quote(names_[entry.first.second]) << ",\"chunk\":" << c.chunk
              << ",\"tokens\":" << c.tokens << ",\"layer\":" << c.layer
              << ",\"implementation\":" << quote(c.implementation)
              << ",\"gu_type\":" << c.gu_type << ",\"down_type\":" << c.down_type
              << ",\"ms\":" << t.ms << ",\"intervals\":" << t.count << ",\"max_ms\":" << t.max_ms;
            output_->write(s.str());
        }
    }
    void mark(int phase, cudaStream_t stream) {
        if (!on || !healthy_) return;
        if (used_ && used_ % 256 == 0) fold(false);
        if (!healthy_) return;
        if (used_ == max_events) { fail("event capacity exceeded (GPU has not caught up)"); return; }
        if (used_ == marks_.size()) {
            cudaEvent_t event = nullptr;
            if (cudaEventCreate(&event) != cudaSuccess) { fail("event creation failed"); return; }
            marks_.push_back({event, 0, {}});
        }
        auto& m = marks_[used_];
        m.phase = phase;
        if (output_ && output_->enabled()) m.context = context;
        if (cudaEventRecord(m.event, stream) != cudaSuccess) { fail("event record failed"); return; }
        ++used_;
    }
    void release() {
        for (const auto& m : marks_) cudaEventDestroy(m.event);
        marks_.clear(); used_ = 0;
    }
    ~Timer() { release(); }
};
}  // namespace strata::prefill::profile
