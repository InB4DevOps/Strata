// Real runtime events: reuse, cross-fold attribution, nonblocking collection and disabled mode.
#include "../../src/prefill/profile.hpp"
#include <cstdlib>

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const char* names[] = {"even", "odd"};
    cudaStream_t stream = nullptr;
    require(cudaStreamCreate(&stream) == cudaSuccess, "stream creation");
    {
        strata::prefill::profile::Output out(0, 0, 512, 0, 2);
        strata::prefill::profile::Timer<2> timer(names, &out, "compute");
        timer.on = false;
        timer.mark(0, stream);
        timer.fold();
        require(timer.intervals[0] == 0 && timer.ms[0] == 0, "disabled timer collected work");
        timer.on = true;
        size_t expected[2] = {};
        for (int i = 0; i <= 12000; ++i) {
            timer.context = {i / 1000, 512, i % 2, i % 2 ? "odd path" : "even path"};
            timer.mark(i % 2, stream);
            if (i < 12000) ++expected[i % 2];
            if (i % 127 == 0) {
                require(cudaStreamSynchronize(stream) == cudaSuccess, "stream sync");
                timer.fold(false);
                timer.fold(false);   // must not charge an interval twice
            }
        }
        require(cudaStreamSynchronize(stream) == cudaSuccess, "final sync");
        timer.fold();
        require(timer.healthy(), "timer failed");
        for (int p = 0; p < 2; ++p)
            require(timer.intervals[p] == expected[p] && timer.ms[p] >= 0, "lost/duplicated interval");
        out.complete = true;
    }
    require(cudaStreamDestroy(stream) == cudaSuccess, "stream destroy");
    std::puts("prefill profile: 12000 intervals across event reuse and repeated folds passed");
}
