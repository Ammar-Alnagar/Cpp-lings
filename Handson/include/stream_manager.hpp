#pragma once
#include "gpu_wrappers.hpp"
#include <vector>
#include <functional>

namespace hpc {

class StreamManager {
public:
    StreamManager() = default;

    explicit StreamManager(size_t num_streams) 
        : streams_(num_streams), events_(num_streams) {}

    size_t size() const { return streams_.size(); }

    CUDAStream& get_stream(size_t index) {
        return streams_.at(index);
    }

    CUDAEvent& get_event(size_t index) {
        return events_.at(index);
    }

    void synchronize_all() {
        for (auto& stream : streams_) {
            stream.synchronize();
        }
    }

    template<typename Func>
    void execute_on_stream(size_t stream_idx, Func&& func) {
        func(streams_[stream_idx].get());
    }

    void record_event(size_t event_idx, size_t stream_idx) {
        events_[event_idx].record(streams_[stream_idx].get());
    }

private:
    std::vector<CUDAStream> streams_;
    std::vector<CUDAEvent> events_;
};

} // namespace hpc
