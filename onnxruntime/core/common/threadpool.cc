
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/threadpool.h"
#include "core/common/common.h"
#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif
#include "core/util/eigen_common_wrapper.h"
#include "core/platform/EigenNonBlockingThreadPool.h"


namespace onnxruntime {
namespace concurrency {

ThreadPool::ThreadPool(Env* env, const std::string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options, const std::string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* /*env*/, const ThreadOptions& /*thread_options*/, const std::string& /*name*/, int num_threads, bool low_latency_hint, Eigen::Allocator* allocator)
    : eigen_thread_env_() {
  ORT_ENFORCE(num_threads>= 1);
  eigen_threadpool_.reset(new ThreadPoolTempl<EigenEnvironment>(num_threads, low_latency_hint, eigen_thread_env_));
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(underlying_threadpool_, num_threads, allocator));
}

ThreadPool::ThreadPool(Eigen::ThreadPoolInterface* user_threadpool) {
  underlying_threadpool_ = user_threadpool;
  threadpool_device_.reset(
      new Eigen::ThreadPoolDevice(underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr));
}

ThreadPool::~ThreadPool() {}
void ThreadPool::ParallelFor(int32_t total, std::function<void(int32_t)> fn) {
  if (total <= 0)
    return;

  if (total == 1) {
    fn(0);
    return;
  }

  Barrier barrier(static_cast<unsigned int>(total - 1));
  std::function<void(int32_t)> handle_iteration = [&barrier, &fn](int iteration) {
    fn(iteration);
    barrier.Notify();
  };

  for (int32_t id = 1; id < total; ++id) {
    Schedule([=, &handle_iteration]() { handle_iteration(id); });
  }

  fn(0);
  barrier.Wait();
}

void ThreadPool::Schedule(std::function<void()> fn) {
  ORT_ENFORCE(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(const int64_t total, const int64_t block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size || NumThreads() == 1) {
    return 1;
  }
  //TODO:check overflow?
  return static_cast<int>((total + block_size - 1) / block_size);
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(const int64_t block_size, const int64_t total) {
  return NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
}

void ThreadPool::ParallelFor(int64_t total, const SchedulingParams& scheduling_params,
                             const std::function<void(int64_t, int64_t)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, static_cast<double>(scheduling_params.cost_per_unit().value()), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(total, scheduling_params.block_size().value(), fn);
      }
      break;
    }
  }
}

void ThreadPool::TransformRangeConcurrently(const int64_t block_size, const int64_t total,
                                            const std::function<void(int64_t, int64_t)>& fn) {
  ParallelFor(total,
              SchedulingParams(SchedulingStrategy::kFixedBlockSize, optional<int64_t>() /* cost_per_unit */, block_size), fn);
}

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(const int64_t total, const int64_t block_size,
                                                     const std::function<void(int64_t, int64_t)>& fn) {
  const int num_shards_used = NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(int64_t, int64_t)> handle_range = [=, &handle_range, &counter, &fn](int64_t first, int64_t last) {
    while (last - first > block_size) {
      // Find something near the midpoint which is a multiple of block size.
      const int64_t mid = first + ((last - first) / 2 + block_size - 1) / block_size * block_size;
      Schedule([=, &handle_range]() { handle_range(mid, last); });
      last = mid;
    }
    // Single block or less, execute directly.
    fn(first, last);
    counter.DecrementCount();  // The shard is done.
  };
  if (num_shards_used <= NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(0, total);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    Schedule([=, &handle_range]() { handle_range(0, total); });
  }
  counter.Wait();
}

void ThreadPool::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
    const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  static_assert(sizeof(onnxruntime::TensorOpCost) == sizeof(Eigen::TensorOpCost),
      "TensorOpCost size mismatch");
  threadpool_device_->parallelFor(total, *reinterpret_cast<const Eigen::TensorOpCost*>(&cost_per_unit), fn);
 }
void ThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                             const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  ORT_ENFORCE(total >= 0);
  threadpool_device_->parallelFor(total, Eigen::TensorOpCost(0, 0, static_cast<double>(cost_per_unit)),
                                  [&fn](std::ptrdiff_t first, std::ptrdiff_t last) { fn(first, last); });
}

void ThreadPool::ParallelForWithWorkerId(int64_t total, int64_t cost_per_unit,
                                         const std::function<void(int64_t, int64_t, int)>& fn) {
  ORT_ENFORCE(total >= 0);
  ORT_ENFORCE(total == (int64_t)(std::ptrdiff_t)total);

  threadpool_device_->parallelFor(total, Eigen::TensorOpCost(0, 0, static_cast<double>(cost_per_unit)),
                                  [this, &fn](int64_t start, int64_t limit) {
                                    // ParallelFor may use the current thread to
                                    // do some work synchronously. When calling
                                    // CurrentThreadId() from outside of the
                                    // thread pool, we get -1, so we can shift
                                    // every id up by 1.
                                    int id = CurrentThreadId() + 1;
                                    fn(start, limit, id);
                                  });
}

void ThreadPool::ParallelForWithWorkerId(int64_t total, const SchedulingParams& scheduling_params,
                                         const std::function<void(int64_t, int64_t, int)>& fn) {
  ParallelFor(total, scheduling_params, [this, &fn](int64_t start, int64_t limit) {
    // We may use the current thread to do some work synchronously.
    // When calling CurrentThreadId() from outside of the thread
    // pool, we get -1, so we can shift every id up by 1.
    int id = CurrentThreadId() + 1;
    fn(start, limit, id);
  });
}

int ThreadPool::NumThreads() const { return underlying_threadpool_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return underlying_threadpool_->CurrentThreadId(); }

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start, int limit) {
  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
  // ThreadPool::SetStealPartitions is only called in the constructor of
  // RunHandlerPool::Impl, which currently instantiates ThreadPool using a
  // constructor that does not take user_threadpool. Thus we assume
  // eigen_threadpool_ is not null here.
  ORT_ENFORCE(eigen_threadpool_ != nullptr);
  eigen_threadpool_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  ORT_ENFORCE(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace thread
}  // namespace tensorflow