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

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "ort_mutex.h"
#include <atomic>
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/common/optional.h"


#include <functional>
#include <memory>

//This file use PIMPL to avoid having eigen headers here

namespace Eigen {
class Allocator;
class ThreadPoolInterface;
struct ThreadPoolDevice;



struct StlThreadEnvironment;
}  // namespace Eigen

namespace onnxruntime {

struct TensorOpCost {
  double bytes_loaded_;
  double bytes_stored_;
  double compute_cycles_;
};

template <typename Environment>
class ThreadPoolTempl;
namespace concurrency {


struct EigenEnvironment {
  struct Task {
    std::function<void()> f;
  };

  // EnvThread constructor must start the thread,
  // destructor must join the thread.
  class EnvThread {
   public:
    EnvThread(std::function<void()> f) : thr_(std::move(f)) {}
    ~EnvThread() { thr_.join(); }
    // This function is called when the threadpool is cancelled.
    void OnCancel() {}

   private:
    std::thread thr_;
  };

  EnvThread* CreateThread(std::function<void()> f) { return new EnvThread(std::move(f)); }
  Task CreateTask(std::function<void()> f) { return Task{std::move(f)}; }
  void ExecuteTask(const Task& t) { t.f(); }
};

struct ThreadOptions {};

class BlockingCounter {
 public:
  BlockingCounter(int initial_count) : state_(initial_count << 1), notified_(false) {
    ORT_ENFORCE(initial_count >= 0);
#ifndef NDEBUG
    ORT_ENFORCE(((initial_count << 1) >> 1) == initial_count);
#endif
  }

  ~BlockingCounter() {}

  inline void DecrementCount() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
#ifndef NDEBUG
      ORT_ENFORCE(((v + 2) & ~1) != 0);
#endif
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::lock_guard<OrtMutex> l(mu_);
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      cond_var_.wait(l);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds ms) {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return true;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(l, ms);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  OrtMutex mu_;
  OrtCondVar cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  bool notified_;
};


class ThreadPool {
 public:
  using ThreadEnvironment = EigenEnvironment;
  // Scheduling strategies for ParallelFor. The strategy governs how the given
  // units of work are distributed among the available threads in the
  // threadpool.
  enum class SchedulingStrategy {
    // The Adaptive scheduling strategy adaptively chooses the shard sizes based
    // on the cost of each unit of work, and the cost model of the underlying
    // threadpool device.
    //
    // The 'cost_per_unit' is an estimate of the number of CPU cycles (or
    // nanoseconds if not CPU-bound) to complete a unit of work. Overestimating
    // creates too many shards and CPU time will be dominated by per-shard
    // overhead, such as Context creation. Underestimating may not fully make
    // use of the specified parallelism, and may also cause inefficiencies due
    // to load balancing issues and stragglers.
    kAdaptive,
    // The Fixed Block Size scheduling strategy shards the given units of work
    // into shards of fixed size. In case the total number of units is not
    // evenly divisible by 'block_size', at most one of the shards may be of
    // smaller size. The exact number of shards may be found by a call to
    // NumShardsUsedByFixedBlockSizeScheduling.
    //
    // Each shard may be executed on a different thread in parallel, depending
    // on the number of threads available in the pool. Note that when there
    // aren't enough threads in the pool to achieve full parallelism, function
    // calls will be automatically queued.
    kFixedBlockSize
  };

  // Contains additional parameters for either the Adaptive or the Fixed Block
  // Size scheduling strategy.
  class SchedulingParams {
   public:
    explicit SchedulingParams(SchedulingStrategy strategy, optional<int64_t> cost_per_unit,
                              optional<int64_t> block_size)
        : strategy_(strategy), cost_per_unit_(cost_per_unit), block_size_(block_size) {}

    SchedulingStrategy strategy() const { return strategy_; }
    optional<int64_t> cost_per_unit() const { return cost_per_unit_; }
    optional<int64_t> block_size() const { return block_size_; }

   private:
    // The underlying Scheduling Strategy for which this instance contains
    // additional parameters.
    SchedulingStrategy strategy_;

    // The estimated cost per unit of work in number of CPU cycles (or
    // nanoseconds if not CPU-bound). Only applicable for Adaptive scheduling
    // strategy.
    optional<int64_t> cost_per_unit_;

    // The block size of each shard. Only applicable for Fixed Block Size
    // scheduling strategy.
    optional<int64_t> block_size_;
  };
  // Constructs a pool that contains "num_threads" threads with specified
  // "name". env->StartThread() is used to create individual threads with the
  // given ThreadOptions. If "low_latency_hint" is true the thread pool
  // implementation may use it as a hint that lower latency is preferred at the
  // cost of higher CPU usage, e.g. by letting one or more idle threads spin
  // wait. Conversely, if the threadpool is used to schedule high-latency
  // operations like I/O the hint should be set to false.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const std::string& name, int num_threads, bool low_latency_hint,
             Eigen::Allocator* allocator = nullptr);
  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const std::string& name, int num_threads);
  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads with the given ThreadOptions.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const std::string& name, int num_threads);
  // Constructs a pool that wraps around the thread::ThreadPoolInterface
  // instance provided by the caller. Caller retains ownership of
  // `user_threadpool` and must ensure its lifetime is longer than the
  // ThreadPool instance.
  explicit ThreadPool(Eigen::ThreadPoolInterface* user_threadpool);

  // Waits until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn);

  void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions);

  void ScheduleWithHint(std::function<void()> fn, int start, int limit);

  // Returns the number of shards used by ParallelForFixedBlockSizeScheduling
  // with these parameters.
  int NumShardsUsedByFixedBlockSizeScheduling(const int64_t total, const int64_t block_size);

  // Returns the number of threads spawned by calling TransformRangeConcurrently
  // with these parameters.
  // Deprecated. Use NumShardsUsedByFixedBlockSizeScheduling.
  int NumShardsUsedByTransformRangeConcurrently(const int64_t block_size, const int64_t total);

  // ParallelFor shards the "total" units of work assuming each unit of work
  // having roughly "cost_per_unit" cost, in cycles. Each unit of work is
  // indexed 0, 1, ..., total - 1. Each shard contains 1 or more units of work
  // and the total cost of each shard is roughly the same.
  //
  // "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
  // if not CPU-bound) to complete a unit of work. Overestimating creates too
  // many shards and CPU time will be dominated by per-shard overhead, such as
  // Context creation. Underestimating may not fully make use of the specified
  // parallelism, and may also cause inefficiencies due to load balancing
  // issues and stragglers.
  void ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn);
  void ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn);
  // Similar to ParallelFor above, but takes the specified scheduling strategy
  // into account.
  void ParallelFor(int64_t total, const SchedulingParams& scheduling_params, const std::function<void(int64_t, int64_t)>& fn);

  // Same as ParallelFor with Fixed Block Size scheduling strategy.
  // Deprecated. Prefer ParallelFor with a SchedulingStrategy argument.
  void TransformRangeConcurrently(const int64_t block_size, const int64_t total,
                                  const std::function<void(int64_t, int64_t)>& fn);

  // Shards the "total" units of work. For more details, see "ParallelFor".
  //
  // The function is passed a thread_id between 0 and NumThreads() *inclusive*.
  // This is because some work can happen on the caller thread while the threads
  // in the pool are also being used.
  //
  // The caller can allocate NumThreads() + 1 separate buffers for each thread.
  // Each thread can safely write to the buffer given by its id without
  // synchronization. However, the worker fn may be called multiple times
  // sequentially with the same id.
  //
  // At most NumThreads() unique ids will actually be used, and only a few may
  // be used for small workloads. If each buffer is expensive, the buffers
  // should be stored in an array initially filled with null, and a buffer
  // should be allocated by fn the first time that the id is used.
  void ParallelForWithWorkerId(int64_t total, int64_t cost_per_unit, const std::function<void(int64_t, int64_t, int)>& fn);

  // Similar to ParallelForWithWorkerId above, but takes the specified
  // scheduling strategy into account.
  void ParallelForWithWorkerId(int64_t total, const SchedulingParams& scheduling_params,
                               const std::function<void(int64_t, int64_t, int)>& fn);

  // Returns the number of threads in the pool.
  int NumThreads() const;

  // Returns current thread id between 0 and NumThreads() - 1, if called from a
  // thread in the pool. Returns -1 otherwise.
  int CurrentThreadId() const;

  // If ThreadPool implementation is compatible with Eigen::ThreadPoolInterface,
  // returns a non-null pointer. The caller does not own the object the returned
  // pointer points to, and should not attempt to delete.
  Eigen::ThreadPoolInterface* AsEigenThreadPool() const;

  //Simple ParallelFor
  //Directly schedule the 'total' tasks to the underlying threadpool, without
  //cutting them by halves
  void ParallelFor(int32_t total, std::function<void(int32_t)> fn);
  /**
  Tries to call the given function in parallel.
  **/
  template <typename F>
  inline static void TryParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn) {
#ifdef USE_OPENMP
#pragma omp parallel for
    for (int32_t i = 0; i < total; ++i) {
      fn(i);
    }
#else
    if (tp != nullptr) {
      tp->ParallelFor(total, std::forward<F>(fn));
    } else {
      for (int32_t i = 0; i < total; ++i) {
        fn(i);
      }
    }
#endif    
  }

   /**
 Tries to call the given function in parallel, with calls split into (num_batches) batches.
 **/
  template <typename F>
  inline static void TryBatchParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn, int32_t num_batches = 0) {
    if (tp != nullptr) {
      if (num_batches <= 0) {
        num_batches = tp->NumThreads();
      }
      int32_t block_size = (total + num_batches - 1) / num_batches;      
      tp->ParallelForFixedBlockSizeScheduling(total, block_size, [&](ptrdiff_t s, ptrdiff_t e) {
        for (s; s != e; ++s) fn(s);
          });      
    } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
      for (int32_t i = 0; i < total; ++i) {
        fn(i);
      }
    }
  }
  Eigen::ThreadPoolDevice& device() { return *threadpool_device_.get(); }

 private:
  // Divides the work represented by the range [0, total) into k shards.
  // Calls fn(i*block_size, (i+1)*block_size) from the ith shard (0 <= i < k).
  // Each shard may be executed on a different thread in parallel, depending on
  // the number of threads available in the pool.
  // When (i+1)*block_size > total, fn(i*block_size, total) is called instead.
  // Here, k = NumShardsUsedByFixedBlockSizeScheduling(total, block_size).
  // Requires 0 < block_size <= total.
  void ParallelForFixedBlockSizeScheduling(const int64_t total, const int64_t block_size,
                                           const std::function<void(int64_t, int64_t)>& fn);
  EigenEnvironment eigen_thread_env_;
  // underlying_threadpool_ is the user_threadpool if user_threadpool is
  // provided in the constructor. Otherwise it is the eigen_threadpool_.
  Eigen::ThreadPoolInterface* underlying_threadpool_;
  // eigen_threadpool_ is instantiated and owned by thread::ThreadPool if
  // user_threadpool is not in the constructor.
  std::unique_ptr<ThreadPoolTempl<EigenEnvironment>> eigen_threadpool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ThreadPool);
};

}  // namespace thread
}  // namespace tensorflow

