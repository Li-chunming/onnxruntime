#include "core/platform/threadpool.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(int thread_pool_size, Env* env, const ThreadOptions& thread_options,
                                             const std::string& name, bool allow_spinning,
                                             Eigen::Allocator* allocator);
}  // namespace concurrency
}  // namespace onnxruntime