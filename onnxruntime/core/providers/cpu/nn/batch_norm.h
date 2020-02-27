/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/util/eigen_common_wrapper.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

template <typename T>
class BatchNorm : public OpKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info),
                                                           is_spatial_(op_kernel_info.GetAttrOrDefault<int64_t>("spatial", 1) == 1) {
    auto st = op_kernel_info.GetAttr<float>("epsilon", &epsilon_);
    ORT_ENFORCE(st.IsOK(), st.ErrorMessage());

    // For opset 6-8, if spatial attribute exists, pick up the value (by default spatial == 1)
    // From opset 9 onwards, by default, only the spatial case (spatial == 1) is defined per spec

    //TODO: momentum
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    // Accroding to the ONNX standard, in convolution networks, X is required to be in NCHW format.
    // However, NHWC performs better in most computation frameworks.
    const auto* X = p_op_kernel_context->Input<Tensor>(0);
    const auto* scale = p_op_kernel_context->Input<Tensor>(1);
    const auto* B = p_op_kernel_context->Input<Tensor>(2);
    const auto* mean = p_op_kernel_context->Input<Tensor>(3);
    const auto* var = p_op_kernel_context->Input<Tensor>(4);

    ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, is_spatial_));

    const TensorShape& x_shape = X->Shape();
    // By definition: Y = (X - mean) / sqrt(var + epsilon) * scale + B
    // Indeed most of the frameworks do it as:
    // inv = 1/sqrt(var + epsilon) * scale
    // Y = x * inv + (B - mean * inv)
    Tensor* Y = p_op_kernel_context->Output(0, x_shape);

    const auto& dims_vec = x_shape.GetDims();
    const size_t N = dims_vec[0];
    const size_t C = dims_vec[1];  // assume NCHW as per the spec

    // calculate sample_size (per individual channel)
    size_t sample_size = 1;
    for (size_t i = 2; i < dims_vec.size(); ++i) {
      sample_size *= dims_vec[i];
    }

    // calculate sample_size (including all channels)
    size_t sample_size_incl_all_channels = sample_size * C;

    if (is_spatial_) {
      typename Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
          scale_arr(scale->Data<T>(), C);
      typename Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> mean_arr(
          mean->Data<T>(), C);
      typename Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> bias_arr(
          B->Data<T>(), C);
      typename Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> var_arr(
          var->Data<T>(), C);

      typename Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
          input_tensor(X->Data<T>(), N * C, sample_size);
      typename Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> output_tensor(
          Y->MutableData<T>(), N * C, sample_size);

      Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> eps(C);
      eps.setConstant(epsilon_);
      Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> inv_std(C);
      concurrency::ThreadPool* tp = p_op_kernel_context->GetOperatorThreadPool();
      Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
      batch_by_one.set(0, C);
      std::array<int, 2> bcast{static_cast<int>(N), static_cast<int>(sample_size)};
      if (tp != nullptr) {
        inv_std.device(tp->device()) = (scale_arr / (var_arr + eps).sqrt());
        output_tensor.device(tp->device()) = input_tensor * inv_std.reshape(batch_by_one).broadcast(bcast) +
                                    (bias_arr - mean_arr * inv_std).eval().reshape(batch_by_one).broadcast(bcast);
      } else {
        inv_std = (scale_arr / (var_arr + eps).sqrt());
        output_tensor = input_tensor * inv_std.reshape(batch_by_one).broadcast(bcast) +
                        (bias_arr - mean_arr * inv_std).eval().reshape(batch_by_one).broadcast(bcast);
      }
    } else {
      // Deprecated
      ConstEigenVectorArrayMap<T> scale_arr(scale->Data<T>(), sample_size_incl_all_channels);
      ConstEigenVectorArrayMap<T> bias_arr(B->Data<T>(), sample_size_incl_all_channels);

      // Regardless of training or testing, we will apply the estimated mean
      // and standard deviation to the input. For testing, they are
      // specified directly by the input, and for training, they are computed
      // by the op.
      Eigen::Array<T, Eigen::Dynamic, 1> inv_std(sample_size_incl_all_channels);
      ConstEigenVectorArrayMap<T> var_arr(var->Data<T>(), sample_size_incl_all_channels);
      inv_std = (var_arr + epsilon_).sqrt().inverse();
      ConstEigenVectorArrayMap<T> mean_arr(mean->Data<T>(), sample_size_incl_all_channels);
      Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
      Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
      EigenArrayMap<T> Y_arr(Y->template MutableData<T>(), sample_size_incl_all_channels, N);
      ConstEigenArrayMap<T> X_arr(X->Data<T>(), sample_size_incl_all_channels, N);
      for (size_t n = 0; n < N; ++n) {
        Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
      }
    }

    return Status::OK();
  }

 protected:
  float epsilon_;
  const bool is_spatial_;
  //int64_t is_test_;   ignored in this implementation since we're doing inferencing only.
};
}  // namespace onnxruntime
