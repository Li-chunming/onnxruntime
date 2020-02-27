// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include <sstream>

namespace onnxruntime {
class BatchNormHelper {
private:
    static bool AreShapesEqual(const TensorShape& left, const TensorShape& right, size_t begin1, size_t begin2,
            std::string* error_str){
        if(left.NumDimensions() < begin1 || right.NumDimensions() < begin2) {
            if(error_str != nullptr){
                *error_str = "expect [???]";
            }
            return false;
        }
        if(left.NumDimensions() != right.NumDimensions() - begin2 + begin1) {
            std::ostringstream s;
            s << "expect a "<<right.NumDimensions() - begin2 + begin1<< " dimensions tensor with shape like [???";
            for(size_t i=begin2;i!=right.NumDimensions();++i){
                s << "," << right[i];
            }
            s << "]";
            if(error_str != nullptr){
                *error_str = s.str();
            }
            return false;
        }
        auto left_p = reinterpret_cast<const std::vector<int64_t>*>(&left);
        auto right_p = reinterpret_cast<const std::vector<int64_t>*>(&right);
        size_t ele_count = left.NumDimensions() - begin1;
        if(memcmp(left_p->data()+begin1, right_p->data()+begin2, ele_count * sizeof(int64_t)) == 0) return true;
        if(error_str != nullptr){
            std::ostringstream s;
            s << "expect [";
            if(begin1>0){
                s << left[0];
                for(size_t i=1;i!=begin1;++i){
                    s << "," << left[i];
                }
            }
            for(size_t i=begin2;i!=right.NumDimensions();++i){
                s << "," << right[i];
            }
            s << "]";
            *error_str = s.str();
        }
        return false;
    }
 public:
  static common::Status ValidateInputs(const Tensor* X,
                                       const Tensor* scale,
                                       const Tensor* B,
                                       const Tensor* mean,
                                       const Tensor* var,
                                       bool is_spatial = true) {
    if (X->Shape().NumDimensions() == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid input X: The rank of input X must be at least 1. ");
    }
    // The op also accepts single dimension input of size N in which case C is assumed to be 1
    TensorShape x_shape = X->Shape().NumDimensions() == 1 ? TensorShape({X->Shape()[0], 1}) : X->Shape();
    const auto& x_dims = x_shape.GetDims();

    int64_t num_channels = x_dims[1];
    int num_feature_dims = static_cast<int>(X->Shape().NumDimensions() - 2);  // the first 2 are respectively - N and C

    // defined as per spec and used for validation
    int kNumInputScaleDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputBiasDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputMeanDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    int kNumInputVarianceDimensions = (is_spatial ? 1 : num_feature_dims + 1);
    //constexpr int kMinCudaNumDims = 4;
    //constexpr int kMaxCudaNumDims = 5;

    // validate 'scales' shape
    const auto& scale_dims = scale->Shape().GetDims();
    std::string err_str;
    if (static_cast<int>(scale_dims.size()) != kNumInputScaleDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
              "Invalid input scale: ", scale->Shape(), ", expect a tensor with ", kNumInputScaleDimensions,
              " dimensions.");
    }
    if (scale_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'scale' must be validated
    if (!is_spatial && !AreShapesEqual(scale->Shape(), x_shape, 1, 2, &err_str)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: ", scale->Shape(), " ", err_str);
    }

    // validate 'B' shape
    const auto& B_dims = B->Shape().GetDims();
    if (static_cast<int>(B_dims.size()) != kNumInputBiasDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: NumDimensions() != ", kNumInputBiasDimensions);
    }
    if (B_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'B' must be validated
    if (!is_spatial && !AreShapesEqual(B->Shape(), x_shape, 1, 2, &err_str)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: ", B->Shape(), " ", err_str);
    }

    // validate 'mean' shape
    const auto& mean_dims = mean->Shape().GetDims();
    if (static_cast<int>(mean_dims.size()) != kNumInputMeanDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: NumDimensions() != ", kNumInputMeanDimensions);
    }
    if (mean_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'mean' must be validated
    if (!is_spatial && !AreShapesEqual(mean->Shape(), x_shape, 1, 2, &err_str)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: ", mean->Shape(), " ", err_str);
    }

    // validate 'var' shape
    const auto& var_dims = var->Shape().GetDims();
    if (static_cast<int>(var_dims.size()) != kNumInputVarianceDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: NumDimensions() != ", kNumInputVarianceDimensions);
    }
    if (var_dims[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: 0th dimension != ", num_channels);
    }
    // in non-spatial cases - the other dims of 'var' must be validated
    if (!is_spatial && !AreShapesEqual(var->Shape(), x_shape, 1, 2, &err_str)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: ", var->Shape(), " ", err_str);
    }

    return common::Status::OK();
  }

  static void NormalizeDims(const TensorShape& x_shape, std::vector<int64_t>& new_dims) {
    new_dims.clear();
    auto& orig_dims = x_shape.GetDims();
    if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
        orig_dims.size() == 5 /*supported size by CUDA*/) {
      new_dims = orig_dims;
      return;
    }

    auto rank = x_shape.NumDimensions();
    auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
    auto num_channels = rank > 1 ? orig_dims[1] : 1;
    auto width = rank > 3 ? orig_dims[3] : 1;
    auto height = rank > 2 ? orig_dims[2] : 1;
    new_dims = {num_samples, num_channels, height, width};
  }
};
}  // namespace onnxruntime
