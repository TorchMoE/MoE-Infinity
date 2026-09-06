/*
 * INT8 symmetric paged-attention decode launcher (reusable declaration).
 * Consumed by the Python extension wrapper (paged_attention.cu) and the
 * standalone CUDA test. This header declares only; it must never define
 * PYBIND11_MODULE.
 */
#pragma once

#include <torch/extension.h>

void paged_attention_int8_v1(torch::Tensor& out, const torch::Tensor& query,
                             const torch::Tensor& key_cache,
                             const torch::Tensor& value_cache,
                             const torch::Tensor& key_scale,
                             const torch::Tensor& value_scale, int num_kv_heads,
                             float scale, const torch::Tensor& block_tables,
                             const torch::Tensor& seq_lens, int block_size,
                             int max_seq_len);
