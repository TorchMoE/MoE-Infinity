/*
 * INT8 symmetric paged-attention decode kernel with FP32 accumulation.
 * Storage contract:
 *   key_cache:   int8  [num_blocks, num_kv_heads, head_size/x, block_size, x]
 *   value_cache: int8  [num_blocks, num_kv_heads, head_size, block_size]
 *   key_scale:   fp16  [num_blocks, num_kv_heads, block_size]
 *   value_scale: fp16  [num_blocks, num_kv_heads, block_size]
 * Each stored int8 element is dequantized as value * scale[token, head] before
 * accumulation, mirroring quantize_tokenwise_symmetric. This file owns the
 * device kernel and its launcher only; it defines no PYBIND11_MODULE.
 */

#include "paged_attention_int8.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace archer {
namespace attention {

template <typename scalar_t>
__global__ void paged_attention_int8_v1_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const int8_t* __restrict__ k_cache, const int8_t* __restrict__ v_cache,
    const at::Half* __restrict__ k_scale, const at::Half* __restrict__ v_scale,
    const int num_kv_heads, const float scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, const int max_num_blocks_per_seq,
    const int block_size, const int x, const int64_t out_stride0,
    const int64_t out_stride1, const int64_t out_stride2,
    const int64_t q_stride0, const int64_t q_stride1, const int64_t q_stride2,
    const int64_t k_stride0, const int64_t k_stride1, const int64_t k_stride2,
    const int64_t k_stride3, const int64_t k_stride4, const int64_t v_stride0,
    const int64_t v_stride1, const int64_t v_stride2, const int64_t v_stride3,
    const int64_t ks_stride0, const int64_t ks_stride1,
    const int64_t ks_stride2, const int64_t vs_stride0,
    const int64_t vs_stride1, const int64_t vs_stride2, const int head_size,
    const int num_heads) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;

  if (threadIdx.x != 0) {
    return;
  }

  const int seq_len = static_cast<int>(seq_lens[seq_idx]);
  if (seq_len <= 0) {
    for (int d = 0; d < head_size; ++d) {
      const int64_t out_idx = seq_idx * out_stride0 + head_idx * out_stride1 +
                              static_cast<int64_t>(d) * out_stride2;
      out[out_idx] = static_cast<scalar_t>(0);
    }
    return;
  }

  const int num_queries_per_kv_head = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv_head;

  float max_logit = -std::numeric_limits<float>::infinity();

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int block_idx = token_idx / block_size;
    const int token_in_block = token_idx % block_size;
    const int32_t physical_block =
        block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

    const int64_t ks_idx = static_cast<int64_t>(physical_block) * ks_stride0 +
                           kv_head_idx * ks_stride1 +
                           static_cast<int64_t>(token_in_block) * ks_stride2;
    const float ks = static_cast<float>(k_scale[ks_idx]);

    float qk = 0.0f;
    for (int d = 0; d < head_size; ++d) {
      const int d_outer = d / x;
      const int d_inner = d % x;

      const int64_t q_idx = seq_idx * q_stride0 + head_idx * q_stride1 +
                            static_cast<int64_t>(d) * q_stride2;
      const int64_t k_idx = static_cast<int64_t>(physical_block) * k_stride0 +
                            kv_head_idx * k_stride1 +
                            static_cast<int64_t>(d_outer) * k_stride2 +
                            static_cast<int64_t>(token_in_block) * k_stride3 +
                            static_cast<int64_t>(d_inner) * k_stride4;

      const float k = static_cast<float>(k_cache[k_idx]) * ks;
      qk += static_cast<float>(q[q_idx]) * k;
    }
    max_logit = fmaxf(max_logit, qk * scale);
  }

  float denom = 0.0f;
  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int block_idx = token_idx / block_size;
    const int token_in_block = token_idx % block_size;
    const int32_t physical_block =
        block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

    const int64_t ks_idx = static_cast<int64_t>(physical_block) * ks_stride0 +
                           kv_head_idx * ks_stride1 +
                           static_cast<int64_t>(token_in_block) * ks_stride2;
    const float ks = static_cast<float>(k_scale[ks_idx]);

    float qk = 0.0f;
    for (int d = 0; d < head_size; ++d) {
      const int d_outer = d / x;
      const int d_inner = d % x;

      const int64_t q_idx = seq_idx * q_stride0 + head_idx * q_stride1 +
                            static_cast<int64_t>(d) * q_stride2;
      const int64_t k_idx = static_cast<int64_t>(physical_block) * k_stride0 +
                            kv_head_idx * k_stride1 +
                            static_cast<int64_t>(d_outer) * k_stride2 +
                            static_cast<int64_t>(token_in_block) * k_stride3 +
                            static_cast<int64_t>(d_inner) * k_stride4;

      const float k = static_cast<float>(k_cache[k_idx]) * ks;
      qk += static_cast<float>(q[q_idx]) * k;
    }
    const float logit = qk * scale;
    denom += expf(logit - max_logit);
  }

  if (denom <= 0.0f) {
    for (int d = 0; d < head_size; ++d) {
      const int64_t out_idx = seq_idx * out_stride0 + head_idx * out_stride1 +
                              static_cast<int64_t>(d) * out_stride2;
      out[out_idx] = static_cast<scalar_t>(0);
    }
    return;
  }

  for (int d_out = 0; d_out < head_size; ++d_out) {
    float acc = 0.0f;
    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
      const int block_idx = token_idx / block_size;
      const int token_in_block = token_idx % block_size;
      const int32_t physical_block =
          block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

      const int64_t ks_idx = static_cast<int64_t>(physical_block) * ks_stride0 +
                             kv_head_idx * ks_stride1 +
                             static_cast<int64_t>(token_in_block) * ks_stride2;
      const float ks = static_cast<float>(k_scale[ks_idx]);

      float qk = 0.0f;
      for (int d = 0; d < head_size; ++d) {
        const int d_outer = d / x;
        const int d_inner = d % x;

        const int64_t q_idx = seq_idx * q_stride0 + head_idx * q_stride1 +
                              static_cast<int64_t>(d) * q_stride2;
        const int64_t k_idx = static_cast<int64_t>(physical_block) * k_stride0 +
                              kv_head_idx * k_stride1 +
                              static_cast<int64_t>(d_outer) * k_stride2 +
                              static_cast<int64_t>(token_in_block) * k_stride3 +
                              static_cast<int64_t>(d_inner) * k_stride4;

        const float k = static_cast<float>(k_cache[k_idx]) * ks;
        qk += static_cast<float>(q[q_idx]) * k;
      }

      const float weight = expf(qk * scale - max_logit) / denom;
      const int64_t vs_idx = static_cast<int64_t>(physical_block) * vs_stride0 +
                             kv_head_idx * vs_stride1 +
                             static_cast<int64_t>(token_in_block) * vs_stride2;
      const float vs = static_cast<float>(v_scale[vs_idx]);
      const int64_t v_idx = static_cast<int64_t>(physical_block) * v_stride0 +
                            kv_head_idx * v_stride1 +
                            static_cast<int64_t>(d_out) * v_stride2 +
                            static_cast<int64_t>(token_in_block) * v_stride3;
      const float v = static_cast<float>(v_cache[v_idx]) * vs;
      acc += weight * v;
    }

    const int64_t out_idx = seq_idx * out_stride0 + head_idx * out_stride1 +
                            static_cast<int64_t>(d_out) * out_stride2;
    out[out_idx] = static_cast<scalar_t>(acc);
  }
}

}  // namespace attention
}  // namespace archer

void paged_attention_int8_v1(torch::Tensor& out, const torch::Tensor& query,
                             const torch::Tensor& key_cache,
                             const torch::Tensor& value_cache,
                             const torch::Tensor& key_scale,
                             const torch::Tensor& value_scale, int num_kv_heads,
                             float scale, const torch::Tensor& block_tables,
                             const torch::Tensor& seq_lens, int block_size,
                             int max_seq_len) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA tensor");
  TORCH_CHECK(key_scale.is_cuda(), "key_scale must be CUDA tensor");
  TORCH_CHECK(value_scale.is_cuda(), "value_scale must be CUDA tensor");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA tensor");

  TORCH_CHECK(out.device() == query.device() &&
                  key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  key_scale.device() == query.device() &&
                  value_scale.device() == query.device() &&
                  block_tables.device() == query.device() &&
                  seq_lens.device() == query.device(),
              "all tensors must be on the same CUDA device");

  // The kernel indexes every payload/scale tensor through its explicit strides,
  // so permuted (non-contiguous) canonical layer views are supported directly.
  // block_tables/seq_lens are read with computed row offsets and must be
  // contiguous.
  TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");

  TORCH_CHECK(query.dim() == 3,
              "query must be [num_seqs, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 5,
              "key_cache must be [num_blocks, num_kv_heads, head_size/x, "
              "block_size, x]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must be [num_blocks, num_kv_heads, head_size, block_size]");
  TORCH_CHECK(key_scale.dim() == 3,
              "key_scale must be [num_blocks, num_kv_heads, block_size]");
  TORCH_CHECK(value_scale.dim() == 3,
              "value_scale must be [num_blocks, num_kv_heads, block_size]");
  TORCH_CHECK(block_tables.dim() == 2,
              "block_tables must be [num_seqs, max_num_blocks_per_seq]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_seqs]");

  TORCH_CHECK(query.scalar_type() == out.scalar_type(),
              "query and out dtypes must match");
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16 ||
                  query.scalar_type() == at::kFloat,
              "query must be fp16, bf16, or fp32");
  TORCH_CHECK(key_cache.scalar_type() == at::kChar, "key_cache must be int8");
  TORCH_CHECK(value_cache.scalar_type() == at::kChar,
              "value_cache must be int8");
  TORCH_CHECK(key_scale.scalar_type() == at::kHalf, "key_scale must be fp16");
  TORCH_CHECK(value_scale.scalar_type() == at::kHalf,
              "value_scale must be fp16");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt,
              "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");

  const int num_seqs = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int head_size = static_cast<int>(query.size(2));
  const int key_num_kv_heads = static_cast<int>(key_cache.size(1));
  const int key_head_chunks = static_cast<int>(key_cache.size(2));
  const int key_block_size = static_cast<int>(key_cache.size(3));
  const int x = static_cast<int>(key_cache.size(4));

  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");
  TORCH_CHECK(key_num_kv_heads == num_kv_heads,
              "key_cache num_kv_heads mismatch");
  TORCH_CHECK(value_cache.size(1) == num_kv_heads,
              "value_cache num_kv_heads mismatch");
  TORCH_CHECK(key_block_size == block_size,
              "key_cache block_size mismatch with block_size argument");
  TORCH_CHECK(static_cast<int>(value_cache.size(3)) == block_size,
              "value_cache block_size mismatch with block_size argument");
  TORCH_CHECK(key_head_chunks * x == head_size,
              "key_cache head_size/x * x must equal query head_size");
  TORCH_CHECK(static_cast<int>(value_cache.size(2)) == head_size,
              "value_cache head_size mismatch");
  TORCH_CHECK(static_cast<int>(key_scale.size(1)) == num_kv_heads,
              "key_scale num_kv_heads mismatch");
  TORCH_CHECK(static_cast<int>(value_scale.size(1)) == num_kv_heads,
              "value_scale num_kv_heads mismatch");
  TORCH_CHECK(static_cast<int>(key_scale.size(2)) == block_size,
              "key_scale block_size mismatch");
  TORCH_CHECK(static_cast<int>(value_scale.size(2)) == block_size,
              "value_scale block_size mismatch");
  TORCH_CHECK(static_cast<int>(block_tables.size(0)) == num_seqs,
              "block_tables first dimension must equal num_seqs");
  TORCH_CHECK(static_cast<int>(seq_lens.size(0)) == num_seqs,
              "seq_lens size must equal num_seqs");
  TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");

  const int max_num_blocks_per_seq = static_cast<int>(block_tables.size(1));
  TORCH_CHECK(max_num_blocks_per_seq > 0, "max_num_blocks_per_seq must be > 0");

  TORCH_CHECK(out.size(0) == query.size(0) && out.size(1) == query.size(1) &&
                  out.size(2) == query.size(2),
              "out shape must match query shape");

  c10::cuda::CUDAGuard device_guard(query.device());
  const dim3 grid(num_heads, num_seqs);
  const dim3 block(64);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, query.scalar_type(), "paged_attention_int8_v1",
      [&] {
        archer::attention::paged_attention_int8_v1_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
                key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
                key_scale.data_ptr<at::Half>(),
                value_scale.data_ptr<at::Half>(), num_kv_heads, scale,
                block_tables.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                max_num_blocks_per_seq, block_size, x, out.stride(0),
                out.stride(1), out.stride(2), query.stride(0), query.stride(1),
                query.stride(2), key_cache.stride(0), key_cache.stride(1),
                key_cache.stride(2), key_cache.stride(3), key_cache.stride(4),
                value_cache.stride(0), value_cache.stride(1),
                value_cache.stride(2), value_cache.stride(3),
                key_scale.stride(0), key_scale.stride(1), key_scale.stride(2),
                value_scale.stride(0), value_scale.stride(1),
                value_scale.stride(2), head_size, num_heads);
      });

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "paged_attention_int8_v1 launch failed: ",
              cudaGetErrorString(err));
}
