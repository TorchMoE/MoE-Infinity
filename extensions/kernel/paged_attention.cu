/*
 * PagedAttention V1 CUDA kernel ported from vLLM (Apache 2.0).
 * Original reference:
 * https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

#include "paged_attention_int8.cuh"

namespace archer {
namespace attention {

template <typename scalar_t>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache, const scalar_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, const int max_num_blocks_per_seq,
    const int block_size, const int x, const int64_t out_stride0,
    const int64_t out_stride1, const int64_t out_stride2,
    const int64_t q_stride0, const int64_t q_stride1, const int64_t q_stride2,
    const int64_t k_stride0, const int64_t k_stride1, const int64_t k_stride2,
    const int64_t k_stride3, const int64_t k_stride4, const int64_t v_stride0,
    const int64_t v_stride1, const int64_t v_stride2, const int64_t v_stride3,
    const int head_size, const int num_heads) {
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

      qk += static_cast<float>(q[q_idx]) * static_cast<float>(k_cache[k_idx]);
    }
    max_logit = fmaxf(max_logit, qk * scale);
  }

  float denom = 0.0f;
  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int block_idx = token_idx / block_size;
    const int token_in_block = token_idx % block_size;
    const int32_t physical_block =
        block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

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

      qk += static_cast<float>(q[q_idx]) * static_cast<float>(k_cache[k_idx]);
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

        qk += static_cast<float>(q[q_idx]) * static_cast<float>(k_cache[k_idx]);
      }

      const float weight = expf(qk * scale - max_logit) / denom;
      const int64_t v_idx = static_cast<int64_t>(physical_block) * v_stride0 +
                            kv_head_idx * v_stride1 +
                            static_cast<int64_t>(d_out) * v_stride2 +
                            static_cast<int64_t>(token_in_block) * v_stride3;
      acc += weight * static_cast<float>(v_cache[v_idx]);
    }

    const int64_t out_idx = seq_idx * out_stride0 + head_idx * out_stride1 +
                            static_cast<int64_t>(d_out) * out_stride2;
    out[out_idx] = static_cast<scalar_t>(acc);
  }
}

}  // namespace attention
}  // namespace archer

void paged_attention_v1(torch::Tensor& out, const torch::Tensor& query,
                        const torch::Tensor& key_cache,
                        const torch::Tensor& value_cache, int num_kv_heads,
                        float scale, const torch::Tensor& block_tables,
                        const torch::Tensor& seq_lens, int block_size,
                        int max_seq_len) {
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA tensor");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA tensor");

  TORCH_CHECK(query.dim() == 3,
              "query must be [num_seqs, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 5,
              "key_cache must be [num_blocks, num_kv_heads, head_size/x, "
              "block_size, x]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must be [num_blocks, num_kv_heads, head_size, block_size]");
  TORCH_CHECK(block_tables.dim() == 2,
              "block_tables must be [num_seqs, max_num_blocks_per_seq]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_seqs]");

  TORCH_CHECK(query.scalar_type() == key_cache.scalar_type(),
              "query and key_cache dtypes must match");
  TORCH_CHECK(query.scalar_type() == value_cache.scalar_type(),
              "query and value_cache dtypes must match");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt,
              "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");

  const int num_seqs = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int head_size = static_cast<int>(query.size(2));
  const int key_num_blocks = static_cast<int>(key_cache.size(0));
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
      at::kHalf, at::kBFloat16, query.scalar_type(), "paged_attention_v1", [&] {
        archer::attention::paged_attention_v1_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(), num_kv_heads, scale,
                block_tables.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                max_num_blocks_per_seq, block_size, x, out.stride(0),
                out.stride(1), out.stride(2), query.stride(0), query.stride(1),
                query.stride(2), key_cache.stride(0), key_cache.stride(1),
                key_cache.stride(2), key_cache.stride(3), key_cache.stride(4),
                value_cache.stride(0), value_cache.stride(1),
                value_cache.stride(2), value_cache.stride(3), head_size,
                num_heads);
      });

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "paged_attention_v1 launch failed: ", cudaGetErrorString(err));

  TORCH_CHECK(key_num_blocks > 0, "key_cache must have at least one block");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_v1", &paged_attention_v1,
        "PagedAttention V1 forward pass");
  m.def("paged_attention_int8_v1", &paged_attention_int8_v1,
        "INT8 symmetric paged attention forward pass");
}
