/*
 * Extension-level INT8 paged-attention decode test.
 * Builds one MHA and one GQA case, runs paged_attention_int8_v1, synchronizes,
 * asserts cudaGetLastError() == cudaSuccess, and compares against an
 * independent host FP32 dequantized attention implementation.
 */

#include "../../extensions/kernel/paged_attention_int8.cuh"

#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct QuantResult {
  torch::Tensor q;      // int8, same shape as input
  torch::Tensor scale;  // fp16, input shape without last dim
};

QuantResult quantize_tokenwise_symmetric(const torch::Tensor& x) {
  auto xf = x.to(torch::kFloat32);
  auto amax = std::get<0>(xf.abs().max(-1));
  auto raw_scale = amax / 127.0;
  auto zero = torch::zeros({}, torch::kFloat16);
  auto inf =
      torch::full({}, std::numeric_limits<float>::infinity(), torch::kFloat16);
  auto min_scale = torch::nextafter(zero, inf).to(torch::kFloat32);
  raw_scale = torch::clamp_min(raw_scale, min_scale);
  auto candidate = raw_scale.to(torch::kFloat16);
  auto scale = torch::where(candidate.to(torch::kFloat32) < raw_scale,
                            torch::nextafter(candidate, inf), candidate);
  auto q =
      torch::clamp(torch::round(xf / scale.to(torch::kFloat32).unsqueeze(-1)),
                   -127, 127)
          .to(torch::kInt8);
  return {q, scale};
}

// Independent host FP32 dequantized attention oracle. It reconstructs logical
// K/V from the packed int8 payload and fp16 scales, then computes softmax
// attention entirely on the host in double precision.
torch::Tensor host_reference(const torch::Tensor& query,
                             const torch::Tensor& k_q, const torch::Tensor& v_q,
                             const torch::Tensor& k_scale,
                             const torch::Tensor& v_scale,
                             const torch::Tensor& block_tables,
                             const torch::Tensor& seq_lens, int num_kv_heads,
                             float scale, int block_size, int head_dim) {
  auto q = query.to(torch::kFloat32).cpu();
  auto kq = k_q.cpu();
  auto vq = v_q.cpu();
  auto ks = k_scale.to(torch::kFloat32).cpu();
  auto vs = v_scale.to(torch::kFloat32).cpu();
  auto bt = block_tables.cpu().to(torch::kInt64);
  auto sl = seq_lens.cpu().to(torch::kInt64);

  int batch = static_cast<int>(q.size(0));
  int num_heads = static_cast<int>(q.size(1));
  int x = static_cast<int>(kq.size(4));
  int head_ratio = num_heads / num_kv_heads;

  auto out = torch::zeros({batch, num_heads, head_dim}, torch::kFloat32);
  auto q_a = q.accessor<float, 3>();
  auto kq_a = kq.accessor<int8_t, 5>();
  auto vq_a = vq.accessor<int8_t, 4>();
  auto ks_a = ks.accessor<float, 3>();
  auto vs_a = vs.accessor<float, 3>();
  auto bt_a = bt.accessor<int64_t, 2>();
  auto sl_a = sl.accessor<int64_t, 1>();
  auto out_a = out.accessor<float, 3>();

  for (int b = 0; b < batch; ++b) {
    int seq_len = static_cast<int>(sl_a[b]);
    for (int h = 0; h < num_heads; ++h) {
      int kvh = h / head_ratio;
      std::vector<double> logits(seq_len);
      double max_logit = -std::numeric_limits<double>::infinity();
      for (int t = 0; t < seq_len; ++t) {
        int blk = t / block_size;
        int tib = t % block_size;
        int phys = static_cast<int>(bt_a[b][blk]);
        double ksv = ks_a[phys][kvh][tib];
        double qk = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          int d_outer = d / x;
          int d_inner = d % x;
          double kval =
              static_cast<double>(kq_a[phys][kvh][d_outer][tib][d_inner]) * ksv;
          qk += static_cast<double>(q_a[b][h][d]) * kval;
        }
        double logit = qk * static_cast<double>(scale);
        logits[t] = logit;
        if (logit > max_logit) max_logit = logit;
      }
      double denom = 0.0;
      for (int t = 0; t < seq_len; ++t)
        denom += std::exp(logits[t] - max_logit);
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int t = 0; t < seq_len; ++t) {
          int blk = t / block_size;
          int tib = t % block_size;
          int phys = static_cast<int>(bt_a[b][blk]);
          double w = std::exp(logits[t] - max_logit) / denom;
          double vsv = vs_a[phys][kvh][tib];
          double vval = static_cast<double>(vq_a[phys][kvh][d][tib]) * vsv;
          acc += w * vval;
        }
        out_a[b][h][d] = static_cast<float>(acc);
      }
    }
  }
  return out;
}

bool run_case(const char* label, int num_heads, int num_kv_heads) {
  torch::manual_seed(19);
  const int batch = 2, head_dim = 128, seq_len = 33, block_size = 16;
  auto device = torch::kCUDA;

  auto query = torch::randn(
      {batch, num_heads, head_dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));

  int pages_per_seq = (seq_len + block_size - 1) / block_size;
  int num_blocks = pages_per_seq * batch;
  auto k = torch::randn(
      {num_blocks, num_kv_heads, head_dim / 8, block_size, 8},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto v = torch::randn(
      {num_blocks, num_kv_heads, head_dim, block_size},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));

  auto k_vectors =
      k.permute({0, 3, 1, 2, 4}).reshape({-1, num_kv_heads, head_dim});
  auto v_vectors =
      v.permute({0, 3, 1, 2}).reshape({-1, num_kv_heads, head_dim});
  auto kq = quantize_tokenwise_symmetric(k_vectors);
  auto vq = quantize_tokenwise_symmetric(v_vectors);

  auto k_q =
      kq.q.reshape({num_blocks, block_size, num_kv_heads, head_dim / 8, 8})
          .permute({0, 2, 3, 1, 4})
          .contiguous();
  auto v_q = vq.q.reshape({num_blocks, block_size, num_kv_heads, head_dim})
                 .permute({0, 2, 3, 1})
                 .contiguous();
  auto k_scale = kq.scale.reshape({num_blocks, block_size, num_kv_heads})
                     .permute({0, 2, 1})
                     .contiguous();
  auto v_scale = vq.scale.reshape({num_blocks, block_size, num_kv_heads})
                     .permute({0, 2, 1})
                     .contiguous();

  auto block_tables =
      torch::arange(num_blocks, torch::TensorOptions().dtype(torch::kInt32))
          .reshape({batch, pages_per_seq})
          .to(device);
  auto seq_lens =
      torch::full({batch}, seq_len, torch::TensorOptions().dtype(torch::kInt32))
          .to(device);

  auto out = torch::zeros_like(query);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  paged_attention_int8_v1(out, query, k_q, v_q, k_scale, v_scale, num_kv_heads,
                          scale, block_tables, seq_lens, block_size, seq_len);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("FAIL %s: cuda error %s\n", label, cudaGetErrorString(err));
    return false;
  }

  auto ref =
      host_reference(query, k_q, v_q, k_scale, v_scale, block_tables, seq_lens,
                     num_kv_heads, scale, block_size, head_dim);
  auto got = out.to(torch::kFloat32).cpu();
  auto diff = (got - ref).abs();
  double max_abs = diff.max().item<double>();
  double max_ref = ref.abs().max().item<double>();
  double tol = 2e-2 + 2e-2 * max_ref;
  if (max_abs > tol) {
    std::printf("FAIL %s: max_abs_diff=%g tol=%g\n", label, max_abs, tol);
    return false;
  }
  std::printf("PASS int8_sym %s (max_abs_diff=%g)\n", label, max_abs);
  return true;
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::printf("SKIP: CUDA not available\n");
    return 0;
  }
  bool ok = true;
  ok &= run_case("MHA", 8, 8);
  ok &= run_case("GQA", 32, 8);
  if (!ok) {
    std::printf("ERROR: one or more cases failed\n");
    return 1;
  }
  std::printf("ALL INT8 PAGED ATTENTION CASES PASSED\n");
  return 0;
}
