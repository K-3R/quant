#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "kernel.h"

#define GROUP_SIZE 16

static void check(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

// ---- ① weight: 모델 로드 시 1회 ----
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor W) {
    check(W, "W");
    TORCH_CHECK(W.dim() == 2, "W must be (K, N)");
    W = W.contiguous();
    const int K = W.size(0), N = W.size(1);
    TORCH_CHECK(K % GROUP_SIZE == 0, "K must be a multiple of ", GROUP_SIZE);

    auto opts  = W.options();
    auto Wq    = torch::empty_like(W);
    auto scale = torch::empty({K / GROUP_SIZE, N}, opts);
    auto tmax  = torch::zeros({1}, opts);        // atomicMax용 → 반드시 0 초기화
    auto s     = at::cuda::getCurrentCUDAStream();

    launch_absmax(W.data_ptr<float>(), tmax.data_ptr<float>(), K, N, s);
    launch_fq_row(W.data_ptr<float>(), tmax.data_ptr<float>(),
                  Wq.data_ptr<float>(), scale.data_ptr<float>(), K, N, s);
    return {Wq, scale};
}

// ---- ② activation fake-quant + matmul: forward마다 ----
torch::Tensor qmatmul(torch::Tensor X, torch::Tensor Wq) {
    check(X, "X"); check(Wq, "Wq");
    auto X2 = X.reshape({-1, X.size(-1)}).contiguous();
    Wq = Wq.contiguous();

    const int M = X2.size(0), K = X2.size(1), N = Wq.size(1);
    TORCH_CHECK(Wq.size(0) == K, "shape mismatch: X(", M, ",", K, ") Wq(", Wq.size(0), ",", N, ")");
    TORCH_CHECK(K % GROUP_SIZE == 0);

    auto opts    = X2.options();
    auto Xq      = torch::empty_like(X2);
    auto scale_a = torch::empty({M, K / GROUP_SIZE}, opts);
    auto tmax    = torch::zeros({1}, opts);
    auto s       = at::cuda::getCurrentCUDAStream();

    launch_absmax(X2.data_ptr<float>(), tmax.data_ptr<float>(), M, K, s);
    launch_fq_col(X2.data_ptr<float>(), tmax.data_ptr<float>(),
                  Xq.data_ptr<float>(), scale_a.data_ptr<float>(), M, K, s);
    auto C = at::matmul(Xq, Wq);          // ← cuBLAS SGEMM
    //auto C = at::matmul(Xq.to(at::kHalf), Wq.to(at::kHalf)).to(at::kFloat);

    auto shape = X.sizes().vec();
    shape.back() = N;
    return C.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weight", &quantize_weight);
    m.def("qmatmul", &qmatmul);
}