#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <string>
#include "kernel.h"

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define HAS_NVTX 1
#else
#define HAS_NVTX 0
#endif

#define GROUP_SIZE 16

struct NvtxRange {
    std::string name;
    explicit NvtxRange(std::string n) : name(std::move(n)) {
#if HAS_NVTX
        nvtxRangePushA(name.c_str());
#endif
    }
    ~NvtxRange() {
#if HAS_NVTX
        nvtxRangePop();
#endif
    }
};

// Empty scope keeps the leaf name (gpt2_quant.py). nsys/ncu pass "#4:prefill:c_attn".
// Use ':' not '/': ncu treats '/' as nested push/pop, so names with '/' never match.
static std::string nvtx_join(const std::string& scope, const char* leaf) {
    if (scope.empty()) return std::string(leaf);
    return scope + ":" + leaf;
}

static void check(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

// Weight fake-quant: once at model load.
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor W, std::string nvtx = "") {
    NvtxRange op(nvtx_join(nvtx, "quantize_weight"));
    check(W, "W");
    TORCH_CHECK(W.dim() == 2, "W must be (K, N)");
    W = W.contiguous();
    const int K = W.size(0), N = W.size(1);
    TORCH_CHECK(K % GROUP_SIZE == 0, "K must be a multiple of ", GROUP_SIZE);

    auto opts = W.options();
    torch::Tensor Wq, scale, tmax;
    {
        NvtxRange a(nvtx_join(nvtx, "alloc+zeros"));
        Wq    = torch::empty_like(W);
        scale = torch::empty({K / GROUP_SIZE, N}, opts);
        tmax  = torch::zeros({1}, opts);  // atomicMax dest; must start at 0
    }
    auto s = at::cuda::getCurrentCUDAStream();
    {
        NvtxRange k(nvtx_join(nvtx, "TensorAbsMax"));
        launch_absmax(W.data_ptr<float>(), tmax.data_ptr<float>(), K, N, s);
    }
    {
        NvtxRange k(nvtx_join(nvtx, "FakeQuantRowDir"));
        launch_fq_row(W.data_ptr<float>(), tmax.data_ptr<float>(),
                      Wq.data_ptr<float>(), scale.data_ptr<float>(), K, N, s);
    }
    return {Wq, scale};
}

// Config 4: fused fake-quant + cuBLAS GEMM. Every forward.
torch::Tensor qmatmul(torch::Tensor X, torch::Tensor Wq, std::string nvtx = "") {
    NvtxRange op(nvtx_join(nvtx, "qmatmul"));
    check(X, "X"); check(Wq, "Wq");
    auto X2 = X.reshape({-1, X.size(-1)}).contiguous();
    Wq = Wq.contiguous();

    const int M = X2.size(0), K = X2.size(1), N = Wq.size(1);
    TORCH_CHECK(Wq.size(0) == K, "shape mismatch: X(", M, ",", K, ") Wq(", Wq.size(0), ",", N, ")");
    TORCH_CHECK(K % GROUP_SIZE == 0);

    auto opts = X2.options();
    torch::Tensor Xq, scale_a, tmax;
    {
        NvtxRange a(nvtx_join(nvtx, "alloc+zeros"));
        Xq      = torch::empty_like(X2);
        scale_a = torch::empty({M, K / GROUP_SIZE}, opts);
        tmax    = torch::zeros({1}, opts);
    }
    auto s = at::cuda::getCurrentCUDAStream();
    {
        NvtxRange k(nvtx_join(nvtx, "TensorAbsMax"));
        launch_absmax(X2.data_ptr<float>(), tmax.data_ptr<float>(), M, K, s);
    }
    {
        NvtxRange k(nvtx_join(nvtx, "FakeQuantColDir"));
        launch_fq_col(X2.data_ptr<float>(), tmax.data_ptr<float>(),
                      Xq.data_ptr<float>(), scale_a.data_ptr<float>(), M, K, s);
    }
    torch::Tensor C;
    {
        NvtxRange k(nvtx_join(nvtx, "cuBLAS GEMM"));
        C = at::matmul(Xq, Wq);
    }

    auto shape = X.sizes().vec();
    shape.back() = N;
    return C.reshape(shape);
}

// Config 3: fused fake-quant + tiled shared-memory GEMM. Every forward.
torch::Tensor custom_qmatmul(torch::Tensor X, torch::Tensor Wq, std::string nvtx = "") {
    NvtxRange op(nvtx_join(nvtx, "custom_qmatmul"));
    check(X, "X"); check(Wq, "Wq");
    auto X2 = X.reshape({-1, X.size(-1)}).contiguous();
    Wq = Wq.contiguous();

    const int M = X2.size(0), K = X2.size(1), N = Wq.size(1);
    TORCH_CHECK(Wq.size(0) == K, "shape mismatch: X(", M, ",", K, ") Wq(", Wq.size(0), ",", N, ")");
    TORCH_CHECK(K % GROUP_SIZE == 0);

    auto opts = X2.options();
    torch::Tensor Xq, scale_a, tmax, C;
    {
        NvtxRange a(nvtx_join(nvtx, "alloc+zeros"));
        Xq      = torch::empty_like(X2);
        scale_a = torch::empty({M, K / GROUP_SIZE}, opts);
        tmax    = torch::zeros({1}, opts);
        C       = torch::empty({M, N}, opts);
    }
    auto s = at::cuda::getCurrentCUDAStream();
    {
        NvtxRange k(nvtx_join(nvtx, "TensorAbsMax"));
        launch_absmax(X2.data_ptr<float>(), tmax.data_ptr<float>(), M, K, s);
    }
    {
        NvtxRange k(nvtx_join(nvtx, "FakeQuantColDir"));
        launch_fq_col(X2.data_ptr<float>(), tmax.data_ptr<float>(),
                      Xq.data_ptr<float>(), scale_a.data_ptr<float>(), M, K, s);
    }
    {
        NvtxRange k(nvtx_join(nvtx, "Custom GEMM"));
        launch_matmul(Xq.data_ptr<float>(), Wq.data_ptr<float>(),
                      C.data_ptr<float>(), M, N, K, s);
    }

    auto shape = X.sizes().vec();
    shape.back() = N;
    return C.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weight", &quantize_weight, py::arg("W"), py::arg("nvtx") = "");
    m.def("qmatmul", &qmatmul, py::arg("X"), py::arg("Wq"), py::arg("nvtx") = "");
    m.def("custom_qmatmul", &custom_qmatmul,
          py::arg("X"), py::arg("Wq"), py::arg("nvtx") = "");
}