import time
import cv2
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import tensorrt as trt
from romatch import tiny_roma_v1_outdoor
import types


class TRTModule(nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_binding_name(0)
        self.output_name = self.engine.get_binding_name(1)

        self.input_idx = self.engine.get_binding_index(self.input_name)
        self.output_idx = self.engine.get_binding_index(self.output_name)

    def __getitem__(self, idx):
        if idx == -1:
            return self
        raise IndexError("TRTModule only supports [-1].")

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()

        ok = self.context.set_binding_shape(self.input_idx, tuple(x.shape))
        if not ok:
            raise RuntimeError(f"set_binding_shape failed, input shape = {tuple(x.shape)}")

        out_shape = (x.shape[0], 3, x.shape[2], x.shape[3])

        y = torch.empty(
            size=out_shape,
            device=x.device,
            dtype=x.dtype
        )

        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = int(x.data_ptr())
        bindings[self.output_idx] = int(y.data_ptr())

        ok = self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=torch.cuda.current_stream().cuda_stream
        )

        if not ok:
            raise RuntimeError("TensorRT inference failed")

        return y


class TRTModuleXFeat(nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_idx = 0
        self.output0_idx = 1
        self.output1_idx = 2

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()

        ok = self.context.set_binding_shape(self.input_idx, tuple(x.shape))
        if not ok:
            raise RuntimeError(f"set_binding_shape failed, input shape = {tuple(x.shape)}")

        B, C, H, W = x.shape

        # 对应 XFeat core 输出
        x2 = torch.empty((B, 24, H // 4, W // 4), device=x.device, dtype=x.dtype)
        feats = torch.empty((B, 64, H // 8, W // 8), device=x.device, dtype=x.dtype)

        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = int(x.data_ptr())
        bindings[self.output0_idx] = int(x2.data_ptr())
        bindings[self.output1_idx] = int(feats.data_ptr())

        ok = self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=torch.cuda.current_stream().cuda_stream
        )

        if not ok:
            raise RuntimeError("TensorRT XFeat inference failed")

        return x2, feats

def trt_forward_single(self, x):
    xfeat = self.xfeat[0]

    with torch.no_grad():
        x = x.mean(dim=1, keepdim=True)
        x = xfeat.norm(x)

    x2, feats = self.xfeat_trt(x)
    return x2, feats

def build_tensorrt_tiny_roma(
    coarse_engine="coarse_matcher_fp16.engine",
    fine_engine="fine_matcher_fp16.engine",
):
    model = tiny_roma_v1_outdoor(device="cuda")
    model.eval()

    model.coarse_matcher = TRTModule(coarse_engine).cuda().eval()
    model.fine_matcher = TRTModule(fine_engine).cuda().eval()

    return model

def build_full_tensorrt_tiny_roma(
    xfeat_engine="xfeat_core_fp16.engine",
    coarse_engine="coarse_matcher_fp16.engine",
    fine_engine="fine_matcher_fp16.engine",
):
    model = tiny_roma_v1_outdoor(device="cuda")
    model.eval()

    model.xfeat_trt = TRTModuleXFeat(xfeat_engine).cuda().eval()

    model.coarse_matcher = TRTModule(coarse_engine).cuda().eval()
    model.fine_matcher = TRTModule(fine_engine).cuda().eval()

    model.forward_single = types.MethodType(trt_forward_single, model)

    return model

def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(model, imA_path, imB_path, fast_grid_cache=False):
    imgA = cv2.imread(imA_path)
    imgB = cv2.imread(imB_path)

    if imgA is None or imgB is None:
        raise FileNotFoundError("请检查图片路径是否正确")

    H_A, W_A = imgA.shape[:2]
    H_B, W_B = imgB.shape[:2]

    with torch.no_grad():
        sync_if_cuda()
        t0 = time.perf_counter()
        out = model.match(imA_path, imB_path, return_timing=True, fast_grid_cache=fast_grid_cache)

        if len(out) == 3:
            warp, certainty, timing = out
        else:
            warp, certainty = out
            timing = {}

        sync_if_cuda()
        t1 = time.perf_counter()

        matches, certainty = model.sample(warp, certainty)

        kptsA, kptsB = model.to_pixel_coordinates(
            matches, H_A, W_A, H_B, W_B
        )

        ptsA = kptsA.cpu().numpy()
        ptsB = kptsB.cpu().numpy()

        F, mask = cv2.findFundamentalMat(
            ptsA,
            ptsB,
            ransacReprojThreshold=0.2,
            method=cv2.USAC_MAGSAC,
            confidence=0.999999,
            maxIters=10000,
        )

        sync_if_cuda()
        t2 = time.perf_counter()

    return {
        "match_time_ms": (t1 - t0) * 1000,
        "post_time_ms": (t2 - t1) * 1000,
        "total_time_ms": (t2 - t0) * 1000,
        "forward_total_ms": timing.get("forward_total_ms", 0.0),
        "coarse_matcher_ms": timing.get("coarse_matcher_ms", 0.0),
        "fine_matcher_ms": timing.get("fine_matcher_ms", 0.0),
        "xfeat_ms": timing.get("xfeat_ms", 0.0),
        "corr_volume_ms": timing.get("corr_volume_ms", 0.0),
        "pos_embed_ms": timing.get("pos_embed_ms", 0.0),
        "inliers": int(mask.sum()) if mask is not None else 0,
        "num_matches": len(ptsA),
        "F": F,
    }


def benchmark(model, imA_path, imB_path, name, warmup=5, repeat=20, fast_grid_cache=False):
    print(f"\n===== {name} =====")

    for _ in range(warmup):
        run_once(model, imA_path, imB_path, fast_grid_cache)

    results = []
    for _ in range(repeat):
        results.append(run_once(model, imA_path, imB_path, fast_grid_cache))

    match_times = [r["match_time_ms"] for r in results]
    post_times = [r["post_time_ms"] for r in results]
    total_times = [r["total_time_ms"] for r in results]
    coarse_times = [r["coarse_matcher_ms"] for r in results]
    xfeat_times = [r["xfeat_ms"] for r in results]
    corr_volume_times = [r["corr_volume_ms"] for r in results]
    forward_total_times = [r["forward_total_ms"] for r in results]
    pos_embed_times = [r["pos_embed_ms"] for r in results]
    fine_times = [r["fine_matcher_ms"] for r in results]
    last = results[-1]

    print(f"match avg: {sum(match_times) / len(match_times):.3f} ms")
    print(f"xfeat avg: {sum(xfeat_times) / len(xfeat_times):.3f} ms")
    print(f"corr_volume avg: {sum(corr_volume_times) / len(corr_volume_times):.3f} ms")
    print(f"pos_embed avg: {sum(pos_embed_times) / len(pos_embed_times):.3f} ms")
    print(f"coarse matcher avg: {sum(coarse_times) / len(coarse_times):.3f} ms")
    print(f"fine matcher avg:   {sum(fine_times) / len(fine_times):.3f} ms")
    print(f"forward_total avg: {sum(forward_total_times) / len(forward_total_times):.3f} ms")
    print(f"post  avg: {sum(post_times) / len(post_times):.3f} ms")
    print(f"total avg: {sum(total_times) / len(total_times):.3f} ms")
    print(f"inliers: {last['inliers']} / {last['num_matches']}")
    print("Fundamental matrix:\n", last["F"])


if __name__ == "__main__":
    imA_path = "assets/toronto_A.jpg"
    imB_path = "assets/toronto_B.jpg"

    # 1. 原版 GPU PyTorch
    # if torch.cuda.is_available():
    #     model_gpu = tiny_roma_v1_outdoor(device="cuda").eval()
    #     benchmark(
    #         model_gpu,
    #         imA_path,
    #         imB_path,
    #         name="Original PyTorch GPU",
    #         warmup=5,
    #         repeat=20,
    #     )

    # # 2. TensorRT FP16 GPU
    # model_trt = build_tensorrt_tiny_roma(
    #     coarse_engine="coarse_matcher_fp16.engine",
    #     fine_engine="fine_matcher_fp16.engine",
    # )
    # benchmark(
    #     model_trt,
    #     imA_path,
    #     imB_path,
    #     name="Matcher FP16",
    #     warmup=5,
    #     repeat=20,
    # )

    # # 2. TensorRT FP16 GPU
    # model_trt_int8 = build_tensorrt_tiny_roma(
    #     coarse_engine="coarse_matcher_int8.engine",
    #     fine_engine="fine_matcher_int8.engine",
    # )
    # benchmark(
    #     model_trt_int8,
    #     imA_path,
    #     imB_path,
    #     name="Matcher INT8",
    #     warmup=5,
    #     repeat=20,
    # )

    # # 4. XFeat FP16 + Matcher FP16
    # model_xfeat_fp16_matcher_fp16 = build_full_tensorrt_tiny_roma(
    #     xfeat_engine="xfeat_core_fp16.engine",
    #     coarse_engine="coarse_matcher_fp16.engine",
    #     fine_engine="fine_matcher_fp16.engine",
    # )

    # benchmark(
    #     model_xfeat_fp16_matcher_fp16,
    #     imA_path,
    #     imB_path,
    #     name="TensorRT XFeat FP16 + Matcher FP16",
    #     warmup=5,
    #     repeat=20,
    # )


    # # 5. XFeat FP16 + Matcher INT8
    # model_xfeat_fp16_matcher_int8 = build_full_tensorrt_tiny_roma(
    #     xfeat_engine="xfeat_core_fp16.engine",
    #     coarse_engine="coarse_matcher_int8.engine",
    #     fine_engine="fine_matcher_int8.engine",
    # )

    # benchmark(
    #     model_xfeat_fp16_matcher_int8,
    #     imA_path,
    #     imB_path,
    #     name="TensorRT XFeat FP16 + Matcher INT8",
    #     warmup=5,
    #     repeat=20,
    # )


    # 6. XFeat FP16 + Matcher INT8 + fast_grid_cache
    model_xfeat_fp16_matcher_int8 = build_full_tensorrt_tiny_roma(
        xfeat_engine="xfeat_core_fp16.engine",
        coarse_engine="coarse_matcher_int8.engine",
        fine_engine="fine_matcher_int8.engine",
    )

    benchmark(
        model_xfeat_fp16_matcher_int8,
        imA_path,
        imB_path,
        name="TensorRT XFeat FP16 + Matcher INT8 + fast_grid_cache",
        warmup=5,
        repeat=20,
        fast_grid_cache=True
    )

    