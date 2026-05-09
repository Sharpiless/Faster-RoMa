import cv2
import torch
import torch.nn as nn
import tensorrt as trt
from romatch import tiny_roma_v1_outdoor

torch.backends.cudnn.enabled = False


class TRTModule(nn.Module):
    def __init__(self, engine_path):
        super().__init__()

        # 给 TinyRoMa.device 用
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
        # 兼容 self.fine_matcher[-1].weight.device
        if idx == -1:
            return self
        raise IndexError("TRTModule only supports [-1] for compatibility.")

    def forward(self, x):
        assert x.is_cuda, "TensorRT 输入必须在 CUDA 上"
        x = x.contiguous()

        ok = self.context.set_binding_shape(self.input_idx, tuple(x.shape))
        if not ok:
            raise RuntimeError(f"set_binding_shape failed, input shape = {tuple(x.shape)}")

        # 不用 get_binding_shape，直接手动确定输出
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


def build_tensorrt_tiny_roma(
    coarse_engine="coarse_matcher_fp16.engine",
    fine_engine="fine_matcher_fp16.engine",
):
    device = "cuda"

    model = tiny_roma_v1_outdoor(device=device)
    model.eval()

    # 替换两个卷积 matcher
    model.coarse_matcher = TRTModule(coarse_engine).cuda().eval()
    model.fine_matcher = TRTModule(fine_engine).cuda().eval()

    return model


def run_matching(model, imA_path, imB_path):
    imgA = cv2.imread(imA_path)
    imgB = cv2.imread(imB_path)

    if imgA is None or imgB is None:
        raise FileNotFoundError("请检查图片路径是否正确")

    H_A, W_A = imgA.shape[:2]
    H_B, W_B = imgB.shape[:2]

    with torch.no_grad():
        warp, certainty = model.match(imA_path, imB_path)

        matches, certainty = model.sample(warp, certainty)

        kptsA, kptsB = model.to_pixel_coordinates(
            matches,
            H_A,
            W_A,
            H_B,
            W_B,
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

    print("Fundamental matrix:\n", F)
    print("Inliers:", int(mask.sum()) if mask is not None else 0, "/", len(ptsA))

    return F, mask, ptsA, ptsB


if __name__ == "__main__":
    imA_path = "assets/toronto_A.jpg"
    imB_path = "assets/toronto_B.jpg"

    model = build_tensorrt_tiny_roma(
        coarse_engine="coarse_matcher_fp16.engine",
        fine_engine="fine_matcher_fp16.engine",
    )

    print("TensorRT Tiny-RoMa loaded.")

    F, mask, ptsA, ptsB = run_matching(
        model,
        imA_path,
        imB_path
    )