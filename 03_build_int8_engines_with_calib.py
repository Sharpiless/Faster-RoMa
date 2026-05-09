import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from romatch import tiny_roma_v1_outdoor


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file):
        super().__init__()

        assert len(calibration_data) > 0, "calibration_data is empty"

        self.data = calibration_data
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None

        batch = self.data[self.current_index]
        batch = np.ascontiguousarray(batch.astype(np.float32))

        cuda.memcpy_htod(self.device_input, batch)

        self.current_index += 1

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def list_image_pairs(data_root):
    img1_dir = Path(data_root) / "image1"
    img2_dir = Path(data_root) / "image2"

    if not img1_dir.exists() or not img2_dir.exists():
        raise FileNotFoundError(
            f"Cannot find image1/image2 under {data_root}"
        )

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    imgs1 = []
    for ext in exts:
        imgs1.extend(glob.glob(str(img1_dir / ext)))

    imgs1 = sorted(imgs1)

    pairs = []

    for p1 in imgs1:
        name = Path(p1).name
        p2 = img2_dir / name

        if p2.exists():
            pairs.append((p1, str(p2)))

    if len(pairs) == 0:
        raise RuntimeError(
            "No paired images found. Make sure images1/xxx and images2/xxx have same filenames."
        )

    return pairs


def collect_matcher_inputs(
    data_root,
    max_pairs=32,
    device="cuda",
):
    model = tiny_roma_v1_outdoor(device=device).eval()

    coarse_inputs = []
    fine_inputs = []

    def coarse_hook(module, inputs, output):
        x = inputs[0].detach().float().cpu().numpy()
        coarse_inputs.append(np.ascontiguousarray(x))

    def fine_hook(module, inputs, output):
        x = inputs[0].detach().float().cpu().numpy()
        fine_inputs.append(np.ascontiguousarray(x))

    h1 = model.coarse_matcher.register_forward_hook(coarse_hook)
    h2 = model.fine_matcher.register_forward_hook(fine_hook)

    pairs = list_image_pairs(data_root)
    pairs = pairs[:max_pairs]

    print(f"Found {len(pairs)} calibration pairs.")

    with torch.no_grad():
        for idx, (im1, im2) in enumerate(pairs):
            print(f"[{idx+1}/{len(pairs)}] {Path(im1).name}")

            try:
                model.match(im1, im2)
            except Exception as e:
                print(f"  skip due to error: {e}")

    h1.remove()
    h2.remove()

    if len(coarse_inputs) == 0 or len(fine_inputs) == 0:
        raise RuntimeError("Failed to collect matcher inputs.")

    print(f"Collected coarse calib tensors: {len(coarse_inputs)}")
    print(f"Collected fine calib tensors: {len(fine_inputs)}")
    print("Example coarse shape:", coarse_inputs[0].shape)
    print("Example fine shape:", fine_inputs[0].shape)

    return coarse_inputs, fine_inputs


def build_int8_engine(
    onnx_path,
    engine_path,
    calibration_data,
    cache_file,
    fp16_fallback=True,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        print("ONNX parse failed:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)

    if fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)

    calibrator = EntropyCalibrator(
        calibration_data=calibration_data,
        cache_file=cache_file,
    )

    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    shapes = [x.shape for x in calibration_data]
    b = 1
    c = shapes[0][1]
    hs = [s[2] for s in shapes]
    ws = [s[3] for s in shapes]

    min_shape = (b, c, min(hs), min(ws))
    opt_shape = (b, c, int(np.median(hs)), int(np.median(ws)))
    max_shape = (b, c, max(hs), max(ws))

    print(f"Building {engine_path}")
    print("  min:", min_shape)
    print("  opt:", opt_shape)
    print("  max:", max_shape)

    profile.set_shape(
        input_name,
        min=min_shape,
        opt=opt_shape,
        max=max_shape,
    )

    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        raise RuntimeError(f"TensorRT INT8 build failed: {engine_path}")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Saved: {engine_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root dir containing images1/ and images2/",
    )

    parser.add_argument(
        "--coarse_onnx",
        type=str,
        default="coarse_matcher.onnx",
    )

    parser.add_argument(
        "--fine_onnx",
        type=str,
        default="fine_matcher.onnx",
    )

    parser.add_argument(
        "--coarse_engine",
        type=str,
        default="coarse_matcher_int8.engine",
    )

    parser.add_argument(
        "--fine_engine",
        type=str,
        default="fine_matcher_int8.engine",
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    coarse_inputs, fine_inputs = collect_matcher_inputs(
        data_root=args.data_root,
        max_pairs=args.max_pairs,
        device="cuda",
    )

    build_int8_engine(
        onnx_path=args.coarse_onnx,
        engine_path=args.coarse_engine,
        calibration_data=coarse_inputs,
        cache_file="coarse_matcher_int8.cache",
    )

    build_int8_engine(
        onnx_path=args.fine_onnx,
        engine_path=args.fine_engine,
        calibration_data=fine_inputs,
        cache_file="fine_matcher_int8.cache",
    )


if __name__ == "__main__":
    main()