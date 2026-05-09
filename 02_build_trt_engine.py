import tensorrt as trt

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        print("ONNX parse failed:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return

    config = builder.create_builder_config()

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # TensorRT 8.x 动态 shape 需要 optimization profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    if "coarse" in onnx_path:
        profile.set_shape(
            input_name,
            min=(1, 130, 30, 40),
            opt=(1, 130, 188, 252),
            max=(1, 130, 256, 320),
        )
    else:
        profile.set_shape(
            input_name,
            min=(1, 50, 60, 80),
            opt=(1, 50, 376, 504),
            max=(1, 50, 512, 640),
        )

    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Saved: {engine_path}")


if __name__ == "__main__":
    build_engine("coarse_matcher.onnx", "coarse_matcher_fp16.engine")
    build_engine("fine_matcher.onnx", "fine_matcher_fp16.engine")