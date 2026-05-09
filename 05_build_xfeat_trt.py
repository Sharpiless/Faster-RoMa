import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)

network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, logger)

with open("xfeat_core_static_sim.onnx", "rb") as f:
    ok = parser.parse(f.read())

if not ok:
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

engine_bytes = builder.build_serialized_network(network, config)

if engine_bytes is None:
    raise RuntimeError("TensorRT build failed")

with open("xfeat_core_fp16.engine", "wb") as f:
    f.write(engine_bytes)

print("Saved xfeat_core_fp16.engine")