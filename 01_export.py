import torch
torch.backends.cudnn.enabled = False
from romatch import tiny_roma_v1_outdoor

device = torch.device("cuda:0")

model = tiny_roma_v1_outdoor(device=device).eval()
model = model.to(device)

coarse = model.coarse_matcher.to(device).eval()
fine = model.fine_matcher.to(device).eval()

# 检查是否还有 CPU tensor
for name, p in coarse.named_parameters():
    print("coarse param", name, p.device)

for name, b in coarse.named_buffers():
    print("coarse buffer", name, b.device)

for name, p in fine.named_parameters():
    print("fine param", name, p.device)

for name, b in fine.named_buffers():
    print("fine buffer", name, b.device)

dummy_coarse = torch.randn(1, 130, 60, 80, device=device)
dummy_fine = torch.randn(1, 50, 120, 160, device=device)

torch.onnx.export(
    coarse,
    dummy_coarse,
    "coarse_matcher.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=False,
    dynamic_axes={
        "input": {0: "B", 2: "H", 3: "W"},
        "output": {0: "B", 2: "H", 3: "W"},
    }
)

torch.onnx.export(
    fine,
    dummy_fine,
    "fine_matcher.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=False,
    dynamic_axes={
        "input": {0: "B", 2: "H", 3: "W"},
        "output": {0: "B", 2: "H", 3: "W"},
    }
)