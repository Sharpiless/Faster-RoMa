import torch
import torch.nn as nn
import torch.nn.functional as F
from romatch import tiny_roma_v1_outdoor


class XFeatCoreExport(nn.Module):
    def __init__(self, xfeat):
        super().__init__()
        self.skip1 = xfeat.skip1
        self.block1 = xfeat.block1
        self.block2 = xfeat.block2
        self.block3 = xfeat.block3
        self.block4 = xfeat.block4
        self.block5 = xfeat.block5
        self.block_fusion = xfeat.block_fusion

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))

        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        # 固定输入 736x992 时，x3 是 92x124
        x4 = F.interpolate(x4, size=(92, 124), mode="bilinear", align_corners=False)
        x5 = F.interpolate(x5, size=(92, 124), mode="bilinear", align_corners=False)

        feats = self.block_fusion(x3 + x4 + x5)
        return x2, feats


device = "cpu"

model = tiny_roma_v1_outdoor(device=device).eval()
xfeat = model.xfeat[0].eval().to(device)

export_model = XFeatCoreExport(xfeat).eval().to(device)

# 注意：这里输入是 norm 后的灰度图，所以是 1 通道
dummy = torch.randn(2, 1, 736, 992)

torch.onnx.export(
    export_model,
    dummy,
    "xfeat_core_static.onnx",
    input_names=["input"],
    output_names=["x2", "feats"],
    opset_version=17,
    do_constant_folding=True,
)

print("Exported xfeat_core_static.onnx")