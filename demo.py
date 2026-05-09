import cv2
import torch
torch.backends.cudnn.enabled = False
from romatch import tiny_roma_v1_outdoor

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 图片路径
imA_path = "assets/toronto_A.jpg"
imB_path = "assets/toronto_B.jpg"

# 读取原图尺寸
imgA = cv2.imread(imA_path)
imgB = cv2.imread(imB_path)

if imgA is None or imgB is None:
    raise FileNotFoundError("请检查图片路径是否正确：assets/toronto_A.jpg / assets/toronto_B.jpg")

H_A, W_A = imgA.shape[:2]
H_B, W_B = imgB.shape[:2]

# 加载 Tiny-RoMa 模型
tiny_roma_model = tiny_roma_v1_outdoor(device=device)

# 匹配
warp, certainty = tiny_roma_model.match(imA_path, imB_path)

# 采样匹配点
matches, certainty = tiny_roma_model.sample(warp, certainty)

# 转换到像素坐标
kptsA, kptsB = tiny_roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

# 转成 numpy
ptsA = kptsA.cpu().numpy()
ptsB = kptsB.cpu().numpy()

# 估计基础矩阵
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

print(tiny_roma_model)