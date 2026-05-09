# Faster-RoMa

基于 TensorRT 的 TinyRoMa 加速与量化部署项目，主要面向实时视觉匹配与嵌入式部署场景。

基于：

- [RoMa](https://github.com/Parskatt/RoMa)
- TinyRoMa
- XFeat

---

## 项目特点

- TensorRT FP16 推理加速
- INT8 Matcher 量化
- ONNX 导出与 ONNX Simplifier 图优化
- Dynamic Shape TensorRT Engine
- Fast Grid Cache 优化
- 模块级 Benchmark
- Warp 可视化保存

---

## 整体优化流程

```text
PyTorch TinyRoMa
        ↓
Export ONNX
        ↓
ONNX Simplifier
        ↓
TensorRT Engine
        ↓
FP16 / INT8 Quantization
        ↓
Fast Inference
```

---

# Benchmark

RTX3090 测试结果：

| XFeat | Matcher | Grid | XFeat (ms) | Corr Volume (ms) | Pos Embed (ms) | Coarse Matcher (ms) | Fine Matcher (ms) | Forward Total (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | FP32 | Ori. | 5.574 | 2.477 | 2.702 | 3.414 | 1.759 | 16.568 |
| FP16 | INT8 | Fast | 1.813 | 2.437 | 2.470 | 0.486 | 0.275 | 8.427 |

整体 Forward 获得约：

- **1.97× 加速**
- XFeat 约 **3× 加速**
- Matcher 约 **8× 加速**

---

# 安装

```bash
git clone https://github.com/Sharpiless/Faster-RoMa.git
cd Faster-RoMa
pip install -r requirements.txt
```

---

# ONNX Simplifier

```bash
python -m onnxsim \
    xfeat_core_static.onnx \
    xfeat_core_static_sim.onnx
```

---

# TensorRT Engine

构建 FP16 Engine：

```bash
python build_engine.py
```

支持：

- FP16
- INT8
- Dynamic Shape

---

# Benchmark

```bash
python benchmark.py
```

包含：

- warmup
- 模块测速
- 可视化保存
- Fundamental Matrix 估计

---

# 可视化

程序会自动保存 warp 可视化结果：

```text
visualizations/
```

---

# TODO

- [ ] Jetson Nano 测试
- [ ] CUDA Kernel 优化
- [ ] Correlation Volume TensorRT Plugin
- [ ] 完整 TensorRT 化

---

# 致谢

本项目基于：

- RoMa
- XFeat
- TensorRT
