# 安装指南

本文档提供详细的安装步骤，包括Python 3.11.14的安装和环境配置。

## 系统要求

### 硬件要求
- **CPU**: 现代多核处理器 (推荐4核+)
- **内存**: 8GB RAM (推荐16GB+)
- **存储**: 10GB+ 可用空间
- **GPU** (可选，但强烈推荐用于训练):
  - NVIDIA GPU with CUDA 12.6
  - AMD GPU with ROCm 6.0

### 软件要求
- **操作系统**:
  - Linux (Ubuntu 20.04+, CentOS 8+)
  - macOS 11.0+
  - Windows 10/11
- **Python**: 3.11.14 (必需)
- **GPU驱动**:
  - CUDA 12.6 (NVIDIA GPU)
  - ROCm 6.0 (AMD GPU)

## 安装步骤

### 1. 安装Python 3.11.14

#### Linux (Ubuntu/Debian)

```bash
# 添加deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# 验证安装
python3.11 --version
```

#### macOS

```bash
# 使用Homebrew安装
brew install python@3.11

# 验证安装
python3.11 --version
```

#### Windows

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载Python 3.11.14安装包
3. 运行安装程序，确保勾选"Add Python to PATH"
4. 验证安装：
```cmd
python --version
```

### 2. 安装GPU驱动和框架

#### 选项A: NVIDIA GPU (CUDA 12.6)

**Linux:**
```bash
# 下载CUDA 12.6安装包
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# 安装CUDA
sudo sh cuda_12.6.0_560.28.03_linux.run

# 设置环境变量
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

**Windows:**
1. 访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
2. 选择CUDA 12.6版本
3. 下载并运行安装程序
4. 验证安装：
```cmd
nvcc --version
nvidia-smi
```

#### 选项B: AMD GPU (ROCm 6.0)

**Linux (Ubuntu 20.04/22.04):**
```bash
# 添加ROCm仓库
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/focal/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# 安装ROCm
sudo amdgpu-install --usecase=rocm

# 添加用户到render和video组
sudo usermod -a -G render,video $LOGNAME

# 重启系统
sudo reboot

# 验证安装
rocm-smi
rocminfo
```

**支持的AMD GPU:**
- Radeon RX 6000系列 (RDNA 2)
- Radeon RX 7000系列 (RDNA 3)
- Radeon Pro系列
- AMD Instinct系列

**注意**:
- ROCm主要支持Linux系统
- Windows和macOS用户建议使用NVIDIA GPU或CPU版本
- 详细兼容性列表请查看 [ROCm官方文档](https://rocm.docs.amd.com/)

### 3. 克隆项目

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader
```

### 4. 创建虚拟环境

```bash
# 使用Python 3.11.14创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 5. 安装PyTorch (CUDA 12.6)

```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch 2.9.0 with CUDA 12.6
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

**验证PyTorch安装：**

```python
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出：
```
PyTorch版本: 2.9.0
CUDA可用: True
CUDA版本: 12.6
```

### 6. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 7. 验证安装

运行测试脚本验证所有组件：

```bash
# 测试配置加载器
python src/utils/config_loader.py

# 测试日志系统
python src/utils/logger.py

# 测试工具函数
python src/utils/helpers.py

# 测试实验跟踪
python tests/test_experiment_tracking.py
```

## CPU版本安装 (无GPU)

如果没有NVIDIA GPU，可以安装CPU版本的PyTorch：

```bash
# 安装CPU版本的PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

**注意**: CPU版本训练速度会显著慢于GPU版本。

## AMD GPU用户 (ROCm)

如果使用AMD GPU，可以安装ROCm版本：

```bash
# 安装ROCm版本的PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/rocm6.0

# 安装其他依赖
pip install -r requirements.txt
```

## 常见问题

### Q1: Python 3.11.14找不到

**A**: 确保已正确安装Python 3.11.14，并且在PATH中。可以使用`which python3.11`(Linux/macOS)或`where python`(Windows)检查。

### Q2: CUDA版本不匹配

**A**: 确保安装的CUDA版本为12.6。使用`nvcc --version`检查CUDA版本。

### Q3: PyTorch无法检测到GPU

**A**: 
1. 检查NVIDIA驱动是否正确安装：`nvidia-smi`
2. 检查CUDA是否正确安装：`nvcc --version`
3. 确保安装了正确的PyTorch CUDA版本

### Q4: 依赖包安装失败

**A**: 
1. 确保pip已升级到最新版本：`pip install --upgrade pip`
2. 如果某个包安装失败，尝试单独安装：`pip install package_name`
3. 检查网络连接，必要时使用国内镜像源

### Q5: 内存不足

**A**: 
1. 减小batch_size配置
2. 使用梯度累积
3. 启用混合精度训练
4. 考虑使用更大内存的机器

## 性能优化建议

### GPU优化
- 使用CUDA 12.6以获得最佳性能
- 启用混合精度训练(FP16)
- 调整batch_size以充分利用GPU内存
- 使用多GPU并行训练(如果有多个GPU)

### CPU优化
- 设置合适的线程数：`export OMP_NUM_THREADS=4`
- 使用Intel MKL加速：`pip install mkl`

### 内存优化
- 启用数据缓存
- 使用数据加载器的num_workers参数
- 定期清理GPU缓存：`torch.cuda.empty_cache()`

## 下一步

安装完成后，请参考：
- [README.md](README.md) - 快速开始指南
- [design_document.md](design_document.md) - 系统设计文档
- [task.md](task.md) - 开发任务列表

## 获取帮助

如果遇到问题：
1. 查看本文档的常见问题部分
2. 搜索项目的[Issues](https://github.com/yourusername/ai-trader/issues)
3. 提交新的Issue描述你的问题
4. 联系项目维护者

---

祝你使用愉快！🚀