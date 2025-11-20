# AI交易系统运维手册

## 文档版本
- **版本**: v1.0
- **创建日期**: 2025-11-20
- **适用系统**: AI Trading System v1.0

---

## 1. 系统概述

### 1.1 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    AI Trading System                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ TS2Vec   │→ │Transform │→ │   PPO    │→ 交易信号   │
│  │  Model   │  │   Model  │  │  Policy  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │         Inference Service                 │          │
│  │  - 模型加载  - 推理优化  - 缓存管理      │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │         Monitoring System                 │          │
│  │  - 性能监控  - 告警系统  - 日志管理      │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 技术栈

- **深度学习框架**: PyTorch 2.0+
- **强化学习**: Stable-Baselines3
- **数据处理**: Pandas, NumPy
- **监控**: psutil, logging
- **部署**: Python 3.9+

---

## 2. 部署指南

### 2.1 环境要求

#### 硬件要求
- **CPU**: 4核心以上
- **内存**: 16GB以上
- **GPU**: NVIDIA GPU (CUDA 11.0+) 或 AMD GPU (ROCm 5.0+) [可选]
- **存储**: 50GB以上可用空间

#### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+) / macOS / Windows 10+
- **Python**: 3.9 或更高版本
- **CUDA**: 11.0+ (如使用NVIDIA GPU)
- **ROCm**: 5.0+ (如使用AMD GPU)

### 2.2 快速部署

#### 步骤1: 克隆项目

```bash
git clone https://github.com/your-org/ai-trader.git
cd ai-trader
```

#### 步骤2: 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

#### 步骤3: 安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤4: 配置系统

编辑 `configs/base_config.yaml`:

```yaml
# GPU配置
gpu:
  device: auto  # auto/cuda/rocm/cpu
  memory_limit: 8192  # MB
  mixed_precision: true

# 推理配置
inference:
  batch_size: 32
  cache_size: 1000
  max_latency_ms: 100

# 监控配置
monitoring:
  enabled: true
  alert_cooldown: 300
  max_error_rate: 0.05
```

#### 步骤5: 执行部署

```bash
python deploy.py --config configs/base_config.yaml
```

#### 步骤6: 启动服务

```bash
python deploy.py --start-service
```

### 2.3 Docker部署（推荐）

#### 构建镜像

```bash
docker build -t ai-trader:latest .
```

#### 运行容器

```bash
docker run -d \
  --name ai-trader \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  ai-trader:latest
```

---

## 3. 运维流程

### 3.1 日常运维

#### 服务启动

```bash
# 方式1: 直接启动
python src/api/inference_service.py

# 方式2: 使用部署脚本
python deploy.py --start-service

# 方式3: 使用systemd (Linux)
sudo systemctl start ai-trader
```

#### 服务停止

```bash
# 方式1: Ctrl+C 停止
# 方式2: 使用systemd
sudo systemctl stop ai-trader
```

#### 服务重启

```bash
sudo systemctl restart ai-trader
```

#### 查看服务状态

```bash
sudo systemctl status ai-trader
```

### 3.2 日志管理

#### 日志目录结构

```
logs/
├── training/          # 训练日志
├── inference/         # 推理日志
├── monitoring/        # 监控日志
└── errors/           # 错误日志
```

#### 查看实时日志

```bash
# 查看推理日志
tail -f logs/inference/inference.log

# 查看错误日志
tail -f logs/errors/error.log

# 查看监控日志
tail -f logs/monitoring/metrics.log
```

#### 日志轮转

系统自动进行日志轮转：
- 单个日志文件最大: 100MB
- 保留历史文件数: 10个
- 压缩旧日志: 是

### 3.3 性能监控

#### 查看系统指标

```python
from src.api.monitoring import SystemMonitor, MonitoringConfig

# 创建监控器
config = MonitoringConfig()
monitor = SystemMonitor(config)

# 获取指标
metrics = monitor.get_metrics()
print(metrics)
```

#### 监控仪表板

```python
from src.api.monitoring import PerformanceDashboard

dashboard = PerformanceDashboard(monitor)
dashboard.print_dashboard()
```

#### 关键指标

| 指标 | 正常范围 | 告警阈值 |
|------|----------|----------|
| 推理延迟 | <50ms | >100ms |
| CPU使用率 | <60% | >80% |
| 内存使用率 | <70% | >80% |
| GPU内存 | <80% | >90% |
| 错误率 | <1% | >5% |

---

## 4. 故障排查

### 4.1 常见问题

#### 问题1: 推理延迟过高

**症状**: 推理延迟 > 100ms

**排查步骤**:
1. 检查GPU是否正常工作
2. 检查批处理大小是否合适
3. 检查模型是否正确加载到GPU
4. 检查系统资源使用情况

**解决方案**:
```python
# 1. 启用GPU加速
config.device = "cuda"

# 2. 增加批处理大小
config.batch_size = 64

# 3. 启用混合精度
config.use_amp = True

# 4. 启用缓存
config.cache_size = 2000
```

#### 问题2: 内存不足

**症状**: OOM (Out of Memory) 错误

**排查步骤**:
1. 检查批处理大小
2. 检查缓存大小
3. 检查是否有内存泄漏

**解决方案**:
```python
# 1. 减小批处理大小
config.batch_size = 16

# 2. 减小缓存大小
config.cache_size = 500

# 3. 定期清理缓存
torch.cuda.empty_cache()
```

#### 问题3: 模型加载失败

**症状**: 无法加载模型文件

**排查步骤**:
1. 检查模型文件是否存在
2. 检查文件权限
3. 检查模型版本兼容性

**解决方案**:
```bash
# 1. 检查文件
ls -lh models/ppo/best_model.pt

# 2. 修复权限
chmod 644 models/ppo/best_model.pt

# 3. 重新下载模型
python scripts/download_models.py
```

#### 问题4: GPU不可用

**症状**: CUDA/ROCm不可用

**排查步骤**:
1. 检查GPU驱动
2. 检查CUDA/ROCm安装
3. 检查PyTorch GPU支持

**解决方案**:
```bash
# 检查NVIDIA GPU
nvidia-smi

# 检查AMD GPU
rocm-smi

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 重新安装PyTorch (ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

### 4.2 错误代码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| E001 | 模型加载失败 | 检查模型文件路径 |
| E002 | GPU内存不足 | 减小批处理大小 |
| E003 | 推理超时 | 增加超时时间 |
| E004 | 数据格式错误 | 检查输入数据格式 |
| E005 | 配置文件错误 | 检查配置文件语法 |

---

## 5. 性能调优

### 5.1 推理优化

#### GPU加速

```python
# 启用GPU
config.device = "cuda"

# 混合精度推理
config.use_amp = True

# 批处理优化
config.batch_size = 64
```

#### 缓存优化

```python
# 启用缓存
config.cache_size = 2000

# 预热缓存
for i in range(100):
    service.predict(sample_data, sample_features)
```

#### 模型优化

```python
from src.models.compression import ModelCompressor

# 模型剪枝
compressor = ModelCompressor(model)
compressor.prune_model(amount=0.3)

# 模型量化
compressor.dynamic_quantization()
```

### 5.2 系统优化

#### CPU优化

```bash
# 设置线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### 内存优化

```python
# 定期清理
import gc
gc.collect()
torch.cuda.empty_cache()
```

#### 网络优化

```bash
# 增加连接数
ulimit -n 65536

# 优化TCP参数
sysctl -w net.core.somaxconn=1024
```

---

## 6. 备份与恢复

### 6.1 备份策略

#### 模型备份

```bash
# 每日备份
0 2 * * * /path/to/backup_models.sh

# 备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf models_backup_$DATE.tar.gz models/
```

#### 配置备份

```bash
# 备份配置
cp -r configs/ configs_backup_$(date +%Y%m%d)/
```

#### 日志备份

```bash
# 压缩旧日志
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# 删除超过30天的日志
find logs/ -name "*.gz" -mtime +30 -delete
```

### 6.2 恢复流程

#### 模型恢复

```bash
# 解压备份
tar -xzf models_backup_20231120.tar.gz

# 验证模型
python scripts/verify_models.py
```

#### 配置恢复

```bash
# 恢复配置
cp -r configs_backup_20231120/ configs/
```

---

## 7. 安全建议

### 7.1 访问控制

- 限制API访问IP
- 使用API密钥认证
- 启用HTTPS

### 7.2 数据安全

- 加密敏感配置
- 定期备份数据
- 限制文件权限

### 7.3 监控告警

- 启用实时监控
- 配置告警通知
- 定期审查日志

---

## 8. 升级指南

### 8.1 版本升级

```bash
# 1. 备份当前版本
./backup.sh

# 2. 拉取新版本
git pull origin main

# 3. 更新依赖
pip install -r requirements.txt --upgrade

# 4. 运行迁移脚本
python scripts/migrate.py

# 5. 重启服务
sudo systemctl restart ai-trader
```

### 8.2 回滚流程

```bash
# 1. 停止服务
sudo systemctl stop ai-trader

# 2. 恢复备份
./restore.sh backup_20231120

# 3. 启动服务
sudo systemctl start ai-trader
```

---

## 9. 联系支持

### 技术支持

- **邮箱**: support@ai-trader.com
- **文档**: https://docs.ai-trader.com
- **GitHub**: https://github.com/your-org/ai-trader

### 紧急联系

- **24/7热线**: +86-xxx-xxxx-xxxx
- **Slack**: #ai-trader-support

---

## 10. 附录

### 10.1 配置文件示例

完整配置文件见 `configs/base_config.yaml`

### 10.2 常用命令

```bash
# 查看服务状态
systemctl status ai-trader

# 查看日志
journalctl -u ai-trader -f

# 重启服务
systemctl restart ai-trader

# 查看资源使用
htop
nvidia-smi
```

### 10.3 性能基准

| 配置 | 延迟 | 吞吐量 | GPU内存 |
|------|------|--------|---------|
| CPU | 80ms | 12 QPS | N/A |
| GPU (FP32) | 15ms | 66 QPS | 2GB |
| GPU (FP16) | 8ms | 125 QPS | 1GB |

---

**文档结束**

本运维手册提供了AI交易系统的完整部署和运维指南。如有问题，请联系技术支持团队。