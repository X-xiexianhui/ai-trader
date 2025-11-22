# 项目同步指南

## 概述

本项目提供了自动同步脚本 [`sync_to_server.sh`](sync_to_server.sh:1)，用于将本地代码变更快速同步到 xj-member 远程服务器。

## 服务器信息

- **服务器地址**: xj-member.bitahub.com
- **SSH 端口**: 42052
- **用户名**: root
- **远程路径**: ~/ai-trader/

## 使用方法

### 方法一：直接运行脚本

```bash
./sync_to_server.sh
```

### 方法二：使用 bash 运行

```bash
bash sync_to_server.sh
```

## 同步内容

脚本会同步以下内容：
- ✅ 所有源代码文件（src/）
- ✅ 配置文件（configs/）
- ✅ 训练脚本（training/）
- ✅ 文档文件（*.md）
- ✅ 依赖配置（requirements.txt）

## 自动排除的内容

根据 [`.gitignore`](.gitignore:1) 配置，以下内容会被自动排除：
- ❌ Python 缓存（__pycache__/, *.pyc）
- ❌ 虚拟环境（venv/, env/）
- ❌ IDE 配置（.vscode/, .idea/）
- ❌ 数据文件（data/raw/*, data/processed/*）
- ❌ 模型文件（*.pth, *.pkl, *.h5）
- ❌ 日志文件（logs/*, *.log）
- ❌ 临时文件（*.tmp, *.bak）

## 脚本特性

1. **交互式确认**: 同步前会显示配置信息并要求确认
2. **进度显示**: 实时显示文件传输进度
3. **彩色输出**: 使用颜色区分不同类型的信息
4. **错误处理**: 自动检测 rsync 是否安装，并提供错误提示
5. **智能排除**: 自动排除不需要同步的文件

## 首次使用前的准备

### 1. 确保 SSH 密钥已配置

检查 SSH 配置文件 `~/.ssh/config` 是否包含以下内容：

```
Host xj-member.bitahub.com
    Port 42052
    User root
    PreferredAuthentications publickey
```

### 2. 测试 SSH 连接

```bash
ssh -p 42052 root@xj-member.bitahub.com
```

如果能成功连接，说明 SSH 配置正确。

### 3. 安装 rsync（如果未安装）

macOS:
```bash
brew install rsync
```

Linux:
```bash
sudo apt-get install rsync  # Ubuntu/Debian
sudo yum install rsync      # CentOS/RHEL
```

## 常见问题

### Q1: 提示 "rsync 未安装"
**解决方案**: 按照上述方法安装 rsync

### Q2: SSH 连接失败
**解决方案**: 
1. 检查网络连接
2. 确认 SSH 密钥已正确配置
3. 验证服务器地址和端口是否正确

### Q3: 权限被拒绝
**解决方案**: 
1. 确认使用正确的用户名（root）
2. 检查 SSH 密钥权限：`chmod 600 ~/.ssh/id_rsa`

### Q4: 同步速度慢
**解决方案**: 
- rsync 会自动跳过未修改的文件，首次同步较慢是正常的
- 后续同步只会传输变更的文件，速度会快很多

## 高级用法

### 仅查看将要同步的文件（不实际同步）

```bash
rsync -avzn --exclude-from='.gitignore' --exclude='.git' --exclude='venv' \
    -e "ssh -p 42052" ./ root@xj-member.bitahub.com:~/ai-trader/
```

注意：添加了 `-n` 参数表示 dry-run（模拟运行）

### 同步特定目录

如果只想同步特定目录，可以修改脚本中的 `LOCAL_PATH` 变量，或直接使用 rsync 命令：

```bash
rsync -avz -e "ssh -p 42052" ./src/ root@xj-member.bitahub.com:~/ai-trader/src/
```

## 自动化同步

### 方法一：使用 Git Hooks

在 `.git/hooks/post-commit` 中添加：

```bash
#!/bin/bash
./sync_to_server.sh
```

每次 git commit 后自动同步。

### 方法二：使用 cron 定时任务

```bash
# 每小时同步一次
0 * * * * cd /Users/xiexianhui/Python/ai-trader && ./sync_to_server.sh
```

### 方法三：使用 fswatch 监控文件变化

```bash
# 安装 fswatch
brew install fswatch

# 监控并自动同步
fswatch -o . | xargs -n1 -I{} ./sync_to_server.sh
```

## 安全建议

1. **定期备份**: 在服务器上定期备份重要数据
2. **版本控制**: 使用 Git 管理代码版本
3. **测试环境**: 建议先在测试环境验证后再同步到生产环境
4. **权限管理**: 确保 SSH 密钥文件权限正确（600）

## 相关文件

- [`sync_to_server.sh`](sync_to_server.sh:1) - 同步脚本
- [`.gitignore`](.gitignore:1) - 排除规则配置
- `~/.ssh/config` - SSH 配置文件

## 技术支持

如遇到问题，请检查：
1. 网络连接状态
2. SSH 配置是否正确
3. 服务器是否可访问
4. 查看脚本输出的错误信息

---

最后更新：2025-01-22