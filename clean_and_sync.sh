#!/bin/bash

# 清理并同步脚本

SERVER_HOST="xj-member.bitahub.com"
SERVER_PORT="42052"
SERVER_USER="root"
REMOTE_PATH="~/ai-trader/"

echo "========================================="
echo "清理服务器 Python 缓存..."
echo "========================================="

# 清理服务器上的 Python 缓存
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd ~/ai-trader
echo "清理 __pycache__ 目录..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "清理 .pyc 文件..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "清理完成！"
EOF

echo ""
echo "========================================="
echo "同步代码到服务器..."
echo "========================================="

# 同步代码
rsync -avz --progress \
    --exclude-from='.gitignore' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.vscode' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.bak' \
    --delete \
    -e "ssh -p $SERVER_PORT" \
    /Users/xiexianhui/Python/ai-trader/ \
    $SERVER_USER@$SERVER_HOST:$REMOTE_PATH

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "同步成功！"
    echo "========================================="
    echo ""
    echo "验证文件是否存在..."
    ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd ~/ai-trader
echo "检查 src/models/__init__.py:"
ls -la src/models/__init__.py
echo ""
echo "检查 src/models/ts2vec/__init__.py:"
ls -la src/models/ts2vec/__init__.py
echo ""
echo "检查 src/models/ts2vec/model.py:"
ls -la src/models/ts2vec/model.py
EOF
else
    echo "同步失败！"
    exit 1
fi