#!/bin/bash

# AI Trader 项目同步脚本
# 用途：将本地项目同步到 xj-member 远程服务器

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 服务器配置
SERVER_HOST="xj-member.bitahub.com"
SERVER_PORT="42052"
SERVER_USER="root"
REMOTE_PATH="~/ai-trader/"

# 本地项目路径（脚本所在目录）
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AI Trader 项目同步工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 rsync 是否安装
if ! command -v rsync &> /dev/null; then
    echo -e "${RED}错误: rsync 未安装${NC}"
    echo "请先安装 rsync: brew install rsync"
    exit 1
fi

# 显示同步信息
echo -e "${YELLOW}同步配置:${NC}"
echo "  本地路径: $LOCAL_PATH"
echo "  远程服务器: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "  远程路径: $REMOTE_PATH"
echo ""

# 询问是否继续
read -p "是否继续同步? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}同步已取消${NC}"
    exit 0
fi

echo -e "${GREEN}开始同步...${NC}"
echo ""

# 执行 rsync 同步
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
    -e "ssh -p $SERVER_PORT" \
    "$LOCAL_PATH" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_PATH"

# 检查同步结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}同步成功完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}提示：${NC}"
    echo "  - 项目已同步到服务器: $REMOTE_PATH"
    echo "  - 可以通过 SSH 登录查看: ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}同步失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}可能的原因：${NC}"
    echo "  1. 网络连接问题"
    echo "  2. SSH 密钥未配置"
    echo "  3. 服务器权限问题"
    echo ""
    exit 1
fi