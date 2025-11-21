#!/bin/bash
# TS2Vec训练快速开始脚本
# 
# 此脚本将执行完整的数据准备和模型训练流程：
# 1. 下载MES数据
# 2. 处理特征
# 3. 训练TS2Vec模型

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TS2Vec训练快速开始"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "建议先激活虚拟环境: source .venv/bin/activate"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 创建输出目录
mkdir -p training/output
mkdir -p models/checkpoints/ts2vec

echo "步骤 1/3: 下载MES数据"
echo "----------------------------------------"
python training/download_mes_data.py
if [ $? -ne 0 ]; then
    echo "❌ 数据下载失败"
    exit 1
fi
echo "✓ 数据下载完成"
echo ""

echo "步骤 2/3: 处理特征"
echo "----------------------------------------"
python training/process_mes_features.py
if [ $? -ne 0 ]; then
    echo "❌ 特征处理失败"
    exit 1
fi
echo "✓ 特征处理完成"
echo ""

echo "步骤 3/3: 训练TS2Vec模型"
echo "----------------------------------------"
python training/train_ts2vec.py
if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败"
    exit 1
fi
echo "✓ 模型训练完成"
echo ""

echo "=========================================="
echo "✓ 所有步骤完成！"
echo "=========================================="
echo ""
echo "输出文件位置："
echo "  - 模型: models/checkpoints/ts2vec/best_model.pt"
echo "  - 训练历史: training/output/ts2vec_training_history.png"
echo "  - 训练摘要: training/output/ts2vec_training_summary.txt"
echo "  - 日志文件: training/output/train_ts2vec.log"
echo ""