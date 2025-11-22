"""测试导入"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")
print(f"sys.path: {sys.path[:3]}")

try:
    from src.models.ts2vec.model import TS2VecModel
    print("✓ 成功导入 TS2VecModel")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

try:
    from src.models.ts2vec.training import TS2VecTrainer, OptimizedDataLoader
    print("✓ 成功导入 TS2VecTrainer 和 OptimizedDataLoader")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

try:
    from src.models.ts2vec.data_preparation import TS2VecDataset
    print("✓ 成功导入 TS2VecDataset")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

print("\n所有导入测试完成!")