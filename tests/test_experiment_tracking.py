"""
测试实验跟踪功能
Test Experiment Tracking
"""
import os
import sys
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_mlflow():
    """测试MLflow基本功能"""
    import mlflow
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 设置跟踪URI
        mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")
        
        # 创建实验
        experiment_name = "test_experiment"
        mlflow.set_experiment(experiment_name)
        
        # 开始运行
        with mlflow.start_run(run_name="test_run"):
            # 记录参数
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("batch_size", 32)
            
            # 记录指标
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)
            
            # 记录标签
            mlflow.set_tag("model", "test_model")
            
        print("✓ MLflow test passed")
        return True
        
    except Exception as e:
        print(f"✗ MLflow test failed: {e}")
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_tensorboard():
    """测试TensorBoard基本功能"""
    from torch.utils.tensorboard import SummaryWriter
    import torch
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建SummaryWriter
        writer = SummaryWriter(log_dir=temp_dir)
        
        # 记录标量
        for i in range(10):
            writer.add_scalar('Loss/train', 1.0 / (i + 1), i)
            writer.add_scalar('Accuracy/train', i * 0.1, i)
        
        # 记录直方图
        for i in range(5):
            x = torch.randn(100)
            writer.add_histogram('distribution', x, i)
        
        # 关闭writer
        writer.close()
        
        print("✓ TensorBoard test passed")
        return True
        
    except Exception as e:
        print(f"✗ TensorBoard test failed: {e}")
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """运行所有测试"""
    print("Testing Experiment Tracking Tools...")
    print("-" * 50)
    
    mlflow_ok = test_mlflow()
    tensorboard_ok = test_tensorboard()
    
    print("-" * 50)
    if mlflow_ok and tensorboard_ok:
        print("✓ All experiment tracking tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())