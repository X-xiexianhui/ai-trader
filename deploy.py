#!/usr/bin/env python3
"""
AI交易系统部署脚本

自动化部署脚本，支持一键部署
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Deployer:
    """部署器"""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        初始化部署器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def check_environment(self) -> bool:
        """检查环境"""
        logger.info("Checking environment...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 9:
            logger.error(f"Python 3.9+ required, got {python_version.major}.{python_version.minor}")
            return False
        logger.info(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查必要的目录
        required_dirs = [
            "data", "models", "logs", "configs", "src"
        ]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                logger.warning(f"Creating directory: {dir_name}")
                dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Directory exists: {dir_name}")
        
        # 检查配置文件
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return False
        logger.info(f"✓ Config file exists: {self.config_path}")
        
        return True
    
    def install_dependencies(self) -> bool:
        """安装依赖"""
        logger.info("Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            return False
    
    def check_models(self) -> bool:
        """检查模型文件"""
        logger.info("Checking model files...")
        
        model_dirs = {
            "TS2Vec": self.project_root / "models" / "ts2vec",
            "Transformer": self.project_root / "models" / "transformer",
            "PPO": self.project_root / "models" / "ppo"
        }
        
        all_exist = True
        for model_name, model_dir in model_dirs.items():
            if model_dir.exists():
                model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
                if model_files:
                    logger.info(f"✓ {model_name} model found: {len(model_files)} file(s)")
                else:
                    logger.warning(f"⚠ {model_name} model directory exists but no model files found")
                    all_exist = False
            else:
                logger.warning(f"⚠ {model_name} model directory not found")
                all_exist = False
        
        return all_exist
    
    def setup_logging(self) -> bool:
        """设置日志"""
        logger.info("Setting up logging...")
        
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 创建日志子目录
        subdirs = ["training", "inference", "monitoring", "errors"]
        for subdir in subdirs:
            (log_dir / subdir).mkdir(exist_ok=True)
        
        logger.info("✓ Logging directories created")
        return True
    
    def run_tests(self) -> bool:
        """运行测试"""
        logger.info("Running tests...")
        
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            logger.warning("Tests directory not found, skipping tests")
            return True
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            if result.returncode == 0:
                logger.info("✓ All tests passed")
                return True
            else:
                logger.warning(f"⚠ Some tests failed:\n{result.stdout}")
                return False
        except FileNotFoundError:
            logger.warning("pytest not installed, skipping tests")
            return True
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False) -> bool:
        """执行部署"""
        logger.info("=" * 60)
        logger.info("Starting deployment...")
        logger.info("=" * 60)
        
        # 1. 检查环境
        if not self.check_environment():
            logger.error("Environment check failed")
            return False
        
        # 2. 安装依赖
        if not self.install_dependencies():
            logger.error("Dependency installation failed")
            return False
        
        # 3. 检查模型
        if not self.check_models():
            logger.warning("Model check failed, but continuing...")
        
        # 4. 设置日志
        if not self.setup_logging():
            logger.error("Logging setup failed")
            return False
        
        # 5. 运行测试
        if not skip_tests:
            if not self.run_tests():
                logger.warning("Tests failed, but continuing...")
        
        logger.info("=" * 60)
        logger.info("✓ Deployment completed successfully!")
        logger.info("=" * 60)
        
        return True
    
    def start_service(self):
        """启动推理服务"""
        logger.info("Starting inference service...")
        
        service_script = self.project_root / "src" / "api" / "inference_service.py"
        if not service_script.exists():
            logger.error("Inference service script not found")
            return False
        
        try:
            subprocess.run(
                [sys.executable, str(service_script)],
                cwd=str(self.project_root)
            )
        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            return False
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Trading System Deployment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--start-service",
        action="store_true",
        help="Start inference service after deployment"
    )
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = Deployer(config_path=args.config)
    
    # 执行部署
    success = deployer.deploy(skip_tests=args.skip_tests)
    
    if not success:
        logger.error("Deployment failed!")
        sys.exit(1)
    
    # 启动服务
    if args.start_service:
        deployer.start_service()


if __name__ == "__main__":
    main()