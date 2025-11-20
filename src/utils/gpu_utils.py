"""
GPU设备检测与配置工具

支持CUDA（NVIDIA GPU）和ROCm（AMD GPU）
"""

import torch
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU设备管理器"""
    
    def __init__(self):
        self.device = None
        self.device_type = None
        self.device_info = {}
        self._detect_device()
    
    def _detect_device(self):
        """检测可用的GPU设备"""
        # 检测CUDA（NVIDIA GPU）
        if torch.cuda.is_available():
            self.device_type = 'cuda'
            self.device = torch.device('cuda')
            self._get_cuda_info()
            logger.info(f"检测到CUDA设备: {self.device_info['name']}")
        # 检测ROCm（AMD GPU）
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            self.device_type = 'rocm'
            self.device = torch.device('cuda')  # ROCm使用相同的API
            self._get_rocm_info()
            logger.info(f"检测到ROCm设备: {self.device_info['name']}")
        # 使用CPU
        else:
            self.device_type = 'cpu'
            self.device = torch.device('cpu')
            self._get_cpu_info()
            logger.info("未检测到GPU，使用CPU")
    
    def _get_cuda_info(self):
        """获取CUDA设备信息"""
        try:
            device_id = torch.cuda.current_device()
            self.device_info = {
                'name': torch.cuda.get_device_name(device_id),
                'total_memory': torch.cuda.get_device_properties(device_id).total_memory,
                'compute_capability': torch.cuda.get_device_capability(device_id),
                'device_count': torch.cuda.device_count(),
                'current_device': device_id
            }
        except Exception as e:
            logger.error(f"获取CUDA信息失败: {e}")
            self.device_info = {}
    
    def _get_rocm_info(self):
        """获取ROCm设备信息"""
        try:
            device_id = torch.cuda.current_device()
            self.device_info = {
                'name': torch.cuda.get_device_name(device_id),
                'total_memory': torch.cuda.get_device_properties(device_id).total_memory,
                'device_count': torch.cuda.device_count(),
                'current_device': device_id,
                'rocm_version': torch.version.hip if hasattr(torch.version, 'hip') else 'unknown'
            }
        except Exception as e:
            logger.error(f"获取ROCm信息失败: {e}")
            self.device_info = {}
    
    def _get_cpu_info(self):
        """获取CPU信息"""
        import platform
        self.device_info = {
            'name': platform.processor() or 'CPU',
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self.device
    
    def get_device_type(self) -> str:
        """获取设备类型"""
        return self.device_type
    
    def get_device_info(self) -> Dict:
        """获取设备详细信息"""
        return self.device_info
    
    def get_memory_info(self) -> Optional[Dict]:
        """获取显存使用信息（仅GPU）"""
        if self.device_type in ['cuda', 'rocm']:
            try:
                return {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved(),
                    'max_allocated': torch.cuda.max_memory_allocated(),
                    'total': self.device_info.get('total_memory', 0)
                }
            except Exception as e:
                logger.error(f"获取显存信息失败: {e}")
                return None
        return None
    
    def clear_cache(self):
        """清空GPU缓存"""
        if self.device_type in ['cuda', 'rocm']:
            try:
                torch.cuda.empty_cache()
                logger.info("GPU缓存已清空")
            except Exception as e:
                logger.error(f"清空GPU缓存失败: {e}")
    
    def set_device(self, device_id: int = 0):
        """设置使用的GPU设备"""
        if self.device_type in ['cuda', 'rocm']:
            try:
                torch.cuda.set_device(device_id)
                self.device = torch.device(f'cuda:{device_id}')
                logger.info(f"切换到GPU设备 {device_id}")
            except Exception as e:
                logger.error(f"设置GPU设备失败: {e}")
    
    def is_available(self) -> bool:
        """检查GPU是否可用"""
        return self.device_type in ['cuda', 'rocm']
    
    def get_optimal_device(self, prefer_gpu: bool = True) -> torch.device:
        """
        获取最优设备
        
        Args:
            prefer_gpu: 是否优先使用GPU
            
        Returns:
            torch.device: 最优设备
        """
        if prefer_gpu and self.is_available():
            return self.device
        return torch.device('cpu')
    
    def monitor_memory(self) -> str:
        """监控显存使用情况"""
        if not self.is_available():
            return "CPU模式，无显存信息"
        
        mem_info = self.get_memory_info()
        if mem_info:
            allocated_gb = mem_info['allocated'] / 1024**3
            total_gb = mem_info['total'] / 1024**3
            usage_percent = (mem_info['allocated'] / mem_info['total']) * 100
            
            return (f"显存使用: {allocated_gb:.2f}GB / {total_gb:.2f}GB "
                   f"({usage_percent:.1f}%)")
        return "无法获取显存信息"
    
    def print_device_info(self):
        """打印设备信息"""
        print(f"\n{'='*50}")
        print(f"设备类型: {self.device_type.upper()}")
        print(f"{'='*50}")
        
        for key, value in self.device_info.items():
            if key == 'total_memory':
                value_gb = value / 1024**3
                print(f"{key}: {value_gb:.2f} GB")
            else:
                print(f"{key}: {value}")
        
        if self.is_available():
            print(f"\n{self.monitor_memory()}")
        print(f"{'='*50}\n")


def get_device(device_str: Optional[str] = None) -> Tuple[torch.device, str]:
    """
    获取计算设备
    
    Args:
        device_str: 设备字符串 ('cuda', 'rocm', 'cpu', 'auto' 或 None)
        
    Returns:
        Tuple[torch.device, str]: (设备对象, 设备类型)
    """
    manager = GPUManager()
    
    if device_str is None or device_str == 'auto':
        device = manager.get_optimal_device(prefer_gpu=True)
        device_type = manager.get_device_type()
    elif device_str == 'cpu':
        device = torch.device('cpu')
        device_type = 'cpu'
    elif device_str in ['cuda', 'rocm']:
        if manager.is_available():
            device = manager.get_device()
            device_type = manager.get_device_type()
        else:
            logger.warning(f"请求的设备 {device_str} 不可用，回退到CPU")
            device = torch.device('cpu')
            device_type = 'cpu'
    else:
        logger.warning(f"未知的设备类型 {device_str}，使用CPU")
        device = torch.device('cpu')
        device_type = 'cpu'
    
    return device, device_type


def check_gpu_availability() -> Dict[str, bool]:
    """
    检查GPU可用性
    
    Returns:
        Dict[str, bool]: 各种GPU后端的可用性
    """
    return {
        'cuda': torch.cuda.is_available(),
        'rocm': hasattr(torch.version, 'hip') and torch.version.hip is not None,
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    }


# 全局GPU管理器实例
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """获取全局GPU管理器实例"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("检查GPU可用性:")
    availability = check_gpu_availability()
    for backend, available in availability.items():
        print(f"  {backend}: {'✓' if available else '✗'}")
    
    print("\n初始化GPU管理器:")
    manager = get_gpu_manager()
    manager.print_device_info()
    
    print("测试设备获取:")
    device, device_type = get_device('auto')
    print(f"  自动选择: {device} ({device_type})")
    
    if manager.is_available():
        print("\n测试显存监控:")
        # 创建一个测试张量
        test_tensor = torch.randn(1000, 1000).to(device)
        print(f"  创建测试张量后: {manager.monitor_memory()}")
        del test_tensor
        manager.clear_cache()
        print(f"  清空缓存后: {manager.monitor_memory()}")