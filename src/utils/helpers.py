"""
工具函数库
Helper Functions

常用的辅助函数，包括随机种子设置、时间处理、文件IO等
"""
import os
import random
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pytz


def set_seed(seed: int = 42) -> None:
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_type: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device_type: 设备类型 (auto, cuda, rocm, cpu)
        
    Returns:
        torch.device对象
    """
    if device_type == "auto":
        # 自动选择最优设备
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # ROCm支持
            return torch.device("cuda")  # ROCm使用cuda接口
        else:
            return torch.device("cpu")
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        return torch.device("cuda")
    elif device_type == "rocm":
        if not (hasattr(torch.version, 'hip') and torch.version.hip is not None):
            raise RuntimeError("ROCm不可用")
        return torch.device("cuda")  # ROCm使用cuda接口
    elif device_type == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"不支持的设备类型: {device_type}")


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取当前时间戳字符串
    
    Args:
        format: 时间格式
        
    Returns:
        时间戳字符串
    """
    return datetime.now().strftime(format)


def parse_datetime(
    date_str: str,
    format: str = "%Y-%m-%d",
    timezone: str = "UTC"
) -> datetime:
    """
    解析日期时间字符串
    
    Args:
        date_str: 日期时间字符串
        format: 日期时间格式
        timezone: 时区
        
    Returns:
        datetime对象
    """
    dt = datetime.strptime(date_str, format)
    tz = pytz.timezone(timezone)
    return tz.localize(dt)


def datetime_to_str(
    dt: datetime,
    format: str = "%Y-%m-%d %H:%M:%S",
    timezone: Optional[str] = None
) -> str:
    """
    将datetime对象转换为字符串
    
    Args:
        dt: datetime对象
        format: 日期时间格式
        timezone: 目标时区（可选）
        
    Returns:
        日期时间字符串
    """
    if timezone:
        tz = pytz.timezone(timezone)
        dt = dt.astimezone(tz)
    
    return dt.strftime(format)


def get_date_range(
    start_date: str,
    end_date: str,
    format: str = "%Y-%m-%d"
) -> List[datetime]:
    """
    获取日期范围内的所有日期
    
    Args:
        start_date: 开始日期字符串
        end_date: 结束日期字符串
        format: 日期格式
        
    Returns:
        日期列表
    """
    start = datetime.strptime(start_date, format)
    end = datetime.strptime(end_date, format)
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    return dates


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        indent: 缩进空格数
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    保存数据为Pickle文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    从Pickle文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_file_size(filepath: Union[str, Path], unit: str = "MB") -> float:
    """
    获取文件大小
    
    Args:
        filepath: 文件路径
        unit: 单位 (B, KB, MB, GB)
        
    Returns:
        文件大小
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    size_bytes = filepath.stat().st_size
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    if unit not in units:
        raise ValueError(f"不支持的单位: {unit}")
    
    return size_bytes / units[unit]


def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为可读字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def format_number(number: float, precision: int = 2) -> str:
    """
    格式化数字为可读字符串（带千位分隔符）
    
    Args:
        number: 数字
        precision: 小数精度
        
    Returns:
        格式化的数字字符串
    """
    return f"{number:,.{precision}f}"


def dict_to_str(d: Dict, indent: int = 2) -> str:
    """
    将字典转换为格式化的字符串
    
    Args:
        d: 字典
        indent: 缩进空格数
        
    Returns:
        格式化的字符串
    """
    return json.dumps(d, indent=indent, ensure_ascii=False)


def merge_dicts(*dicts: Dict) -> Dict:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        
    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    展平嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键
        sep: 分隔符
        
    Returns:
        展平后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """
    获取内存使用情况
    
    Returns:
        内存使用信息字典
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    result = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
        'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
    }
    
    # GPU内存（如果可用）
    if torch.cuda.is_available():
        result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return result


if __name__ == "__main__":
    # 测试工具函数
    print("工具函数库测试")
    print("-" * 50)
    
    # 测试随机种子
    set_seed(42)
    print(f"✓ 随机种子设置: 42")
    print(f"  随机数: {random.random():.6f}")
    print(f"  NumPy随机数: {np.random.rand():.6f}")
    
    # 测试设备获取
    device = get_device("auto")
    print(f"✓ 计算设备: {device}")
    
    # 测试时间函数
    timestamp = get_timestamp()
    print(f"✓ 当前时间戳: {timestamp}")
    
    # 测试日期范围
    dates = get_date_range("2024-01-01", "2024-01-05")
    print(f"✓ 日期范围: {len(dates)}天")
    
    # 测试文件操作
    test_dir = Path("./test_helpers")
    ensure_dir(test_dir)
    
    # 测试JSON
    test_data = {"name": "test", "value": 123}
    json_file = test_dir / "test.json"
    save_json(test_data, json_file)
    loaded_data = load_json(json_file)
    print(f"✓ JSON保存和加载: {loaded_data}")
    
    # 测试Pickle
    pickle_file = test_dir / "test.pkl"
    save_pickle(test_data, pickle_file)
    loaded_pickle = load_pickle(pickle_file)
    print(f"✓ Pickle保存和加载: {loaded_pickle}")
    
    # 测试格式化
    print(f"✓ 格式化时间: {format_time(3665)}")
    print(f"✓ 格式化数字: {format_number(1234567.89)}")
    
    # 测试内存使用
    memory = get_memory_usage()
    print(f"✓ 内存使用: {memory['rss_mb']:.2f} MB")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    
    print("-" * 50)
    print("✓ 工具函数库测试完成")