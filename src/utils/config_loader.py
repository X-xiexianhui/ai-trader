"""
配置加载器
Configuration Loader

支持YAML格式配置文件和环境变量覆盖
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from copy import deepcopy


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 configs/base_config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "configs" / "base_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        if self._config is None:
            self._config = {}
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖配置"""
        # 遍历所有环境变量
        for key, value in os.environ.items():
            # 只处理以 AI_TRADER_ 开头的环境变量
            if key.startswith('AI_TRADER_'):
                # 移除前缀并转换为小写
                config_key = key[len('AI_TRADER_'):].lower()
                
                # 将下划线分隔的键转换为嵌套字典路径
                keys = config_key.split('__')
                
                # 设置配置值
                self._set_nested_value(keys, value)
    
    def _set_nested_value(self, keys: list, value: str) -> None:
        """
        设置嵌套字典的值
        
        Args:
            keys: 键路径列表
            value: 要设置的值
        """
        current = self._config
        
        # 遍历到倒数第二个键
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置最后一个键的值
        # 尝试转换值类型
        final_key = keys[-1]
        current[final_key] = self._convert_value(value)
    
    def _convert_value(self, value: Any) -> Any:
        """
        转换字符串值为适当的类型
        
        Args:
            value: 值（字符串或其他类型）
            
        Returns:
            转换后的值
        """
        # 如果不是字符串，直接返回
        if not isinstance(value, str):
            return value
        
        # 尝试转换为布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 返回原始字符串
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键 (如 'data.root_dir')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        self._set_nested_value(keys, value)
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            配置字典的深拷贝
        """
        return deepcopy(self._config)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_dict: 配置字典
        """
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """
        深度更新字典
        
        Args:
            base: 基础字典
            update: 更新字典
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出文件路径，默认覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def load_additional_config(self, config_path: Union[str, Path]) -> None:
        """
        加载额外的配置文件并合并
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            additional_config = yaml.safe_load(f)
        
        if additional_config:
            self.update(additional_config)
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigLoader(config_path='{self.config_path}')"


# 全局配置实例
_global_config = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigLoader实例
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    
    return _global_config


def reset_config() -> None:
    """重置全局配置实例"""
    global _global_config
    _global_config = None


if __name__ == "__main__":
    # 测试配置加载器
    config = ConfigLoader()
    
    print("配置加载器测试")
    print("-" * 50)
    
    # 测试获取配置
    print(f"项目名称: {config.get('project.name')}")
    print(f"数据根目录: {config.get('data.root_dir')}")
    print(f"批次大小: {config.get('training.batch_size')}")
    print(f"设备类型: {config.get('environment.device.type')}")
    
    # 测试设置配置
    config.set('training.batch_size', 64)
    print(f"修改后的批次大小: {config.get('training.batch_size')}")
    
    # 测试字典式访问
    print(f"学习率: {config['training.learning_rate']}")
    
    # 测试默认值
    print(f"不存在的键: {config.get('non.existent.key', 'default_value')}")
    
    print("-" * 50)
    print("✓ 配置加载器测试完成")