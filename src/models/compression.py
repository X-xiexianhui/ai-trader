"""
模型压缩模块

实现模型剪枝、量化等压缩技术
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import copy

logger = logging.getLogger(__name__)


class ModelCompressor:
    """模型压缩器"""
    
    def __init__(self, model: nn.Module):
        """
        初始化模型压缩器
        
        Args:
            model: PyTorch模型
        """
        self.model = model
        self.original_size = self._get_model_size()
        
    def _get_model_size(self) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def prune_model(
        self,
        amount: float = 0.3,
        method: str = "l1_unstructured",
        layers_to_prune: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        剪枝模型
        
        Args:
            amount: 剪枝比例 (0-1)
            method: 剪枝方法 (l1_unstructured/random_unstructured)
            layers_to_prune: 要剪枝的层名称列表
            
        Returns:
            stats: 剪枝统计信息
        """
        logger.info(f"Pruning model with {method}, amount={amount}")
        
        # 如果未指定层，剪枝所有Linear和Conv层
        if layers_to_prune is None:
            layers_to_prune = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    layers_to_prune.append(name)
        
        # 应用剪枝
        pruned_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if name in layers_to_prune:
                if isinstance(module, nn.Linear):
                    if method == "l1_unstructured":
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    elif method == "random_unstructured":
                        prune.random_unstructured(module, name='weight', amount=amount)
                    
                    # 统计
                    mask = module.weight_mask
                    pruned_params += (mask == 0).sum().item()
                    total_params += mask.numel()
                    
                elif isinstance(module, nn.Conv2d):
                    if method == "l1_unstructured":
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    elif method == "random_unstructured":
                        prune.random_unstructured(module, name='weight', amount=amount)
                    
                    mask = module.weight_mask
                    pruned_params += (mask == 0).sum().item()
                    total_params += mask.numel()
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        
        stats = {
            "target_amount": amount,
            "actual_sparsity": actual_sparsity,
            "pruned_params": pruned_params,
            "total_params": total_params,
            "layers_pruned": len(layers_to_prune)
        }
        
        logger.info(f"Pruning complete: {actual_sparsity:.2%} sparsity")
        
        return stats
    
    def make_pruning_permanent(self):
        """使剪枝永久化（移除mask）"""
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        
        logger.info("Pruning made permanent")
    
    def quantize_model(
        self,
        dtype: torch.dtype = torch.qint8,
        backend: str = "fbgemm"
    ) -> Dict[str, Any]:
        """
        量化模型
        
        Args:
            dtype: 量化数据类型
            backend: 量化后端 (fbgemm/qnnpack)
            
        Returns:
            stats: 量化统计信息
        """
        logger.info(f"Quantizing model to {dtype} with {backend} backend")
        
        # 设置量化配置
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 准备量化
        torch.quantization.prepare(self.model, inplace=True)
        
        # 这里需要用代表性数据进行校准
        # 在实际使用中，应该传入校准数据
        logger.warning("Quantization calibration skipped - use calibrate_quantization() with real data")
        
        # 转换为量化模型
        torch.quantization.convert(self.model, inplace=True)
        
        quantized_size = self._get_model_size()
        compression_ratio = self.original_size / quantized_size
        
        stats = {
            "original_size_mb": self.original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "dtype": str(dtype),
            "backend": backend
        }
        
        logger.info(f"Quantization complete: {compression_ratio:.2f}x compression")
        
        return stats
    
    def calibrate_quantization(self, calibration_data: torch.Tensor):
        """
        使用校准数据进行量化校准
        
        Args:
            calibration_data: 校准数据
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(calibration_data)
        
        logger.info("Quantization calibration complete")
    
    def dynamic_quantization(
        self,
        dtype: torch.dtype = torch.qint8
    ) -> Dict[str, Any]:
        """
        动态量化（仅量化权重）
        
        Args:
            dtype: 量化数据类型
            
        Returns:
            stats: 量化统计信息
        """
        logger.info(f"Applying dynamic quantization to {dtype}")
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        
        self.model = quantized_model
        
        quantized_size = self._get_model_size()
        compression_ratio = self.original_size / quantized_size
        
        stats = {
            "original_size_mb": self.original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "dtype": str(dtype),
            "method": "dynamic"
        }
        
        logger.info(f"Dynamic quantization complete: {compression_ratio:.2f}x compression")
        
        return stats
    
    def knowledge_distillation(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        temperature: float = 3.0,
        alpha: float = 0.5,
        epochs: int = 10
    ) -> nn.Module:
        """
        知识蒸馏
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            train_loader: 训练数据加载器
            temperature: 温度参数
            alpha: 蒸馏损失权重
            epochs: 训练轮数
            
        Returns:
            student_model: 训练后的学生模型
        """
        logger.info(f"Starting knowledge distillation for {epochs} epochs")
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 教师模型输出
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                
                # 学生模型输出
                student_output = student_model(data)
                
                # 计算损失
                # 硬标签损失
                loss_ce = criterion_ce(student_output, target)
                
                # 软标签损失（蒸馏损失）
                loss_kl = criterion_kl(
                    torch.log_softmax(student_output / temperature, dim=1),
                    torch.softmax(teacher_output / temperature, dim=1)
                ) * (temperature ** 2)
                
                # 总损失
                loss = alpha * loss_kl + (1 - alpha) * loss_ce
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Knowledge distillation complete")
        
        return student_model
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        current_size = self._get_model_size()
        compression_ratio = self.original_size / current_size
        
        # 计算稀疏度
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        stats = {
            "original_size_mb": self.original_size,
            "current_size_mb": current_size,
            "compression_ratio": compression_ratio,
            "size_reduction_mb": self.original_size - current_size,
            "size_reduction_percent": (1 - current_size / self.original_size) * 100,
            "sparsity": sparsity,
            "total_params": total_params,
            "zero_params": zero_params
        }
        
        return stats
    
    def evaluate_compressed_model(
        self,
        test_loader: torch.utils.data.DataLoader,
        metric_fn: callable
    ) -> Dict[str, float]:
        """
        评估压缩后的模型
        
        Args:
            test_loader: 测试数据加载器
            metric_fn: 评估指标函数
            
        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                all_outputs.append(output)
                all_targets.append(target)
        
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        metrics = metric_fn(all_outputs, all_targets)
        
        return metrics


def example_usage():
    """使用示例"""
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SimpleModel()
    
    # 初始化压缩器
    compressor = ModelCompressor(model)
    
    print(f"Original model size: {compressor.original_size:.2f} MB")
    
    # 剪枝
    prune_stats = compressor.prune_model(amount=0.3, method="l1_unstructured")
    print(f"\nPruning stats:")
    print(f"  Sparsity: {prune_stats['actual_sparsity']:.2%}")
    print(f"  Pruned params: {prune_stats['pruned_params']:,}")
    
    # 使剪枝永久化
    compressor.make_pruning_permanent()
    
    # 动态量化
    quant_stats = compressor.dynamic_quantization()
    print(f"\nQuantization stats:")
    print(f"  Compression ratio: {quant_stats['compression_ratio']:.2f}x")
    print(f"  Size: {quant_stats['quantized_size_mb']:.2f} MB")
    
    # 获取最终统计
    final_stats = compressor.get_compression_stats()
    print(f"\nFinal compression stats:")
    print(f"  Size reduction: {final_stats['size_reduction_percent']:.1f}%")
    print(f"  Compression ratio: {final_stats['compression_ratio']:.2f}x")


if __name__ == "__main__":
    example_usage()