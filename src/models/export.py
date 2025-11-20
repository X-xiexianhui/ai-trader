"""
模型导出模块

支持将PyTorch模型导出为ONNX格式，用于跨平台部署
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器"""
    
    def __init__(self, output_dir: str = "models/onnx"):
        """
        初始化模型导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: Tuple,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        verify: bool = True
    ) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            input_shape: 输入形状
            opset_version: ONNX opset版本
            dynamic_axes: 动态轴配置
            input_names: 输入名称列表
            output_names: 输出名称列表
            verify: 是否验证导出的模型
            
        Returns:
            onnx_path: ONNX模型路径
        """
        model.eval()
        
        # 生成输出路径
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        # 创建示例输入
        dummy_input = torch.randn(*input_shape)
        
        # 默认输入输出名称
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        
        # 默认动态轴（支持批次大小动态）
        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }
        
        logger.info(f"Exporting {model_name} to ONNX...")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Opset version: {opset_version}")
        
        try:
            # 导出ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"Model exported to: {onnx_path}")
            
            # 验证ONNX模型
            if verify:
                self.verify_onnx_model(str(onnx_path), dummy_input)
            
            # 优化ONNX模型
            self.optimize_onnx_model(str(onnx_path))
            
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def verify_onnx_model(
        self,
        onnx_path: str,
        test_input: torch.Tensor,
        tolerance: float = 1e-5
    ):
        """
        验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            test_input: 测试输入
            tolerance: 容差
        """
        logger.info("Verifying ONNX model...")
        
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model structure is valid")
            
            # 创建推理会话
            ort_session = ort.InferenceSession(onnx_path)
            
            # 获取输入名称
            input_name = ort_session.get_inputs()[0].name
            
            # ONNX推理
            onnx_output = ort_session.run(
                None,
                {input_name: test_input.numpy()}
            )[0]
            
            logger.info("✓ ONNX model inference successful")
            logger.info(f"  Output shape: {onnx_output.shape}")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise
    
    def optimize_onnx_model(self, onnx_path: str):
        """
        优化ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
        """
        try:
            import onnxoptimizer
            
            logger.info("Optimizing ONNX model...")
            
            # 加载模型
            onnx_model = onnx.load(onnx_path)
            
            # 优化
            passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'extract_constant_to_initializer',
                'eliminate_unused_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ]
            
            optimized_model = onnxoptimizer.optimize(onnx_model, passes)
            
            # 保存优化后的模型
            onnx.save(optimized_model, onnx_path)
            
            logger.info("✓ ONNX model optimized")
            
        except ImportError:
            logger.warning("onnxoptimizer not installed, skipping optimization")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    def compare_outputs(
        self,
        pytorch_model: torch.nn.Module,
        onnx_path: str,
        test_input: torch.Tensor,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        比较PyTorch和ONNX模型输出
        
        Args:
            pytorch_model: PyTorch模型
            onnx_path: ONNX模型路径
            test_input: 测试输入
            tolerance: 容差
            
        Returns:
            comparison: 比较结果
        """
        pytorch_model.eval()
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX推理
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(
            None,
            {input_name: test_input.numpy()}
        )[0]
        
        # 计算差异
        abs_diff = np.abs(pytorch_output - onnx_output)
        rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        is_close = np.allclose(pytorch_output, onnx_output, atol=tolerance)
        
        comparison = {
            "is_close": is_close,
            "max_absolute_diff": float(max_abs_diff),
            "max_relative_diff": float(max_rel_diff),
            "mean_absolute_diff": float(mean_abs_diff),
            "mean_relative_diff": float(mean_rel_diff),
            "tolerance": tolerance
        }
        
        if is_close:
            logger.info("✓ PyTorch and ONNX outputs match within tolerance")
        else:
            logger.warning(f"⚠ Outputs differ: max_abs_diff={max_abs_diff:.2e}")
        
        return comparison
    
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        获取ONNX模型信息
        
        Args:
            onnx_path: ONNX模型路径
            
        Returns:
            info: 模型信息
        """
        onnx_model = onnx.load(onnx_path)
        
        # 获取输入输出信息
        inputs = []
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": inp.type.tensor_type.elem_type
            })
        
        outputs = []
        for out in onnx_model.graph.output:
            shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": out.type.tensor_type.elem_type
            })
        
        # 获取模型大小
        import os
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        info = {
            "path": onnx_path,
            "file_size_mb": file_size_mb,
            "opset_version": onnx_model.opset_import[0].version,
            "inputs": inputs,
            "outputs": outputs,
            "num_nodes": len(onnx_model.graph.node)
        }
        
        return info
    
    def benchmark_onnx(
        self,
        onnx_path: str,
        input_shape: Tuple,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        ONNX模型性能基准测试
        
        Args:
            onnx_path: ONNX模型路径
            input_shape: 输入形状
            num_iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数
            
        Returns:
            results: 基准测试结果
        """
        import time
        
        # 创建推理会话
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        # 生成测试输入
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热
        for _ in range(warmup_iterations):
            _ = ort_session.run(None, {input_name: test_input})
        
        # 基准测试
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = ort_session.run(None, {input_name: test_input})
            times.append(time.time() - start)
        
        times = np.array(times)
        
        results = {
            "mean_latency_ms": float(np.mean(times) * 1000),
            "median_latency_ms": float(np.median(times) * 1000),
            "p95_latency_ms": float(np.percentile(times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(times, 99) * 1000),
            "min_latency_ms": float(np.min(times) * 1000),
            "max_latency_ms": float(np.max(times) * 1000),
            "throughput_qps": float(num_iterations / np.sum(times))
        }
        
        logger.info(f"ONNX Benchmark: {results['mean_latency_ms']:.2f}ms latency, "
                   f"{results['throughput_qps']:.2f} QPS")
        
        return results


def export_ts2vec(model: torch.nn.Module, output_dir: str = "models/onnx"):
    """导出TS2Vec模型"""
    exporter = ModelExporter(output_dir)
    
    onnx_path = exporter.export_to_onnx(
        model=model,
        model_name="ts2vec",
        input_shape=(1, 256, 4),  # [batch, seq_len, features]
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    return onnx_path


def export_transformer(model: torch.nn.Module, output_dir: str = "models/onnx"):
    """导出Transformer模型"""
    exporter = ModelExporter(output_dir)
    
    onnx_path = exporter.export_to_onnx(
        model=model,
        model_name="transformer",
        input_shape=(1, 64, 155),  # [batch, seq_len, features]
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    return onnx_path


def export_ppo(model: torch.nn.Module, output_dir: str = "models/onnx"):
    """导出PPO模型"""
    exporter = ModelExporter(output_dir)
    
    onnx_path = exporter.export_to_onnx(
        model=model,
        model_name="ppo",
        input_shape=(1, 263),  # [batch, state_dim]
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    return onnx_path


def example_usage():
    """使用示例"""
    import torch.nn as nn
    
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # 初始化导出器
    exporter = ModelExporter(output_dir="models/onnx")
    
    # 导出模型
    onnx_path = exporter.export_to_onnx(
        model=model,
        model_name="simple_model",
        input_shape=(1, 100),
        verify=True
    )
    
    print(f"\nModel exported to: {onnx_path}")
    
    # 获取模型信息
    info = exporter.get_model_info(onnx_path)
    print(f"\nModel Info:")
    print(f"File size: {info['file_size_mb']:.2f} MB")
    print(f"Opset version: {info['opset_version']}")
    print(f"Number of nodes: {info['num_nodes']}")
    
    # 比较输出
    test_input = torch.randn(1, 100)
    comparison = exporter.compare_outputs(model, onnx_path, test_input)
    print(f"\nOutput Comparison:")
    print(f"Match: {comparison['is_close']}")
    print(f"Max absolute diff: {comparison['max_absolute_diff']:.2e}")
    
    # 基准测试
    results = exporter.benchmark_onnx(
        onnx_path=onnx_path,
        input_shape=(1, 100),
        num_iterations=1000
    )
    print(f"\nBenchmark Results:")
    print(f"Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Throughput: {results['throughput_qps']:.2f} QPS")


if __name__ == "__main__":
    example_usage()