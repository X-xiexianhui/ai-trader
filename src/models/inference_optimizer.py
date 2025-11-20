"""
推理优化模块

实现模型推理性能优化，包括批处理、缓存、GPU加速等
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import logging
from functools import lru_cache
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        use_amp: bool = True,
        cache_size: int = 1000
    ):
        """
        初始化推理优化器
        
        Args:
            model: PyTorch模型
            device: 设备 (cuda/cpu)
            batch_size: 批处理大小
            use_amp: 是否使用混合精度
            cache_size: 缓存大小
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp and device == "cuda"
        
        # 缓存
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 批处理队列
        self.batch_queue = Queue()
        self.result_queue = Queue()
        
        # 性能统计
        self.inference_times = []
        self.throughput_history = []
        
        logger.info(f"Initialized InferenceOptimizer on {device}")
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled")
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        单样本预测（带缓存）
        
        Args:
            x: 输入张量
            
        Returns:
            output: 预测结果
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(x)
        
        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # 推理
        x = x.to(self.device)
        
        start_time = time.time()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(x)
        else:
            output = self.model(x)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 更新缓存
        self._update_cache(cache_key, output)
        
        return output
    
    @torch.no_grad()
    def predict_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        批量预测
        
        Args:
            x_batch: 输入批次 [batch_size, ...]
            
        Returns:
            outputs: 预测结果批次
        """
        x_batch = x_batch.to(self.device)
        
        start_time = time.time()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x_batch)
        else:
            outputs = self.model(x_batch)
        
        inference_time = time.time() - start_time
        throughput = len(x_batch) / inference_time
        
        self.inference_times.append(inference_time)
        self.throughput_history.append(throughput)
        
        return outputs
    
    def predict_streaming(
        self,
        x_list: List[torch.Tensor],
        return_numpy: bool = False
    ) -> List:
        """
        流式批处理预测
        
        Args:
            x_list: 输入列表
            return_numpy: 是否返回numpy数组
            
        Returns:
            results: 预测结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(x_list), self.batch_size):
            batch = x_list[i:i + self.batch_size]
            
            # 堆叠为批次
            x_batch = torch.stack(batch)
            
            # 批量预测
            outputs = self.predict_batch(x_batch)
            
            # 拆分结果
            if return_numpy:
                outputs = outputs.cpu().numpy()
                results.extend([outputs[j] for j in range(len(outputs))])
            else:
                results.extend([outputs[j] for j in range(len(outputs))])
        
        return results
    
    def _generate_cache_key(self, x: torch.Tensor) -> str:
        """生成缓存键"""
        # 使用张量的哈希值作为键
        return str(hash(x.cpu().numpy().tobytes()))
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """更新缓存（LRU策略）"""
        # 如果缓存已满，删除最旧的项
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = value.detach().clone()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        inference_times = np.array(self.inference_times)
        
        stats = {
            "mean_latency_ms": float(np.mean(inference_times) * 1000),
            "median_latency_ms": float(np.median(inference_times) * 1000),
            "p95_latency_ms": float(np.percentile(inference_times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(inference_times, 99) * 1000),
            "min_latency_ms": float(np.min(inference_times) * 1000),
            "max_latency_ms": float(np.max(inference_times) * 1000),
        }
        
        if self.throughput_history:
            throughput = np.array(self.throughput_history)
            stats.update({
                "mean_throughput": float(np.mean(throughput)),
                "max_throughput": float(np.max(throughput))
            })
        
        return stats
    
    def optimize_for_inference(self):
        """优化模型用于推理"""
        # 1. 设置为评估模式
        self.model.eval()
        
        # 2. 禁用梯度计算
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 3. 融合BatchNorm（如果有）
        if hasattr(torch.quantization, 'fuse_modules'):
            try:
                torch.quantization.fuse_modules(self.model, inplace=True)
                logger.info("Fused BatchNorm layers")
            except Exception as e:
                logger.warning(f"Could not fuse modules: {e}")
        
        # 4. JIT编译（可选）
        try:
            dummy_input = torch.randn(1, *self.model.input_shape).to(self.device)
            self.model = torch.jit.trace(self.model, dummy_input)
            logger.info("Model traced with TorchScript")
        except Exception as e:
            logger.warning(f"Could not trace model: {e}")
        
        logger.info("Model optimized for inference")
    
    def benchmark(
        self,
        input_shape: Tuple,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状
            num_iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数
            
        Returns:
            results: 基准测试结果
        """
        logger.info(f"Running benchmark with {num_iterations} iterations...")
        
        # 生成随机输入
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # 预热
        for _ in range(warmup_iterations):
            _ = self.predict(dummy_input)
        
        # 清空统计
        self.inference_times.clear()
        
        # 基准测试
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = self.predict(dummy_input)
        
        total_time = time.time() - start_time
        
        # 计算统计
        results = {
            "total_time_s": total_time,
            "iterations": num_iterations,
            "mean_latency_ms": (total_time / num_iterations) * 1000,
            "throughput_qps": num_iterations / total_time
        }
        
        results.update(self.get_performance_stats())
        
        logger.info(f"Benchmark results: {results['mean_latency_ms']:.2f}ms latency, "
                   f"{results['throughput_qps']:.2f} QPS")
        
        return results
    
    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用"""
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            max_memory_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
            
            return {
                "memory_allocated_mb": memory_allocated,
                "memory_reserved_mb": memory_reserved,
                "max_memory_allocated_mb": max_memory_allocated
            }
        else:
            return {"message": "Memory profiling only available on CUDA"}
    
    def reset_stats(self):
        """重置统计信息"""
        self.inference_times.clear()
        self.throughput_history.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Statistics reset")


class BatchInferenceEngine:
    """批量推理引擎（支持动态批处理）"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化批量推理引擎
        
        Args:
            model: PyTorch模型
            max_batch_size: 最大批次大小
            max_wait_time: 最大等待时间（秒）
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.request_queue = Queue()
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """启动批处理工作线程"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Batch inference engine started")
    
    def stop(self):
        """停止批处理工作线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Batch inference engine stopped")
    
    def predict_async(self, x: torch.Tensor) -> Queue:
        """
        异步预测
        
        Args:
            x: 输入张量
            
        Returns:
            result_queue: 结果队列
        """
        result_queue = Queue()
        self.request_queue.put((x, result_queue))
        return result_queue
    
    def _batch_worker(self):
        """批处理工作线程"""
        while self.running:
            batch_requests = []
            start_time = time.time()
            
            # 收集请求直到达到批次大小或超时
            while len(batch_requests) < self.max_batch_size:
                elapsed = time.time() - start_time
                
                if elapsed >= self.max_wait_time and batch_requests:
                    break
                
                try:
                    timeout = max(0.001, self.max_wait_time - elapsed)
                    request = self.request_queue.get(timeout=timeout)
                    batch_requests.append(request)
                except:
                    if batch_requests:
                        break
            
            if not batch_requests:
                continue
            
            # 批量推理
            inputs = [req[0] for req in batch_requests]
            result_queues = [req[1] for req in batch_requests]
            
            try:
                # 堆叠输入
                batch_input = torch.stack(inputs).to(self.device)
                
                # 推理
                with torch.no_grad():
                    batch_output = self.model(batch_input)
                
                # 分发结果
                for i, result_queue in enumerate(result_queues):
                    result_queue.put(batch_output[i])
                    
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                for result_queue in result_queues:
                    result_queue.put(None)


def example_usage():
    """使用示例"""
    import torch.nn as nn
    
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 10)
            self.input_shape = (100,)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # 初始化优化器
    optimizer = InferenceOptimizer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        use_amp=True,
        cache_size=1000
    )
    
    # 优化模型
    optimizer.optimize_for_inference()
    
    # 基准测试
    results = optimizer.benchmark(
        input_shape=(1, 100),
        num_iterations=1000
    )
    print("\nBenchmark Results:")
    print(f"Mean Latency: {results['mean_latency_ms']:.2f}ms")
    print(f"Throughput: {results['throughput_qps']:.2f} QPS")
    
    # 批量预测
    x_list = [torch.randn(100) for _ in range(100)]
    outputs = optimizer.predict_streaming(x_list)
    print(f"\nProcessed {len(outputs)} samples")
    
    # 缓存统计
    cache_stats = optimizer.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
    
    # 性能统计
    perf_stats = optimizer.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"P95 Latency: {perf_stats['p95_latency_ms']:.2f}ms")
    print(f"P99 Latency: {perf_stats['p99_latency_ms']:.2f}ms")


if __name__ == "__main__":
    example_usage()