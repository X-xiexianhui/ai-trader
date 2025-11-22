"""
LSTM + Attention 神经网络模型

功能：
1. 输入：OHLCV K线数据
2. 输出：市场状态向量
3. 辅助任务：
   - 预测未来5根K线的市场状态（上涨/下跌/震荡/反转）
   - 预测未来5根K线的波动率
   - 预测未来5根K线的收益率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AttentionLayer(nn.Module):
    """
    注意力机制层
    
    使用scaled dot-product attention计算序列中每个时间步的重要性权重
    """
    
    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: LSTM隐藏层大小
        """
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            lstm_output: LSTM输出 [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: 加权后的上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        # 计算注意力分数 [batch_size, seq_len, 1]
        attention_scores = self.attention(lstm_output)
        
        # 应用softmax得到注意力权重 [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权求和得到上下文向量 [batch_size, hidden_size]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        # 返回上下文向量和注意力权重（用于可视化）
        return context_vector, attention_weights.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """
    LSTM + Attention 模型
    
    架构：
    1. 输入层：OHLCV数据 (5维)
    2. LSTM层：提取时序特征
    3. Attention层：关注重要时间步
    4. 市场状态向量：编码当前市场状态
    5. 辅助任务头：
       - 分类头：预测市场状态类别
       - 回归头1：预测波动率
       - 回归头2：预测收益率
    """
    
    def __init__(
        self,
        input_size: int = 5,  # OHLCV
        hidden_size: int = 128,
        num_layers: int = 2,
        state_vector_size: int = 64,
        num_classes: int = 4,  # 上涨/下跌/震荡/反转
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 输入特征维度（默认5：OHLCV）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            state_vector_size: 市场状态向量维度
            num_classes: 分类任务类别数
            dropout: Dropout比例
        """
        super(LSTMAttentionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_vector_size = state_vector_size
        self.num_classes = num_classes
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention层
        self.attention = AttentionLayer(hidden_size)
        
        # 市场状态向量投影层
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, state_vector_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 辅助任务1：分类头（预测市场状态）
        self.classifier = nn.Sequential(
            nn.Linear(state_vector_size, state_vector_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_vector_size // 2, num_classes)
        )
        
        # 辅助任务2：波动率预测头
        self.volatility_head = nn.Sequential(
            nn.Linear(state_vector_size, state_vector_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_vector_size // 2, 1)
        )
        
        # 辅助任务3：收益率预测头
        self.return_head = nn.Sequential(
            nn.Linear(state_vector_size, state_vector_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_vector_size // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, seq_len, input_size]
            return_attention: 是否返回注意力权重
            
        Returns:
            字典包含：
            - state_vector: 市场状态向量 [batch_size, state_vector_size]
            - class_logits: 分类logits [batch_size, num_classes]
            - volatility: 波动率预测 [batch_size, 1]
            - returns: 收益率预测 [batch_size, 1]
            - attention_weights: 注意力权重（可选）[batch_size, seq_len]
        """
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Attention机制
        context_vector, attention_weights = self.attention(lstm_out)
        # context_vector: [batch_size, hidden_size]
        
        # 生成市场状态向量
        state_vector = self.state_projection(context_vector)
        # state_vector: [batch_size, state_vector_size]
        
        # 辅助任务1：分类
        class_logits = self.classifier(state_vector)
        
        # 辅助任务2：波动率预测
        volatility = self.volatility_head(state_vector)
        
        # 辅助任务3：收益率预测
        returns = self.return_head(state_vector)
        
        # 构建输出字典
        output = {
            'state_vector': state_vector,
            'class_logits': class_logits,
            'volatility': volatility,
            'returns': returns
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def get_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅获取市场状态向量（用于聚类等下游任务）
        
        Args:
            x: 输入数据 [batch_size, seq_len, input_size]
            
        Returns:
            state_vector: 市场状态向量 [batch_size, state_vector_size]
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            context_vector, _ = self.attention(lstm_out)
            state_vector = self.state_projection(context_vector)
        return state_vector


class MultiTaskLoss(nn.Module):
    """
    多任务学习损失函数
    
    结合三个辅助任务的损失：
    1. 分类损失（交叉熵）
    2. 波动率预测损失（MSE）
    3. 收益率预测损失（MSE）
    
    使用可学习的权重平衡各任务
    """
    
    def __init__(
        self,
        alpha: float = 1.0,  # 分类任务权重
        beta: float = 1.0,   # 波动率任务权重
        gamma: float = 1.0   # 收益率任务权重
    ):
        """
        Args:
            alpha: 分类任务权重
            beta: 波动率任务权重
            gamma: 收益率任务权重
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 定义各任务的损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        self.volatility_loss = nn.MSELoss()
        self.return_loss = nn.MSELoss()
    
    def forward(
        self,
        class_logits: torch.Tensor,
        class_labels: torch.Tensor,
        volatility_pred: torch.Tensor,
        volatility_target: torch.Tensor,
        return_pred: torch.Tensor,
        return_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            class_logits: 分类预测 [batch_size, num_classes]
            class_labels: 分类标签 [batch_size]
            volatility_pred: 波动率预测 [batch_size, 1]
            volatility_target: 波动率目标 [batch_size, 1]
            return_pred: 收益率预测 [batch_size, 1]
            return_target: 收益率目标 [batch_size, 1]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各任务损失的字典
        """
        # 计算各任务损失
        loss_class = self.classification_loss(class_logits, class_labels)
        loss_vol = self.volatility_loss(volatility_pred, volatility_target)
        loss_ret = self.return_loss(return_pred, return_target)
        
        # 加权求和
        total_loss = (
            self.alpha * loss_class +
            self.beta * loss_vol +
            self.gamma * loss_ret
        )
        
        # 返回总损失和各任务损失
        loss_dict = {
            'total': total_loss.item(),
            'classification': loss_class.item(),
            'volatility': loss_vol.item(),
            'returns': loss_ret.item()
        }
        
        return total_loss, loss_dict


def create_model(config: dict = None) -> LSTMAttentionModel:
    """
    创建LSTM+Attention模型的工厂函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        LSTMAttentionModel实例
    """
    if config is None:
        config = {}
    
    model = LSTMAttentionModel(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        state_vector_size=config.get('state_vector_size', 64),
        num_classes=config.get('num_classes', 4),
        dropout=config.get('dropout', 0.2)
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试LSTM+Attention模型...")
    
    # 创建模型
    model = create_model({
        'input_size': 5,
        'hidden_size': 128,
        'num_layers': 2,
        'state_vector_size': 64,
        'num_classes': 4,
        'dropout': 0.2
    })
    
    print(f"\n模型结构:")
    print(model)
    
    # 测试前向传播
    batch_size = 32
    seq_len = 60  # 60根K线
    input_size = 5  # OHLCV
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output = model(x, return_attention=True)
    
    print(f"\n输出:")
    print(f"  - 市场状态向量: {output['state_vector'].shape}")
    print(f"  - 分类logits: {output['class_logits'].shape}")
    print(f"  - 波动率预测: {output['volatility'].shape}")
    print(f"  - 收益率预测: {output['returns'].shape}")
    print(f"  - 注意力权重: {output['attention_weights'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 测试损失函数
    print(f"\n测试多任务损失函数...")
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=1.0)
    
    # 创建假标签
    class_labels = torch.randint(0, 4, (batch_size,))
    volatility_target = torch.randn(batch_size, 1)
    return_target = torch.randn(batch_size, 1)
    
    total_loss, loss_dict = criterion(
        output['class_logits'],
        class_labels,
        output['volatility'],
        volatility_target,
        output['returns'],
        return_target
    )
    
    print(f"损失值:")
    for key, value in loss_dict.items():
        print(f"  - {key}: {value:.4f}")
    
    print("\n模型测试完成！")