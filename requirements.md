# 任务
基于TS2Vec（形态）→ Transformer（状态）→ PPO（执行）训练一个AI交易模型
# 开发语言
python，回测使用第三方量化框架
# 1. 总体架构
Raw OHLCV → TS2Vec（形态编码）→ Transformer（状态建模）
        → PPO（动作决策）→ Trade Execution

目标：构建一个融合“形态 → 状态 → 动作”的智能交易系统，用于 5 分钟级别期货交易。

# 2. 数据输入
2.1 基础数据（OHLC）  
数据来源使用雅虎金融
Open（开盘价）  
High（最高价）  
Low（最低价）  
Close（收盘价）  
# 3. 模型 1：TS2Vec — 形态特征编码
TS2Vec优先训练，训练好后参数保存到本地。后续训练transformer和PPO时，离线加载TS2Vec模型
## 3.1 输入
滑动窗口长度 L（如 128、256 或 512） 
输入维度：OHLC  
例：  
input shape: [batch, seq_len=L, 5] 
## 3.2 输出
时间步级别 embedding： 
[batch, L, D_ts2vec]  
最终用于 Transformer 的 embedding： 
z_t = TS2Vec(x_t) → R^(D_ts2vec)
## 3.3 超参数
| 参数                           | 默认值     |
| ---------------------------- | ------- |
| Window size                  | 128–512 |
| Embedding dimension D_ts2vec | 64–256  |
| Contrastive temperature      | 0.1     |
| Temporal dropout             | 0.2–0.5 |
| Optimizer                    | AdamW   |


# 4. 模型 2：Transformer — 状态建模（含手工特征）
Transformer 的作用不是预测价格，而是将“形态 + 时间序列特征 → 市场状态向量”。
## 4.1 输入（两类）
### A. 来自 TS2Vec 的形态 embedding
z_t ∈ R^(D_ts2vec)，L2 Norm归一化
### B. 手工时间序列特征（强烈推荐，显著提高性能）
#### 1. 价格与收益（5 维）
1. ret_1：1 根 K 的 log return，使用 StandardScaler归一化  
2. ret_5：5 根 K 的 log return  ，使用 StandardScaler归一化  
3. ret_20：20 根 K 的 log return ，使用 StandardScaler归一化   
4. price_slope_20：回归斜率拟合 20 根价格趋势  ，使用 StandardScaler归一化   
5. C_div_MA20：收盘价相对 MA20 的比例（趋势偏离）  ，使用 StandardScaler归一化   
#### 2. 波动率与区间（5 维）
1. ATR14_norm：ATR(14) / Close ，RobustScaler归一化 
2. vol_20：20 根 rolling std（波动率）RobustScaler归一化   
3. range_20_norm：(H20 − L20) / Close(唐奇安通道)  RobustScaler归一化 
4. BB_width_norm：布林带宽度 / Close  RobustScaler归一化  
5. parkinson_vol：Parkinson 波动率（基于高低点） RobustScaler归一化 
#### 3. 均线与动量（4 维）
1. EMA20，RobustScaler归一化  
2. stoch(9,3,3)：随机指标，RobustScaler归一化 
3. MACD(12,26,9)，RobustScaler归一化
4. VWAP，RobustScaler归一化
#### 4. 成交量结构（4 维）
1. volume 成交量，RobustScaler归一化
2. volume_zscore：Volume的 z-score，使用 StandardScaler归一化   
3. volume_change_1：成交量的 1 根变化率，使用 StandardScaler归一化   
4. OBV_slope_20：20 根 OBV 斜率（资金流方向） ，RobustScaler归一化 
#### 5. K 线位置结构（6 维）
1. pos_in_range_20：C 在最近 20 根区间中的相对位置，使用 StandardScaler归一化     
2. dist_to_HH20_norm：(C − HH20) / Close 使用 StandardScaler归一化    
3. dist_to_LL20_norm：(C − LL20) / Close ，使用 StandardScaler归一化
4. body_ratio：实体长度 / (高低区间)  使用 StandardScaler归一化   
5. upper_shadow_ratio：上影线长度 / 总 K 线高度  使用 StandardScaler归一化   
6. lower_shadow_ratio：下影线长度 / 总 K 线高度  使用 StandardScaler归一化 
7. FVG公允价值缺口，使用 StandardScaler归一化  
#### 6. 时间周期特征（3 维）
支持根据不同的品种设置不同的交易时间，数据清洗步骤也只使用交易时间的数据
1. sin_tod：时间周期（一天中的角度 sin），不需要归一化
2. cos_tod：时间周期（一天中的角度 cos），不需要归一化
cos(θ) 表示时间位置（0° = 新的一天刚开始）  
sin(θ) 表示相位（125° vs 289°…）  
两个特征共同表示：  
一天中的时间位置  
每日的周期性节奏  
连续性与对称性  
### 4.2 Transformer 输出
状态向量 s_t（给 PPO 使用）（128-256维） 
s_t ∈ R^(D_state) 
## 5. 模型 3：PPO（强化学习执行层）
PPO 输入状态 → 输出交易动作（离散 + 连续）。
### 5.1 输入
s_t（来自 Transformer 的状态向量） 
交易持仓信息（position size, entry price） 
风险参数（最大杠杆、当前浮盈浮亏）  
组合成： 
state_for_RL = concat([s_t, position_info, risk_info]) 
### 5.2 输出（动作空间）
A. 开仓方向 — 离散动作 
0: 空仓   
1: 做多   
2: 做空   

B. 仓位大小 — 连续动作（0–1） 
代表风险敞口（根据账户余额折算）。 
C. 止损/止盈 — 连续动作 
stop_loss（如 0.1%–2%） 

take_profit（如 0.2%–5%） 
最终 PPO 输出： 
a_t = {
    direction, 
    position_size, 
    stop_loss, 
    take_profit 
} 


# 6. 奖励函数设计
建议分解 reward：
## 1. 盈利（主导项）
r_profit = realized_PnL / account_balance
## 2. 风险控制奖励
r_risk 
惩罚超过亏损阈值 
惩罚持仓过久无波动 
惩罚超过最大回撤 

## 3. 稳定性奖励
r_stability 
夏普率奖励
连续正确方向奖励 
连续错误方向惩罚 

总奖励： 
r_t = r_profit + λ1*r_risk + λ2*r_stability 


# 7. 系统超参数建议
| 模块          | 参数          | 取值        |
| ----------- | ----------- | --------- |
| TS2Vec      | embedding D | 128       |
| TS2Vec      | window      | 256       |
| Transformer | layers      | 4–6       |
| Transformer | heads       | 4–8       |
| Transformer | D_model     | 128–256   |
| PPO         | γ           | 0.95–0.99 |
| PPO         | clip        | 0.1–0.2   |
| PPO         | gae λ       | 0.95      |
| PPO         | policy lr   | 1e-4      |
| PPO         | value lr    | 3e-4      |

# 8. 维度校验
一：必须做的验证实验（按顺序执行）

这些实验既能验证“信息含量”，也能检测“过拟合风险”。

1) 时间序列分层验证：Walk-forward / Rolling CV（必做）

做法：

划分多个时间段（例如每 6 个月为一段），用滚动窗口做训练→验证→测试（例如训练 24 个月，验证 6 个月，测试 6 个月，向前滚动）。
要点：

禁止随机抽样交叉验证（会泄露时间信息）。

度量：每折记录主指标（Sharpe, CAGR, MaxDD, Profit factor）和监督学习指标（AUC/MSE/accuracy，视任务而定）。
合格条件：

验证期与测试期的主要指标降幅不超过 20%（相对训练）并且方向一致（不出现训练正、测试显著负的情况）。

2) 单特征信息量测试：单变量回归/分类与 Mutual Information

做法：

对每个手工特征单独训练一个简单模型（如线性回归/逻辑回归/小树）预测短期 label（比如下 5-bar return sign 或 vol bucket）。

计算 mutual information（或单变量 AUC/R²）。
合格条件：

特征的 MI 或单变量 R² 显著优于随机噪声 baseline（比如在置换下 p < 0.05）。

3) 置换重要性（Permutation Importance）与统计显著性（关键）

做法：

在训练好完整模型（TS2Vec+Transformer）后，对每个特征做 置换检验：把该特征在验证集上做多次随机打乱，计算验证指标下降的均值与分布。

重复 N=100 次以构建分布，得出 p-value。
合格条件：

若置换导致指标显著下降（比如平均 Sharpe 降 >10% 且 p<0.05），说明该特征贡献显著；反之若无显著变化，则该特征信息价值低。

4) Ablation（逐步剔除）与 Greedy Selection

做法：

先做全量基线（27 维），记录验证指标。

逐个或分组剔除特征并测评（backward elimination），或用前向贪心增加（forward selection）。
合格条件：

若剔除某特征后指标提升或持平，该特征应被考虑移除（说明带来噪声/过拟合）。

5) 特征重要性稳定性（Across folds / Seeds）

做法：

在多个 walk-forward 折与不同随机种子下训练模型，记录每个特征的重要性（如基于注意力权重、feature-permutation importance、或基于树模型的 feature importance）。
合格条件：

对于“有效特征”其 importance 的变异系数（std/mean）应低（建议 std/mean < 0.5）。若某个特征 importance 在不同折里完全不稳定，怀疑过拟合/偶然性。

6) 相关性与共线性检测（避免冗余）

做法：

计算手工特征间 Pearson/Spearman 相关矩阵与 VIF（方差膨胀因子）。
处理规则：

对高度相关对（|ρ|>0.85）考虑只保留更稳定/更易解释的那个

若某特征 VIF>10，说明共线性强，需降维或删除。

7) 经济显著性检验（Bootstrap / t-test on returns）

做法：

对策略在验证期间的净收益做 bootstrap计算收益均值的 95% CI。
合格条件：

若剔除某特征导致收益中位数或均值显著下降且 95% CI 不包含 0，则该特征经济上有意义。反之则可删。

8) 垂直应力测试（Stress Test / Regime Split）

做法：

在不同市场 regime（2008 类危机、2018 波动期、低波动期）分别评估。
合格条件：

有效特征应在多数 regime 中至少保持非负贡献；对仅在极端 regime 有效的特征需记录为“条件性使用”。

# 9.防止过拟合
A. 模型正则化

Weight decay（L2）：建议 1e-4 ~ 1e-3。

Dropout：Transformer 中用 0.05–0.2。
B. 多次跑不同 seed / 集成（Ensembling）

对每个配置至少跑 5–10 个随机种子，计算指标均值与方差。

若单次最好但平均差强人意，说明不稳，回退。

C. 特征归一化且仅基于历史（避免泄露）

所有 rolling stats 用历史窗口计算（no lookahead）。

保存 scaler，严格区分 train/val/test 的 scaler。
Layer norm / gradient clipping：max_grad_norm = 0.5。
# 10. 数据流程总结（强结构化）
[1] 原始 OHLC → 5min bar   
[2] 归一化 + 滑动窗口   
[3] TS2Vec → z_t   
[4] 手工特征 → f_t   
[5] Transformer 输入 concat(z_t, f_t) → s_t   
[6] PPO 输入 concat(s_t, position, risk)   
[7] PPO 输出动作   
[8] 执行交易，记录 reward   
[9] 再训练 PPO 


# 11. 输出接口（面向实盘/回测）
{  
  "direction": long/
  short/flat,  
  "position_size": float,  
  "stop_loss_price": float,  
  "take_profit_price": float,  
  "confidence": float  
} 
