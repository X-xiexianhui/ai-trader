"""
回测报告生成器

生成详细的回测报告，包括文本、HTML和PDF格式
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """回测报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"报告生成器初始化，输出目录: {self.output_dir}")
    
    def generate_report(self,
                       results: Dict,
                       trades: List[Dict],
                       equity_curve: pd.Series,
                       format: str = "all") -> Dict[str, str]:
        """
        生成回测报告
        
        Args:
            results: 回测结果
            trades: 交易记录
            equity_curve: 权益曲线
            format: 报告格式 ('text', 'html', 'json', 'all')
            
        Returns:
            Dict[str, str]: 生成的报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # 生成文本报告
        if format in ['text', 'all']:
            text_file = self.output_dir / f"report_{timestamp}.txt"
            self._generate_text_report(results, trades, equity_curve, text_file)
            report_files['text'] = str(text_file)
            logger.info(f"文本报告已生成: {text_file}")
        
        # 生成HTML报告
        if format in ['html', 'all']:
            html_file = self.output_dir / f"report_{timestamp}.html"
            self._generate_html_report(results, trades, equity_curve, html_file)
            report_files['html'] = str(html_file)
            logger.info(f"HTML报告已生成: {html_file}")
        
        # 生成JSON报告
        if format in ['json', 'all']:
            json_file = self.output_dir / f"report_{timestamp}.json"
            self._generate_json_report(results, trades, equity_curve, json_file)
            report_files['json'] = str(json_file)
            logger.info(f"JSON报告已生成: {json_file}")
        
        return report_files
    
    def _generate_text_report(self,
                             results: Dict,
                             trades: List[Dict],
                             equity_curve: pd.Series,
                             output_file: Path):
        """生成文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("回测报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            f.write("【基本信息】\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"回测开始时间: {equity_curve.index[0]}\n")
            f.write(f"回测结束时间: {equity_curve.index[-1]}\n")
            f.write(f"回测天数: {(equity_curve.index[-1] - equity_curve.index[0]).days}\n")
            f.write("\n")
            
            # 收益指标
            f.write("【收益指标】\n")
            f.write(f"初始资金: {results.get('initial_capital', 0):,.2f}\n")
            f.write(f"最终资金: {results.get('final_capital', 0):,.2f}\n")
            f.write(f"总收益: {results.get('total_return', 0):,.2f}\n")
            f.write(f"总收益率: {results.get('total_return_pct', 0):.2%}\n")
            f.write(f"年化收益率: {results.get('annual_return', 0):.2%}\n")
            f.write(f"CAGR: {results.get('cagr', 0):.2%}\n")
            f.write("\n")
            
            # 风险指标
            f.write("【风险指标】\n")
            f.write(f"最大回撤: {results.get('max_drawdown', 0):.2%}\n")
            f.write(f"最大回撤持续期: {results.get('max_drawdown_duration', 0)} 天\n")
            f.write(f"波动率: {results.get('volatility', 0):.2%}\n")
            f.write(f"下行波动率: {results.get('downside_volatility', 0):.2%}\n")
            f.write(f"VaR (95%): {results.get('var_95', 0):.2%}\n")
            f.write(f"CVaR (95%): {results.get('cvar_95', 0):.2%}\n")
            f.write("\n")
            
            # 风险调整收益
            f.write("【风险调整收益】\n")
            f.write(f"夏普比率: {results.get('sharpe_ratio', 0):.4f}\n")
            f.write(f"索提诺比率: {results.get('sortino_ratio', 0):.4f}\n")
            f.write(f"卡玛比率: {results.get('calmar_ratio', 0):.4f}\n")
            f.write(f"信息比率: {results.get('information_ratio', 0):.4f}\n")
            f.write("\n")
            
            # 交易统计
            f.write("【交易统计】\n")
            f.write(f"总交易次数: {results.get('total_trades', 0)}\n")
            f.write(f"盈利交易: {results.get('winning_trades', 0)}\n")
            f.write(f"亏损交易: {results.get('losing_trades', 0)}\n")
            f.write(f"胜率: {results.get('win_rate', 0):.2%}\n")
            f.write(f"盈亏比: {results.get('profit_factor', 0):.2f}\n")
            f.write(f"平均盈利: {results.get('avg_win', 0):,.2f}\n")
            f.write(f"平均亏损: {results.get('avg_loss', 0):,.2f}\n")
            f.write(f"最大盈利: {results.get('max_win', 0):,.2f}\n")
            f.write(f"最大亏损: {results.get('max_loss', 0):,.2f}\n")
            f.write(f"平均持仓时间: {results.get('avg_holding_period', 0):.1f} 小时\n")
            f.write("\n")
            
            # 连续交易统计
            f.write("【连续交易统计】\n")
            f.write(f"最大连续盈利: {results.get('max_consecutive_wins', 0)}\n")
            f.write(f"最大连续亏损: {results.get('max_consecutive_losses', 0)}\n")
            f.write("\n")
            
            # 交易明细
            if trades:
                f.write("【交易明细】\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'时间':<20} {'操作':<8} {'价格':<12} {'数量':<12} {'盈亏':<12}\n")
                f.write("-" * 80 + "\n")
                
                for trade in trades[-20:]:  # 只显示最近20笔交易
                    timestamp = trade.get('timestamp', '')
                    action = trade.get('action', '')
                    price = trade.get('price', 0)
                    size = trade.get('size', 0)
                    pnl = trade.get('pnl', 0)
                    
                    f.write(f"{str(timestamp):<20} {action:<8} {price:<12.2f} "
                           f"{size:<12.2f} {pnl:<12.2f}\n")
                
                f.write("-" * 80 + "\n")
                f.write(f"（仅显示最近20笔交易，共{len(trades)}笔）\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
    
    def _generate_html_report(self,
                             results: Dict,
                             trades: List[Dict],
                             equity_curve: pd.Series,
                             output_file: Path):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.negative {{
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .info-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #888;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 回测报告</h1>
        
        <div class="info-box">
            <p><strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>回测周期:</strong> {equity_curve.index[0]} 至 {equity_curve.index[-1]}</p>
            <p><strong>回测天数:</strong> {(equity_curve.index[-1] - equity_curve.index[0]).days} 天</p>
        </div>
        
        <h2>💰 收益指标</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if results.get('total_return_pct', 0) > 0 else 'negative'}">
                <div class="metric-label">总收益率</div>
                <div class="metric-value">{results.get('total_return_pct', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value">{results.get('annual_return', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{results.get('cagr', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最终资金</div>
                <div class="metric-value">¥{results.get('final_capital', 0):,.0f}</div>
            </div>
        </div>
        
        <h2>⚠️ 风险指标</h2>
        <div class="metrics-grid">
            <div class="metric-card negative">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value">{results.get('max_drawdown', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">波动率</div>
                <div class="metric-value">{results.get('volatility', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">VaR (95%)</div>
                <div class="metric-value">{results.get('var_95', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CVaR (95%)</div>
                <div class="metric-value">{results.get('cvar_95', 0):.2%}</div>
            </div>
        </div>
        
        <h2>📈 风险调整收益</h2>
        <div class="metrics-grid">
            <div class="metric-card positive">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{results.get('sharpe_ratio', 0):.4f}</div>
            </div>
            <div class="metric-card positive">
                <div class="metric-label">索提诺比率</div>
                <div class="metric-value">{results.get('sortino_ratio', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">卡玛比率</div>
                <div class="metric-value">{results.get('calmar_ratio', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">信息比率</div>
                <div class="metric-value">{results.get('information_ratio', 0):.4f}</div>
            </div>
        </div>
        
        <h2>📊 交易统计</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>总交易次数</td>
                <td>{results.get('total_trades', 0)}</td>
            </tr>
            <tr>
                <td>盈利交易</td>
                <td style="color: green;">{results.get('winning_trades', 0)}</td>
            </tr>
            <tr>
                <td>亏损交易</td>
                <td style="color: red;">{results.get('losing_trades', 0)}</td>
            </tr>
            <tr>
                <td>胜率</td>
                <td>{results.get('win_rate', 0):.2%}</td>
            </tr>
            <tr>
                <td>盈亏比</td>
                <td>{results.get('profit_factor', 0):.2f}</td>
            </tr>
            <tr>
                <td>平均盈利</td>
                <td style="color: green;">¥{results.get('avg_win', 0):,.2f}</td>
            </tr>
            <tr>
                <td>平均亏损</td>
                <td style="color: red;">¥{results.get('avg_loss', 0):,.2f}</td>
            </tr>
            <tr>
                <td>最大盈利</td>
                <td style="color: green;">¥{results.get('max_win', 0):,.2f}</td>
            </tr>
            <tr>
                <td>最大亏损</td>
                <td style="color: red;">¥{results.get('max_loss', 0):,.2f}</td>
            </tr>
        </table>
        
        <p class="timestamp">报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_json_report(self,
                             results: Dict,
                             trades: List[Dict],
                             equity_curve: pd.Series,
                             output_file: Path):
        """生成JSON报告"""
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'start_date': str(equity_curve.index[0]),
                'end_date': str(equity_curve.index[-1]),
                'duration_days': (equity_curve.index[-1] - equity_curve.index[0]).days
            },
            'results': results,
            'trades': trades,
            'equity_curve': {
                'dates': [str(d) for d in equity_curve.index],
                'values': equity_curve.tolist()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    def generate_summary(self, results: Dict) -> str:
        """
        生成简要摘要
        
        Args:
            results: 回测结果
            
        Returns:
            str: 摘要文本
        """
        summary = f"""
回测摘要
========
总收益率: {results.get('total_return_pct', 0):.2%}
年化收益率: {results.get('annual_return', 0):.2%}
最大回撤: {results.get('max_drawdown', 0):.2%}
夏普比率: {results.get('sharpe_ratio', 0):.4f}
胜率: {results.get('win_rate', 0):.2%}
总交易次数: {results.get('total_trades', 0)}
"""
        return summary


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    equity_curve = pd.Series(
        np.random.randn(100).cumsum() + 100000,
        index=dates
    )
    
    results = {
        'initial_capital': 100000,
        'final_capital': 110000,
        'total_return': 10000,
        'total_return_pct': 0.10,
        'annual_return': 0.40,
        'cagr': 0.38,
        'max_drawdown': -0.15,
        'max_drawdown_duration': 10,
        'volatility': 0.20,
        'downside_volatility': 0.15,
        'var_95': -0.03,
        'cvar_95': -0.05,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'calmar_ratio': 2.5,
        'information_ratio': 1.2,
        'total_trades': 50,
        'winning_trades': 30,
        'losing_trades': 20,
        'win_rate': 0.60,
        'profit_factor': 1.8,
        'avg_win': 500,
        'avg_loss': -300,
        'max_win': 2000,
        'max_loss': -1000,
        'avg_holding_period': 24,
        'max_consecutive_wins': 5,
        'max_consecutive_losses': 3
    }
    
    trades = [
        {
            'timestamp': dates[i],
            'action': 'buy' if i % 2 == 0 else 'sell',
            'price': 50000 + i * 100,
            'size': 0.1,
            'pnl': np.random.randn() * 100
        }
        for i in range(20)
    ]
    
    # 生成报告
    generator = ReportGenerator()
    report_files = generator.generate_report(results, trades, equity_curve)
    
    print("\n生成的报告文件:")
    for format_type, file_path in report_files.items():
        print(f"  {format_type}: {file_path}")
    
    # 生成摘要
    summary = generator.generate_summary(results)
    print(summary)