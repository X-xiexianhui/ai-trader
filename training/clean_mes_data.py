"""
清洗MES期货数据
读取原始数据，进行数据清洗，保存到processed文件夹

注意：对于真实市场数据，我们只处理缺失值，不修正异常值
因为所谓的"异常"可能是真实的市场波动（闪崩、重大新闻等）
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.data_cleaner import DataCleaner

# 配置日志 - 保存到logs文件夹
log_dir = project_root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'mes_data_cleaning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数：清洗MES期货数据"""
    
    logger.info("=" * 80)
    logger.info("MES期货数据清洗")
    logger.info("=" * 80)
    logger.info("注意：对于真实市场数据，我们只处理缺失值，不修正异常值")
    logger.info("      因为所谓的'异常'可能是真实的市场波动")
    logger.info("=" * 80)
    
    try:
        # 1. 读取原始数据
        input_file = project_root / 'data' / 'raw' / 'mes_data' / 'MES_stitched_5m_ratio.csv'
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return
        
        logger.info(f"\n读取原始数据: {input_file}")
        df = pd.read_csv(input_file)
        
        # 转换date列为datetime并设置为索引
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        logger.info(f"原始数据: {len(df)} 行 × {len(df.columns)} 列")
        logger.info(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        logger.info(f"数据列: {list(df.columns)}")
        
        # 重命名列以匹配DataCleaner的要求（首字母大写）
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # 2. 创建数据清洗器
        logger.info("\n" + "=" * 80)
        logger.info("初始化数据清洗器")
        logger.info("=" * 80)
        
        cleaner = DataCleaner(
            max_consecutive_missing=5,  # 最多允许连续5根K线缺失
            interpolation_limit=3,       # 线性插值最多3个点
            sigma_threshold=3.0          # 3σ异常值检测（仅用于统计报告）
        )
        
        # 3. 执行数据清洗（只处理缺失值）
        logger.info("\n" + "=" * 80)
        logger.info("开始数据清洗流程")
        logger.info("=" * 80)
        
        # 步骤1: 缺失值处理
        logger.info("-" * 60)
        logger.info("步骤1/2: 缺失值处理")
        cleaned_df, missing_report = cleaner.handle_missing_values(df)
        
        # 步骤2: 质量验证
        logger.info("-" * 60)
        logger.info("步骤2/2: 数据质量验证")
        is_valid, validation_report = cleaner.validate_data_quality(cleaned_df)
        
        # 生成异常值统计（仅用于报告，不修正数据）
        logger.info("-" * 60)
        logger.info("生成异常值统计（仅报告，不修正数据）")
        close_prices = cleaned_df['Close'].replace(0, np.nan)
        log_returns = np.log(close_prices / close_prices.shift(1))
        valid_returns = log_returns.dropna()
        
        outlier_stats = {}
        if len(valid_returns) >= 10:
            mean_return = valid_returns.mean()
            std_return = valid_returns.std()
            
            if std_return > 0 and not np.isnan(std_return):
                # 3σ异常值统计
                threshold_3sigma = 3 * std_return
                outliers_3sigma = np.abs(log_returns - mean_return) > threshold_3sigma
                outlier_count_3sigma = int(outliers_3sigma.sum())
                
                # 5σ异常值统计
                threshold_5sigma = 5 * std_return
                outliers_5sigma = np.abs(log_returns - mean_return) > threshold_5sigma
                outlier_count_5sigma = int(outliers_5sigma.sum())
                
                outlier_stats = {
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'outliers_3sigma': {
                        'count': outlier_count_3sigma,
                        'ratio': outlier_count_3sigma / len(cleaned_df),
                        'threshold': float(threshold_3sigma)
                    },
                    'outliers_5sigma': {
                        'count': outlier_count_5sigma,
                        'ratio': outlier_count_5sigma / len(cleaned_df),
                        'threshold': float(threshold_5sigma)
                    },
                    'note': '这些统计仅供参考，数据未被修正'
                }
                
                logger.info(f"  检测到 {outlier_count_3sigma} 个>3σ的波动 ({outlier_count_3sigma/len(cleaned_df):.2%})")
                logger.info(f"  检测到 {outlier_count_5sigma} 个>5σ的波动 ({outlier_count_5sigma/len(cleaned_df):.2%})")
                logger.info(f"  注意：这些是真实的市场波动，未被修正")
        
        # 生成完整报告
        report = {
            'missing_values': missing_report,
            'outlier_statistics': outlier_stats,
            'validation': validation_report,
            'final_validation': is_valid,
            'note': '本次清洗只处理了缺失值，未修正任何异常值'
        }
        
        logger.info("=" * 60)
        logger.info(
            f"数据清洗完成: "
            f"最终{len(cleaned_df)}行数据, "
            f"验证{'通过✓' if is_valid else '失败✗'}"
        )
        logger.info("=" * 60)
        
        # 4. 保存清洗后的数据
        output_dir = project_root / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'MES_cleaned_5m.csv'
        cleaned_df.to_csv(output_file)
        logger.info(f"\n清洗后的数据已保存到: {output_file}")
        
        # 5. 保存清洗报告
        report_file = output_dir / 'MES_cleaning_report.json'
        
        # 转换报告中的特殊类型为可序列化格式
        def convert_to_serializable(obj):
            """递归转换对象为可序列化格式"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_report = convert_to_serializable(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"清洗报告已保存到: {report_file}")
        
        # 6. 生成并保存文本格式的报告摘要
        report_txt_file = output_dir / 'MES_cleaning_report.txt'
        
        with open(report_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MES期货数据清洗报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"输出文件: {output_file}\n")
            f.write(f"清洗策略: 只处理缺失值，不修正异常值\n\n")
            
            # 原始数据信息
            f.write("【原始数据】\n")
            f.write(f"  总行数: {missing_report.get('total_rows', 0):,}\n")
            f.write(f"  时间范围: {df.index.min()} 到 {df.index.max()}\n")
            f.write(f"  数据列: {', '.join(df.columns)}\n\n")
            
            # 缺失值处理
            f.write("【缺失值处理】\n")
            f.write(f"  处理前缺失值:\n")
            for col, count in missing_report.get('missing_before', {}).items():
                f.write(f"    {col}: {count}\n")
            f.write(f"  删除的数据段: {len(missing_report.get('removed_segments', []))} 个\n")
            f.write(f"  删除的行数: {missing_report.get('removed_rows', 0)}\n")
            f.write(f"  插值点数:\n")
            for col, count in missing_report.get('interpolated_points', {}).items():
                f.write(f"    {col}: {count}\n")
            f.write(f"  处理后缺失值:\n")
            for col, count in missing_report.get('missing_after', {}).items():
                f.write(f"    {col}: {count}\n")
            f.write(f"  处理耗时: {missing_report.get('processing_time', 0):.3f} 秒\n\n")
            
            # 异常值统计（未修正）
            if outlier_stats:
                f.write("【异常值统计】（仅统计，未修正）\n")
                f.write(f"  平均收益率: {outlier_stats['mean_return']:.6f}\n")
                f.write(f"  收益率标准差: {outlier_stats['std_return']:.6f}\n")
                f.write(f"\n  >3σ 波动:\n")
                f.write(f"    数量: {outlier_stats['outliers_3sigma']['count']}\n")
                f.write(f"    占比: {outlier_stats['outliers_3sigma']['ratio']:.2%}\n")
                f.write(f"    阈值: {outlier_stats['outliers_3sigma']['threshold']:.6f}\n")
                f.write(f"\n  >5σ 波动:\n")
                f.write(f"    数量: {outlier_stats['outliers_5sigma']['count']}\n")
                f.write(f"    占比: {outlier_stats['outliers_5sigma']['ratio']:.2%}\n")
                f.write(f"    阈值: {outlier_stats['outliers_5sigma']['threshold']:.6f}\n")
                f.write(f"\n  说明: 这些是真实的市场波动，未被修正\n\n")
            
            # 质量验证
            vd = validation_report
            f.write("【质量验证】\n")
            f.write(f"  验证结果: {'通过 ✓' if vd.get('is_valid', False) else '失败 ✗'}\n")
            f.write(f"  问题数量: {vd.get('summary', {}).get('total_issues', 0)}\n")
            f.write(f"  警告数量: {vd.get('summary', {}).get('total_warnings', 0)}\n")
            
            if vd.get('issues'):
                f.write(f"\n  问题列表:\n")
                for issue in vd['issues']:
                    f.write(f"    - {issue}\n")
            
            if vd.get('warnings'):
                f.write(f"\n  警告列表:\n")
                for warning in vd['warnings']:
                    f.write(f"    - {warning}\n")
            
            # 完整性检查
            if 'completeness' in vd:
                f.write(f"\n  完整性检查:\n")
                for col, info in vd['completeness'].items():
                    f.write(f"    {col}: 缺失 {info['missing_count']} ({info['missing_ratio']:.2%})\n")
            
            # 一致性检查
            if 'consistency' in vd:
                cons = vd['consistency']
                f.write(f"\n  一致性检查:\n")
                high_result = '通过' if cons.get('high_valid') else f"失败 ({cons.get('high_violations', 0)}处)"
                low_result = '通过' if cons.get('low_valid') else f"失败 ({cons.get('low_violations', 0)}处)"
                hl_result = '通过' if cons.get('high_low_valid') else f"失败 ({cons.get('hl_violations', 0)}处)"
                f.write(f"    High >= max(O,C): {high_result}\n")
                f.write(f"    Low <= min(O,C): {low_result}\n")
                f.write(f"    High >= Low: {hl_result}\n")
            
            # 时间质量
            if 'time_quality' in vd:
                tq = vd['time_quality']
                f.write(f"\n  时间质量:\n")
                if 'duplicates' in tq:
                    f.write(f"    重复时间戳: {tq['duplicates']}\n")
                if 'is_sorted' in tq:
                    f.write(f"    时间排序: {'正确' if tq['is_sorted'] else '错误'}\n")
                if 'median_interval' in tq:
                    f.write(f"    中位时间间隔: {tq['median_interval']}\n")
            
            f.write(f"\n  验证耗时: {vd.get('processing_time', 0):.3f} 秒\n\n")
            
            # 清洗后数据信息
            f.write("【清洗后数据】\n")
            f.write(f"  总行数: {len(cleaned_df):,}\n")
            f.write(f"  时间范围: {cleaned_df.index.min()} 到 {cleaned_df.index.max()}\n")
            f.write(f"  数据列: {', '.join(cleaned_df.columns)}\n\n")
            
            # 数据统计
            f.write("【数据统计】\n")
            stats = cleaned_df[['Open', 'High', 'Low', 'Close']].describe()
            f.write(stats.to_string())
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"文本报告已保存到: {report_txt_file}")
        
        # 7. 打印摘要
        logger.info("\n" + "=" * 80)
        logger.info("数据清洗摘要")
        logger.info("=" * 80)
        logger.info(f"原始数据: {missing_report.get('total_rows', 0):,} 行")
        logger.info(f"删除数据段: {len(missing_report.get('removed_segments', []))} 个")
        logger.info(f"删除行数: {missing_report.get('removed_rows', 0)}")
        logger.info(f"清洗后数据: {len(cleaned_df):,} 行")
        logger.info(f"验证结果: {'通过 ✓' if is_valid else '失败 ✗'}")
        if outlier_stats:
            logger.info(f">3σ 波动: {outlier_stats['outliers_3sigma']['count']} 个（未修正）")
            logger.info(f">5σ 波动: {outlier_stats['outliers_5sigma']['count']} 个（未修正）")
        
        logger.info("=" * 80)
        logger.info("数据清洗完成！")
        logger.info("=" * 80)
        logger.info(f"清洗后数据: {output_file}")
        logger.info(f"JSON报告: {report_file}")
        logger.info(f"文本报告: {report_txt_file}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\n用户中断清洗")
    except Exception as e:
        logger.error(f"\n清洗过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\n程序结束")


if __name__ == '__main__':
    main()