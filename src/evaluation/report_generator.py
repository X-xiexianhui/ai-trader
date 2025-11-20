"""
评估报告生成器

生成综合评估报告（HTML/PDF/Markdown格式）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """
    评估报告生成器
    
    整合所有评估结果，生成综合报告
    """
    
    def __init__(self, project_name: str = "AI Trading System"):
        """
        初始化报告生成器
        
        Args:
            project_name: 项目名称
        """
        self.project_name = project_name
        self.sections = {}
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'project_name': project_name
        }
        
        logger.info(f"评估报告生成器初始化: {project_name}")
    
    def add_section(self, section_name: str, content: Dict):
        """
        添加报告章节
        
        Args:
            section_name: 章节名称
            content: 章节内容
        """
        self.sections[section_name] = content
        logger.info(f"添加章节: {section_name}")
    
    def generate_markdown(self, output_path: str):
        """
        生成Markdown格式报告
        
        Args:
            output_path: 输出文件路径
        """
        md_content = []
        
        # 标题
        md_content.append(f"# {self.project_name} - 评估报告\n")
        md_content.append(f"生成时间: {self.metadata['generated_at']}\n")
        md_content.append("---\n")
        
        # 目录
        md_content.append("## 目录\n")
        for i, section_name in enumerate(self.sections.keys(), 1):
            md_content.append(f"{i}. [{section_name}](#{section_name.lower().replace(' ', '-')})\n")
        md_content.append("\n---\n")
        
        # 各章节内容
        for section_name, content in self.sections.items():
            md_content.append(f"\n## {section_name}\n")
            md_content.append(self._format_section_markdown(content))
            md_content.append("\n---\n")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        logger.info(f"Markdown报告已生成: {output_path}")
    
    def _format_section_markdown(self, content: Dict) -> str:
        """格式化章节内容为Markdown"""
        lines = []
        
        for key, value in content.items():
            if isinstance(value, dict):
                lines.append(f"\n### {key}\n")
                lines.append(self._format_dict_markdown(value))
            elif isinstance(value, pd.DataFrame):
                lines.append(f"\n### {key}\n")
                lines.append(value.to_markdown())
                lines.append("\n")
            elif isinstance(value, (list, tuple)):
                lines.append(f"\n### {key}\n")
                for item in value:
                    lines.append(f"- {item}\n")
            else:
                lines.append(f"**{key}**: {value}\n")
        
        return '\n'.join(lines)
    
    def _format_dict_markdown(self, d: Dict, indent: int = 0) -> str:
        """递归格式化字典为Markdown"""
        lines = []
        prefix = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:\n")
                lines.append(self._format_dict_markdown(value, indent + 1))
            elif isinstance(value, (list, tuple)):
                lines.append(f"{prefix}- **{key}**:\n")
                for item in value:
                    lines.append(f"{prefix}  - {item}\n")
            else:
                lines.append(f"{prefix}- **{key}**: {value}\n")
        
        return '\n'.join(lines)
    
    def generate_html(self, output_path: str):
        """
        生成HTML格式报告
        
        Args:
            output_path: 输出文件路径
        """
        html_content = []
        
        # HTML头部
        html_content.append("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - 评估报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        h3 {{ color: #764ba2; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{
            display: inline-block;
            background: #e8eaf6;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #667eea;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{}</h1>
        <p class="timestamp">生成时间: {}</p>
    </div>
""".format(self.project_name, self.project_name, self.metadata['generated_at']))
        
        # 各章节内容
        for section_name, content in self.sections.items():
            html_content.append(f'<div class="section">')
            html_content.append(f'<h2>{section_name}</h2>')
            html_content.append(self._format_section_html(content))
            html_content.append('</div>')
        
        # HTML尾部
        html_content.append("""
</body>
</html>
""")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"HTML报告已生成: {output_path}")
    
    def _format_section_html(self, content: Dict) -> str:
        """格式化章节内容为HTML"""
        lines = []
        
        for key, value in content.items():
            if isinstance(value, dict):
                lines.append(f'<h3>{key}</h3>')
                lines.append(self._format_dict_html(value))
            elif isinstance(value, pd.DataFrame):
                lines.append(f'<h3>{key}</h3>')
                lines.append(value.to_html(index=False, classes='data-table'))
            elif isinstance(value, (list, tuple)):
                lines.append(f'<h3>{key}</h3>')
                lines.append('<ul>')
                for item in value:
                    lines.append(f'<li>{item}</li>')
                lines.append('</ul>')
            else:
                lines.append(f'<div class="metric">')
                lines.append(f'<span class="metric-label">{key}:</span> {value}')
                lines.append('</div>')
        
        return '\n'.join(lines)
    
    def _format_dict_html(self, d: Dict) -> str:
        """格式化字典为HTML"""
        lines = ['<ul>']
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f'<li><strong>{key}</strong>:')
                lines.append(self._format_dict_html(value))
                lines.append('</li>')
            elif isinstance(value, (list, tuple)):
                lines.append(f'<li><strong>{key}</strong>:')
                lines.append('<ul>')
                for item in value:
                    lines.append(f'<li>{item}</li>')
                lines.append('</ul>')
                lines.append('</li>')
            else:
                lines.append(f'<li><strong>{key}</strong>: {value}</li>')
        
        lines.append('</ul>')
        return '\n'.join(lines)
    
    def generate_json(self, output_path: str):
        """
        生成JSON格式报告
        
        Args:
            output_path: 输出文件路径
        """
        report_data = {
            'metadata': self.metadata,
            'sections': self._convert_to_serializable(self.sections)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"JSON报告已生成: {output_path}")
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """转换对象为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_all(self, output_dir: str, base_name: str = "evaluation_report"):
        """
        生成所有格式的报告
        
        Args:
            output_dir: 输出目录
            base_name: 基础文件名
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成各种格式
        self.generate_markdown(str(output_path / f"{base_name}.md"))
        self.generate_html(str(output_path / f"{base_name}.html"))
        self.generate_json(str(output_path / f"{base_name}.json"))
        
        logger.info(f"所有格式报告已生成到: {output_dir}")


def create_comprehensive_report(
    walk_forward_results: Optional[Dict] = None,
    feature_importance: Optional[Dict] = None,
    ablation_results: Optional[Dict] = None,
    overfitting_detection: Optional[Dict] = None,
    stability_test: Optional[Dict] = None,
    regime_analysis: Optional[Dict] = None,
    stress_test: Optional[Dict] = None,
    benchmark_comparison: Optional[Dict] = None,
    output_dir: str = "results/evaluation"
) -> EvaluationReportGenerator:
    """
    创建综合评估报告
    
    Args:
        walk_forward_results: Walk-Forward验证结果
        feature_importance: 特征重要性分析结果
        ablation_results: 消融实验结果
        overfitting_detection: 过拟合检测结果
        stability_test: 稳定性测试结果
        regime_analysis: 市场状态分析结果
        stress_test: 压力测试结果
        benchmark_comparison: 基准对比结果
        output_dir: 输出目录
        
    Returns:
        EvaluationReportGenerator: 报告生成器实例
    """
    generator = EvaluationReportGenerator()
    
    # 添加各个章节
    if walk_forward_results:
        generator.add_section("Walk-Forward验证", walk_forward_results)
    
    if feature_importance:
        generator.add_section("特征重要性分析", feature_importance)
    
    if ablation_results:
        generator.add_section("消融实验", ablation_results)
    
    if overfitting_detection:
        generator.add_section("过拟合检测", overfitting_detection)
    
    if stability_test:
        generator.add_section("稳定性测试", stability_test)
    
    if regime_analysis:
        generator.add_section("市场状态分析", regime_analysis)
    
    if stress_test:
        generator.add_section("压力测试", stress_test)
    
    if benchmark_comparison:
        generator.add_section("基准策略对比", benchmark_comparison)
    
    # 生成所有格式报告
    generator.generate_all(output_dir)
    
    return generator


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_results = {
        'summary': {
            'total_tests': 10,
            'passed': 8,
            'failed': 2
        },
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88
        },
        'details': [
            'Test 1: Passed',
            'Test 2: Passed',
            'Test 3: Failed'
        ]
    }
    
    # 创建报告
    generator = EvaluationReportGenerator("Test Project")
    generator.add_section("测试结果", test_results)
    
    # 生成报告
    generator.generate_all("test_output", "test_report")
    
    print("测试报告生成完成")