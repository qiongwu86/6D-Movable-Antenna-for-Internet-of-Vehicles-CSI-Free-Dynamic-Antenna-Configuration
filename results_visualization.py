

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置高级样式（兼容性修复）
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        print("⚠️  使用默认样式（seaborn不可用）")

try:
    sns.set_palette("husl")
except:
    print("⚠️  seaborn调色板不可用，使用matplotlib默认配色")

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 强制显示边框线
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.linewidth'] = 1.0

# 重置可能影响边框的参数
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

# 高级配色方案（6种方法）
COLORS = {
    'fpa': '#2E86AB',        # 深蓝色 - 传统FPA
    'random': '#A23B72',     # 深紫红 - 随机方法
    'optimized': '#F18F01',  # 橙色 - 优化方法
    'predictive': '#2ECC71', # 绿色 - 预测性部署
    'rotation': '#592E83',   # 深紫色 - 离散旋转
    'circular': '#C73E1D'    # 深红色 - 圆形位置
}

# 图案填充（用于灰白打印区分）
HATCH_PATTERNS = {
    'fpa': '///',             # 斜线 - 传统FPA
    'random': '|||',          # 竖线 - 随机方法  
    'optimized': '+++',       # 十字 - 优化方法
    'predictive': '***',      # 星号 - 预测性部署
    'rotation': 'xxx',        # 叉号 - 离散旋转
    'circular': '...'         # 点 - 圆形位置
}

# 按照指定顺序排列的方法名称（6种方法）
METHOD_ORDER = ['fpa', 'circular', 'rotation', 'predictive', 'optimized']

METHOD_NAMES = {
    'fpa': '传统FPA',
    'random': '随机天线',
    'optimized': '优化天线',
    'predictive': '预测性部署',
    'rotation': '离散旋转6DMA',
    'circular': '圆形位置6DMA'
}

METHOD_LABELS_EN = {
    'fpa': 'Traditional FPA',
    'random': 'Random 6DMA Antenna',
    'optimized': 'Proposed, N=1',
    'predictive': 'Proposed, N=10',
    'rotation': 'Discrete Rotation 6DMA',
    'circular': 'Circular Position 6DMA'
}


def load_test_results():
    """加载所有测试结果数据（6种方法）"""
    results = {}
    
    # 1. 加载demo_dynamic_scenario _up - 副本的用户数量测试结果（4种方法）
    try:
        with open('predictive_deployment_results/user_count_test_results.json', 'r', encoding='utf-8') as f:
            predictive_user_data = json.load(f)
            print("✅ 加载预测性部署用户数量测试数据（4种方法）")
    except FileNotFoundError:
        print("⚠️  未找到预测性部署用户数量测试数据")
        predictive_user_data = {}
    
    # 2. 加载demo_dynamic_scenario _up - 副本的功率测试结果（4种方法）
    try:
        with open('predictive_deployment_results/power_test_results.json', 'r', encoding='utf-8') as f:
            predictive_power_data = json.load(f)
            print("✅ 加载预测性部署功率测试数据（4种方法）")
    except FileNotFoundError:
        print("⚠️  未找到预测性部署功率测试数据")
        predictive_power_data = {}
    
    # 3. 加载adaptive_6dma_methods的用户数量测试结果（2种6DMA变体）
    try:
        with open('adaptive_6dma_user_test_results/user_count_test_results.json', 'r', encoding='utf-8') as f:
            adaptive_user_data = json.load(f)
            print("✅ 加载adaptive_6dma用户数量测试数据（2种6DMA变体）")
    except FileNotFoundError:
        print("⚠️  未找到adaptive_6dma用户数量测试数据")
        adaptive_user_data = {}
    
    # 4. 加载adaptive_6dma_methods的功率测试结果（2种6DMA变体）
    try:
        with open('adaptive_6dma_power_test_results/power_test_results.json', 'r', encoding='utf-8') as f:
            adaptive_power_data = json.load(f)
            print("✅ 加载adaptive_6dma功率测试数据（2种6DMA变体）")
    except FileNotFoundError:
        print("⚠️  未找到adaptive_6dma功率测试数据")
        adaptive_power_data = {}
    
    return {
        'predictive_user_data': predictive_user_data,
        'predictive_power_data': predictive_power_data,
        'adaptive_user_data': adaptive_user_data,
        'adaptive_power_data': adaptive_power_data
    }


def extract_user_count_data(all_results):
    """提取用户数量测试数据（6种方法）"""
    user_counts = []
    method_data = {
        'fpa': [],
        'random': [],
        'optimized': [],
        'predictive': [],
        'rotation': [],
        'circular': []
    }
    
    # 从预测性部署数据提取（4种方法）
    predictive_user_data = all_results['predictive_user_data']
    adaptive_user_data = all_results['adaptive_user_data']
    
    # 获取用户数量列表
    if predictive_user_data:
        user_counts = sorted([int(k) for k in predictive_user_data.keys()])
    elif adaptive_user_data:
        user_counts = sorted([int(k) for k in adaptive_user_data.keys()])
    
    for user_count in user_counts:
        user_count_str = str(user_count)
        
        # 从预测性部署数据获取4种方法数据
        if user_count_str in predictive_user_data:
            data = predictive_user_data[user_count_str]
            method_data['fpa'].append(data.get('fpa', {}).get('avg_rate', 0))
            method_data['random'].append(data.get('random', {}).get('avg_rate', 0))
            method_data['optimized'].append(data.get('optimized', {}).get('avg_rate', 0))
            
            # 预测性部署数据
            if data.get('predictive') and data['predictive'] is not None:
                method_data['predictive'].append(data['predictive'].get('avg_rate', 0))
            else:
                method_data['predictive'].append(0)
        else:
            method_data['fpa'].append(0)
            method_data['random'].append(0)
            method_data['optimized'].append(0)
            method_data['predictive'].append(0)
        
        # 从adaptive_6dma获取2种6DMA变体数据
        if user_count_str in adaptive_user_data:
            data = adaptive_user_data[user_count_str]
            method_data['circular'].append(data.get('circular_avg_rate', 0))
            method_data['rotation'].append(data.get('rotation_avg_rate', 0))
        else:
            method_data['circular'].append(0)
            method_data['rotation'].append(0)
    
    return user_counts, method_data


def extract_power_data(all_results):
    """提取功率测试数据（6种方法）"""
    power_values = []
    method_data = {
        'fpa': [],
        'random': [],
        'optimized': [],
        'predictive': [],
        'rotation': [],
        'circular': []
    }
    
    # 从预测性部署数据提取（4种方法）
    predictive_power_data = all_results['predictive_power_data']
    adaptive_power_data = all_results['adaptive_power_data']
    
    # 获取功率值列表
    if predictive_power_data:
        power_values = sorted([int(k) for k in predictive_power_data.keys()])
    elif adaptive_power_data:
        power_values = sorted([int(k) for k in adaptive_power_data.keys()])
    
    for power_mw in power_values:
        power_str = str(power_mw)
        
        # 从预测性部署数据获取4种方法数据
        if power_str in predictive_power_data:
            data = predictive_power_data[power_str]
            method_data['fpa'].append(data.get('fpa', {}).get('avg_rate', 0))
            method_data['random'].append(data.get('random', {}).get('avg_rate', 0))
            method_data['optimized'].append(data.get('optimized', {}).get('avg_rate', 0))
            
            # 预测性部署数据
            if data.get('predictive') and data['predictive'] is not None:
                method_data['predictive'].append(data['predictive'].get('avg_rate', 0))
            else:
                method_data['predictive'].append(0)
        else:
            method_data['fpa'].append(0)
            method_data['random'].append(0)
            method_data['optimized'].append(0)
            method_data['predictive'].append(0)
        
        # 从adaptive_6dma获取2种6DMA变体数据
        if power_str in adaptive_power_data:
            data = adaptive_power_data[power_str]
            method_data['circular'].append(data.get('circular_avg_rate', 0))
            method_data['rotation'].append(data.get('rotation_avg_rate', 0))
        else:
            method_data['circular'].append(0)
            method_data['rotation'].append(0)
    
    return power_values, method_data


def create_user_count_comparison_chart(user_counts, method_data, output_dir):
    """创建用户数量对比柱状图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 设置柱状图参数（6种方法）
    x = np.arange(len(user_counts))
    width = 0.13  # 调整柱子宽度适应6种方法
    
    # 创建6个柱状图组
    bars = []
    for i, method_key in enumerate(METHOD_ORDER):
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        offset = (i - 2.5) * width  # 居中对齐（6种方法）
        bars.append(ax.bar(x + offset, method_data[method_key], width, 
                          label=method_name, color=COLORS[method_key], 
                          alpha=0.8, edgecolor='white', linewidth=0.5,
                          hatch=HATCH_PATTERNS[method_key]))
    
    # 设置图表属性（英文标题）
    ax.set_xlabel('Number of Ground Vehicles', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total User Rate (Mbps)', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{count}' for count in user_counts], fontsize=18)
    
    # 设置网格和样式
    # 设置清晰的方格线
    ax.grid(True, alpha=0.7, axis='both', linestyle='-', linewidth=0.8, color='gray')
    # 添加次要网格线
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.4, linestyle=':', linewidth=0.5, color='lightgray')
    ax.set_axisbelow(True)
    
    # 设置图例（放大字号）
    legend = ax.legend(loc='upper left', fontsize=18, framealpha=0.9, 
                      fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # 取消数值标签（按要求移除）
    
    # 设置y轴范围
    max_rate = max([max(rates) for rates in method_data.values() if rates])
    ax.set_ylim(0, max_rate * 1.15)
    
    # 显示所有边框线
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # 最终强制设置边框（确保不被覆盖）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # 强制刷新图表
    fig.canvas.draw()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/user_count_comparison.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/user_count_comparison.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 用户数量对比图已保存至: {output_dir}/user_count_comparison.png")
    
    return fig


def create_power_comparison_chart(power_values, method_data, output_dir):
    """创建功率对比柱状图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 设置柱状图参数（6种方法）
    x = np.arange(len(power_values))
    width = 0.13  # 调整柱子宽度适应6种方法
    
    # 创建6个柱状图组
    bars = []
    for i, method_key in enumerate(METHOD_ORDER):
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        offset = (i - 2.5) * width  # 居中对齐（6种方法）
        bars.append(ax.bar(x + offset, method_data[method_key], width, 
                          label=method_name, color=COLORS[method_key], 
                          alpha=0.8, edgecolor='white', linewidth=0.5,
                          hatch=HATCH_PATTERNS[method_key]))
    
    # 设置图表属性（英文标题）
    ax.set_xlabel('Transmit Power (mW)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total User Rate (Mbps)', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{power}mW' for power in power_values], fontsize=18)
    
    # 设置网格和样式
    # 设置清晰的方格线
    ax.grid(True, alpha=0.7, axis='both', linestyle='-', linewidth=0.8, color='gray')
    # 添加次要网格线
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.4, linestyle=':', linewidth=0.5, color='lightgray')
    ax.set_axisbelow(True)
    
    # 设置图例（放大字号）
    legend = ax.legend(loc='upper left', fontsize=18, framealpha=0.9, 
                      fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # 取消数值标签（按要求移除）
    
    # 设置y轴范围
    max_rate = max([max(rates) for rates in method_data.values() if rates])
    ax.set_ylim(0, max_rate * 1.15)
    
    # 显示所有边框线
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # 最终强制设置边框（确保不被覆盖）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # 强制刷新图表
    fig.canvas.draw()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/power_comparison.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/power_comparison.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 功率对比图已保存至: {output_dir}/power_comparison.png")
    
    return fig


def create_comprehensive_comparison_chart(user_counts, user_method_data, 
                                        power_values, power_method_data, output_dir):
    """创建综合对比图（2x1布局）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # === 上图：用户数量对比 ===
    x1 = np.arange(len(user_counts))
    width = 0.13  # 适应6种方法
    
    bars1 = []
    for i, method_key in enumerate(METHOD_ORDER):
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        offset = (i - 2.5) * width  # 居中对齐（6种方法）
        bars1.append(ax1.bar(x1 + offset, user_method_data[method_key], width, 
                            label=method_name, color=COLORS[method_key], 
                            alpha=0.8, edgecolor='white', linewidth=0.5,
                            hatch=HATCH_PATTERNS[method_key]))
    
    ax1.set_xlabel('Number of Ground Vehicles', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average User Rate (Mbps)', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Performance vs User Count', fontsize=18, fontweight='bold', pad=15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f'{count}V+5U' for count in user_counts], fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # 添加数值标签
    for bar_group in bars1:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12,
                           fontweight='bold')
    
    # === 下图：功率对比 ===
    x2 = np.arange(len(power_values))
    
    bars2 = []
    for i, method_key in enumerate(METHOD_ORDER):
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        offset = (i - 2.5) * width  # 居中对齐（6种方法）
        bars2.append(ax2.bar(x2 + offset, power_method_data[method_key], width, 
                            label=method_name, color=COLORS[method_key], 
                            alpha=0.8, edgecolor='white', linewidth=0.5,
                            hatch=HATCH_PATTERNS[method_key]))
    
    ax2.set_xlabel('Transmit Power (mW)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Average User Rate (Mbps)', fontsize=16, fontweight='bold')
    ax2.set_title('(b) Performance vs Transmit Power', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'{power}mW' for power in power_values], fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # 添加数值标签
    for bar_group in bars2:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12,
                           fontweight='bold')
    
    # 设置y轴范围
    max_user_rate = max([max(rates) for rates in user_method_data.values() if rates])
    max_power_rate = max([max(rates) for rates in power_method_data.values() if rates])
    ax1.set_ylim(0, max_user_rate * 1.15)
    ax2.set_ylim(0, max_power_rate * 1.15)
    
    # 显示所有边框线
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
    
    # 添加统一图例（在上图，放大字号）
    legend = ax1.legend(loc='upper left', fontsize=16, framealpha=0.9, 
                       fancybox=True, shadow=True, ncol=3)
    legend.get_frame().set_facecolor('white')
    
    # 添加整体标题（英文）
    fig.suptitle('Comprehensive Performance Comparison of Five Antenna Methods', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # 最终强制设置边框（确保不被覆盖）
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    # 强制刷新图表
    fig.canvas.draw()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存图表
    plt.savefig(f"{output_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/comprehensive_comparison.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 综合对比图已保存至: {output_dir}/comprehensive_comparison.png")
    
    return fig


def create_performance_summary_table(user_counts, user_method_data, 
                                   power_values, power_method_data, output_dir):
    """创建性能汇总表格"""
    print("\n📊 六种方法性能汇总")
    print("=" * 120)
    
    # 用户数量测试汇总
    print("\n🔸 用户数量测试结果 (平均速率 Mbps)")
    print("-" * 80)
    header = f"{'用户数':<10}"
    for method_key in METHOD_ORDER:
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        header += f"{method_name:<12}"
    print(header)
    print("-" * 80)
    
    for i, user_count in enumerate(user_counts):
        row = f"{user_count}车+5空{'':<2}"
        for method_key in METHOD_ORDER:
            rate = user_method_data[method_key][i] if i < len(user_method_data[method_key]) else 0
            row += f"{rate:<12.1f}"
        print(row)
    
    # 功率测试汇总
    print(f"\n🔸 功率测试结果 (平均速率 Mbps, 30车+5空用户)")
    print("-" * 80)
    header = f"{'功率':<10}"
    for method_key in METHOD_ORDER:
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        header += f"{method_name:<12}"
    print(header)
    print("-" * 80)
    
    for i, power in enumerate(power_values):
        row = f"{power}mW{'':<6}"
        for method_key in METHOD_ORDER:
            rate = power_method_data[method_key][i] if i < len(power_method_data[method_key]) else 0
            row += f"{rate:<12.1f}"
        print(row)
    
    # 计算相对性能提升
    print(f"\n🔸 相对传统FPA的性能提升 (%)")
    print("-" * 80)
    
    # 用户数量测试的平均提升
    print("用户数量测试平均提升:")
    for method_key in METHOD_ORDER:
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        if method_key != 'fpa' and user_method_data[method_key]:
            improvements = []
            for i in range(len(user_counts)):
                if (i < len(user_method_data['fpa']) and i < len(user_method_data[method_key]) and
                    user_method_data['fpa'][i] > 0):
                    improvement = ((user_method_data[method_key][i] - user_method_data['fpa'][i]) / 
                                 user_method_data['fpa'][i]) * 100
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                print(f"  {method_name}: {avg_improvement:+.1f}%")
    
    # 功率测试的平均提升
    print("\n功率测试平均提升:")
    for method_key in METHOD_ORDER:
        method_name = METHOD_LABELS_EN[method_key]  # 使用英文标签
        if method_key != 'fpa' and power_method_data[method_key]:
            improvements = []
            for i in range(len(power_values)):
                if (i < len(power_method_data['fpa']) and i < len(power_method_data[method_key]) and
                    power_method_data['fpa'][i] > 0):
                    improvement = ((power_method_data[method_key][i] - power_method_data['fpa'][i]) / 
                                 power_method_data['fpa'][i]) * 100
                    improvements.append(improvement)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                print(f"  {method_name}: {avg_improvement:+.1f}%")


def main():
    """主函数：生成所有可视化图表"""
    print("🚀 六种天线方法结果可视化工具")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有测试结果
    print("📂 加载测试结果数据...")
    all_results = load_test_results()
    
    # 提取数据
    print("\n🔄 处理数据...")
    user_counts, user_method_data = extract_user_count_data(all_results)
    power_values, power_method_data = extract_power_data(all_results)
    
    print(f"  用户数量测试: {len(user_counts)}个数据点")
    print(f"  功率测试: {len(power_values)}个数据点")
    
    # 检查数据完整性
    if not user_counts:
        print("❌ 未找到用户数量测试数据")
    if not power_values:
        print("❌ 未找到功率测试数据")
    
    if not user_counts and not power_values:
        print("❌ 没有可用的测试数据，请先运行测试")
        return
    
    # 生成图表
    print(f"\n🎨 生成可视化图表...")
    
    if user_counts:
        print("  生成用户数量对比图...")
        create_user_count_comparison_chart(user_counts, user_method_data, output_dir)
    
    if power_values:
        print("  生成功率对比图...")
        create_power_comparison_chart(power_values, power_method_data, output_dir)
    
    if user_counts and power_values:
        print("  生成综合对比图...")
        create_comprehensive_comparison_chart(user_counts, user_method_data, 
                                            power_values, power_method_data, output_dir)
    
    # 生成性能汇总表格
    if user_counts or power_values:
        create_performance_summary_table(user_counts, user_method_data, 
                                       power_values, power_method_data, output_dir)
    
    print(f"\n🎉 可视化完成！")
    print(f"📁 所有图表已保存至: {output_dir}/")
    print(f"   - user_count_comparison.png/pdf: 用户数量对比图")
    print(f"   - power_comparison.png/pdf: 功率对比图")
    print(f"   - comprehensive_comparison.png/pdf: 综合对比图")
    print("=" * 80)


if __name__ == "__main__":
    main()
