

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

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


def load_grid_optimization_data():
    """加载网格优化数据"""
    try:
        with open('demo_optimization_results/optimization_analysis.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("✅ 成功加载网格优化分析数据")
            return data
    except FileNotFoundError:
        print("❌ 未找到demo_optimization_results/optimization_analysis.json")
        print("   请先运行 python demo_grid_optimization.py 生成优化结果")
        return None
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return None


def extract_grid_performance_data(optimization_data):
    """提取网格性能数据
    
    注意：原数据基于100MHz带宽，现调整为20MHz带宽（更实际），
    因此所有速率值都除以5进行调整。
    """
    if not optimization_data:
        return None, None
    
    grid_analysis = optimization_data.get('grid_analysis', {})
    
    print(f"📊 分析网格性能数据...")
    print(f"  总网格数: {len(grid_analysis)}")
    
    # 存储网格性能数据
    grid_performance = {}
    grid_positions = {}
    
    # 统计信息
    ground_grids = 0
    air_grids = 0
    max_rate = 0.0
    min_rate = float('inf')
    
    for grid_id_str, grid_info in grid_analysis.items():
        grid_id = int(grid_id_str)
        
        # 提取网格信息
        grid_type = grid_info.get('grid_type', 'unknown')
        grid_center = grid_info.get('grid_center', [0, 0, 0])
        best_rate = grid_info.get('best_average_rate_mbps', 0.0)
        best_max_rate = grid_info.get('best_max_rate_mbps', 0.0)
        
        # 使用最大速率作为性能指标
        performance_value = best_max_rate if best_max_rate > 0 else best_rate
        # 调整带宽：从100MHz调整到20MHz（除以5）
        performance_value = performance_value / 5.0
        
        grid_performance[grid_id] = {
            'grid_type': grid_type,
            'center_position': grid_center,
            'best_average_rate': best_rate / 5.0,  # 调整带宽
            'best_max_rate': best_max_rate / 5.0,  # 调整带宽
            'performance_value': performance_value
        }
        
        grid_positions[grid_id] = grid_center
        
        # 统计
        if grid_type == 'ground':
            ground_grids += 1
        elif grid_type == 'air':
            air_grids += 1
        
        if performance_value > max_rate:
            max_rate = performance_value
        if performance_value < min_rate and performance_value > 0:
            min_rate = performance_value
    
    print(f"  地面网格: {ground_grids}个")
    print(f"  空中网格: {air_grids}个")
    print(f"  性能范围: {min_rate:.3f} - {max_rate:.3f} Mbps")
    
    return grid_performance, grid_positions


def organize_grids_for_visualization(grid_performance, grid_positions):
    """组织网格数据用于可视化
    
    将800个网格组织成2D矩阵：
    - 20列 × 40行 = 800个网格
    - 每2行为一组：第1行空中网格，第2行地面网格
    - 按照物理位置的x,y坐标排序
    """
    print(f"📐 组织网格数据用于可视化...")
    
    # 分离地面和空中网格
    ground_grids = {}
    air_grids = {}
    
    for grid_id, perf_data in grid_performance.items():
        if perf_data['grid_type'] == 'ground':
            ground_grids[grid_id] = perf_data
        elif perf_data['grid_type'] == 'air':
            air_grids[grid_id] = perf_data
    
    print(f"  分离得到: {len(ground_grids)}个地面网格, {len(air_grids)}个空中网格")
    
    # 按照x,y坐标对网格进行排序
    def sort_grids_by_position(grids_dict):
        """按照x,y坐标排序网格"""
        grid_list = []
        for grid_id, perf_data in grids_dict.items():
            x, y, z = perf_data['center_position']
            grid_list.append({
                'grid_id': grid_id,
                'x': x,
                'y': y,
                'z': z,
                'performance': perf_data['performance_value']
            })
        
        # 按y坐标排序，然后按x坐标排序
        grid_list.sort(key=lambda g: (g['y'], g['x']))
        return grid_list
    
    sorted_ground_grids = sort_grids_by_position(ground_grids)
    sorted_air_grids = sort_grids_by_position(air_grids)
    
    # 假设是20×20的网格布局
    grid_cols = 20
    grid_rows_per_type = 20  # 每种类型20行
    
    # 创建可视化矩阵 (40行 × 20列)
    # 奇数行：空中网格，偶数行：地面网格
    visualization_matrix = np.zeros((40, 20))
    grid_id_matrix = np.full((40, 20), -1, dtype=int)
    
    # 填充地面网格数据
    for i, grid_data in enumerate(sorted_ground_grids):
        if i < 400:  # 确保不超出范围
            row = (i // 20) * 2 + 1  # 偶数行（从1开始）
            col = i % 20
            visualization_matrix[row, col] = grid_data['performance']
            grid_id_matrix[row, col] = grid_data['grid_id']
    
    # 填充空中网格数据
    for i, grid_data in enumerate(sorted_air_grids):
        if i < 400:  # 确保不超出范围
            row = (i // 20) * 2  # 奇数行（从0开始）
            col = i % 20
            visualization_matrix[row, col] = grid_data['performance']
            grid_id_matrix[row, col] = grid_data['grid_id']
    
    print(f"  组织完成: 40行 × 20列 矩阵")
    print(f"  行模式: 奇数行=空中网格, 偶数行=地面网格")
    
    return visualization_matrix, grid_id_matrix


def create_grid_performance_heatmap(visualization_matrix, grid_id_matrix, output_dir):
    """创建网格性能热图"""
    print(f"🎨 创建网格性能热图...")
    
    fig, ax = plt.subplots(figsize=(12, 16))  # 调整比例适应40行×20列
    
    # 创建热图
    # 使用对数尺度来更好地显示数据差异
    masked_matrix = np.ma.masked_where(visualization_matrix == 0, visualization_matrix)
    
    # 使用高对比度颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    
    # 定义颜色节点：黑色->深红->红色->橙色->黄色->白色
    colors = ['#000000', '#800000', '#FF0000', '#FF8000', '#FFFF00', '#FFFFFF']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=n_bins)
    
    cmap = custom_cmap
    
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', 
                   interpolation='nearest', origin='upper')
    
    # 设置标题和标签
    ax.set_title('Grid Performance Heatmap: Maximum Theoretical Rate per Grid', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Grid Column Index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Grid Row Index (Air-Ground Alternating)', fontsize=18, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(0, 20, 2))
    ax.set_xticklabels(np.arange(0, 20, 2))
    
    # y轴标签：标注空中和地面网格
    y_ticks = []
    y_labels = []
    for i in range(0, 40, 4):  # 每4行标注一次
        y_ticks.extend([i, i+1])
        y_labels.extend([f'Air {i//2}', f'Ground {i//2}'])
    
    ax.set_yticks(y_ticks[:20])  # 限制标签数量
    ax.set_yticklabels(y_labels[:20], fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Maximum Rate (Mbps, 20MHz BW)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # 添加网格分隔线
    # 水平线：分隔空中和地面网格
    for i in range(1, 40, 2):
        ax.axhline(y=i-0.5, color='white', linewidth=1.5, alpha=0.8)
    
    # 垂直线：分隔列
    for i in range(1, 20):
        ax.axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.5)
    
    # 强制设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/grid_performance_heatmap.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/grid_performance_heatmap.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 网格性能热图已保存至: {output_dir}/grid_performance_heatmap.png")
    
    return fig


def create_detailed_grid_analysis(grid_performance, output_dir):
    """创建详细的网格分析"""
    print(f"📈 生成详细网格分析...")
    
    # 分析统计信息
    ground_performances = []
    air_performances = []
    
    for grid_id, perf_data in grid_performance.items():
        perf_value = perf_data['performance_value']
        if perf_value > 0:  # 只考虑有效数据
            if perf_data['grid_type'] == 'ground':
                ground_performances.append(perf_value)
            elif perf_data['grid_type'] == 'air':
                air_performances.append(perf_value)
    
    # 打印统计信息
    print(f"\n📊 网格性能统计分析:")
    print("=" * 80)
    
    if ground_performances:
        print(f"🚗 地面网格性能统计:")
        print(f"  有效网格数: {len(ground_performances)}")
        print(f"  平均性能: {np.mean(ground_performances):.3f} Mbps")
        print(f"  最大性能: {np.max(ground_performances):.3f} Mbps")
        print(f"  最小性能: {np.min(ground_performances):.3f} Mbps")
        print(f"  标准差: {np.std(ground_performances):.3f} Mbps")
    
    if air_performances:
        print(f"\n✈️ 空中网格性能统计:")
        print(f"  有效网格数: {len(air_performances)}")
        print(f"  平均性能: {np.mean(air_performances):.3f} Mbps")
        print(f"  最大性能: {np.max(air_performances):.3f} Mbps")
        print(f"  最小性能: {np.min(air_performances):.3f} Mbps")
        print(f"  标准差: {np.std(air_performances):.3f} Mbps")
    
    # 对比分析
    if ground_performances and air_performances:
        ground_avg = np.mean(ground_performances)
        air_avg = np.mean(air_performances)
        improvement = ((air_avg - ground_avg) / ground_avg) * 100
        
        print(f"\n🔄 地面vs空中对比:")
        print(f"  空中网格平均性能相对地面网格: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"  ✅ 空中网格总体性能更好")
        else:
            print(f"  ⚠️ 地面网格总体性能更好")
    
    # 找出性能最佳的网格
    best_grid_id = None
    best_performance = 0.0
    worst_grid_id = None
    worst_performance = float('inf')
    
    for grid_id, perf_data in grid_performance.items():
        perf_value = perf_data['performance_value']
        if perf_value > 0:
            if perf_value > best_performance:
                best_performance = perf_value
                best_grid_id = grid_id
            if perf_value < worst_performance:
                worst_performance = perf_value
                worst_grid_id = grid_id
    
    if best_grid_id is not None:
        best_grid = grid_performance[best_grid_id]
        print(f"\n🏆 最佳性能网格:")
        print(f"  网格ID: {best_grid_id}")
        print(f"  类型: {best_grid['grid_type']}")
        print(f"  位置: {best_grid['center_position']}")
        print(f"  最大速率: {best_performance:.3f} Mbps")
    
    if worst_grid_id is not None:
        worst_grid = grid_performance[worst_grid_id]
        print(f"\n📉 最低性能网格:")
        print(f"  网格ID: {worst_grid_id}")
        print(f"  类型: {worst_grid['grid_type']}")
        print(f"  位置: {worst_grid['center_position']}")
        print(f"  最大速率: {worst_performance:.3f} Mbps")
    
    # 保存分析结果
    analysis_summary = {
        'total_grids': len(grid_performance),
        'ground_grids_count': len(ground_performances),
        'air_grids_count': len(air_performances),
        'ground_stats': {
            'mean': np.mean(ground_performances) if ground_performances else 0,
            'max': np.max(ground_performances) if ground_performances else 0,
            'min': np.min(ground_performances) if ground_performances else 0,
            'std': np.std(ground_performances) if ground_performances else 0
        },
        'air_stats': {
            'mean': np.mean(air_performances) if air_performances else 0,
            'max': np.max(air_performances) if air_performances else 0,
            'min': np.min(air_performances) if air_performances else 0,
            'std': np.std(air_performances) if air_performances else 0
        },
        'best_grid': {
            'grid_id': best_grid_id,
            'performance': best_performance,
            'type': grid_performance[best_grid_id]['grid_type'] if best_grid_id else None,
            'position': grid_performance[best_grid_id]['center_position'] if best_grid_id else None
        },
        'worst_grid': {
            'grid_id': worst_grid_id,
            'performance': worst_performance,
            'type': grid_performance[worst_grid_id]['grid_type'] if worst_grid_id else None,
            'position': grid_performance[worst_grid_id]['center_position'] if worst_grid_id else None
        }
    }
    
    with open(f"{output_dir}/grid_performance_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    return analysis_summary


def create_enhanced_grid_heatmap(grid_performance, output_dir):
    """创建增强的网格热图（类似你提供的示例）"""
    print(f"🎨 创建增强网格热图...")
    
    # 组织数据：按照物理位置创建矩阵
    # 假设环境是300×300，分成20×20的网格
    grid_size_x = 20  # x方向网格数
    grid_size_y = 20  # y方向网格数
    
    # 创建地面和空中的性能矩阵
    ground_matrix = np.zeros((grid_size_y, grid_size_x))
    air_matrix = np.zeros((grid_size_y, grid_size_x))
    
    # 填充数据
    for grid_id, perf_data in grid_performance.items():
        x, y, z = perf_data['center_position']
        
        # 将物理坐标转换为网格索引
        # 假设环境范围是0-300m
        col_idx = int(x / 15)  # 300/20 = 15m per grid
        row_idx = int(y / 15)
        
        # 确保索引在有效范围内
        col_idx = np.clip(col_idx, 0, grid_size_x - 1)
        row_idx = np.clip(row_idx, 0, grid_size_y - 1)
        
        if perf_data['grid_type'] == 'ground':
            ground_matrix[row_idx, col_idx] = perf_data['performance_value']
        elif perf_data['grid_type'] == 'air':
            air_matrix[row_idx, col_idx] = perf_data['performance_value']
    
    # 创建组合矩阵：空中网格在上，地面网格在下
    combined_matrix = np.vstack([air_matrix, ground_matrix])
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 使用对数尺度处理数据（如果数据范围很大）
    masked_matrix = np.ma.masked_where(combined_matrix == 0, combined_matrix)
    
    # 选择颜色映射（更明显的颜色分区）
    # 创建自定义高对比度颜色映射，类似你的示例图
    from matplotlib.colors import LinearSegmentedColormap
    
    # 定义颜色节点：黑色->深红->红色->橙色->黄色->白色
    colors = ['#000000', '#800000', '#FF0000', '#FF8000', '#FFFF00', '#FFFFFF']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=n_bins)
    
    # 使用自定义颜色映射
    cmap = custom_cmap
    
    # 备选高对比度颜色映射：
    # cmap = plt.cm.jet      # 蓝色->青色->绿色->黄色->红色
    # cmap = plt.cm.hot      # 黑色->红色->黄色->白色  
    # cmap = plt.cm.inferno  # 黑色->紫色->红色->黄色
    
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', 
                   interpolation='nearest', origin='upper')
    
    # 设置标题和标签
    ax.set_title('Grid Performance Heatmap: Maximum Theoretical Rate', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Grid Column Index', fontsize=18, fontweight='bold')
    ax.set_ylabel('Grid Row Index', fontsize=18, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(0, 20, 2))
    ax.set_xticklabels(np.arange(0, 20, 2))
    
    # y轴标签：区分空中和地面
    y_ticks = [5, 15, 25, 35]
    y_labels = ['Air Grids', 'Air Grids', 'Ground Grids', 'Ground Grids']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    # 添加分隔线
    ax.axhline(y=19.5, color='white', linewidth=3, alpha=0.9)  # 分隔空中和地面
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Maximum Rate (Mbps, 20MHz BW)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # 强制设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/enhanced_grid_heatmap.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/enhanced_grid_heatmap.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 增强网格热图已保存至: {output_dir}/enhanced_grid_heatmap.png")
    
    return fig


def create_ground_grids_only_heatmap(grid_performance, output_dir):
    """创建仅地面网格的性能热图"""
    print(f"🎨 创建地面网格专用热图...")
    
    # 提取地面网格数据
    ground_grids = {}
    for grid_id, perf_data in grid_performance.items():
        if perf_data['grid_type'] == 'ground':
            ground_grids[grid_id] = perf_data
    
    print(f"  地面网格数量: {len(ground_grids)}")
    
    # 组织地面网格数据：按照物理位置创建矩阵
    # 假设地面网格是20×20的布局
    grid_size_x = 20  # x方向网格数
    grid_size_y = 20  # y方向网格数
    
    ground_matrix = np.zeros((grid_size_y, grid_size_x))
    ground_id_matrix = np.full((grid_size_y, grid_size_x), -1, dtype=int)
    
    # 填充地面网格数据
    for grid_id, perf_data in ground_grids.items():
        x, y, z = perf_data['center_position']
        
        # 将物理坐标转换为网格索引
        # 假设环境范围是0-300m
        col_idx = int(x / 15)  # 300/20 = 15m per grid
        row_idx = int(y / 15)
        
        # 确保索引在有效范围内
        col_idx = np.clip(col_idx, 0, grid_size_x - 1)
        row_idx = np.clip(row_idx, 0, grid_size_y - 1)
        
        ground_matrix[row_idx, col_idx] = perf_data['performance_value']
        ground_id_matrix[row_idx, col_idx] = grid_id
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(8, 8))  # 正方形比例适合20×20网格
    
    # 处理数据
    masked_matrix = np.ma.masked_where(ground_matrix == 0, ground_matrix)
    
    # 使用自定义高对比度颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    
    # 定义颜色节点：黑色->深红->红色->橙色->黄色->白色
    colors = ['#000000', '#800000', '#FF0000', '#FF8000', '#FFFF00', '#FFFFFF']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=n_bins)
    
    cmap = custom_cmap
    
    im = ax.imshow(masked_matrix, cmap=cmap, aspect='equal', 
                   interpolation='nearest', origin='upper')
    
    # 设置标题和标签
    ax.set_xlabel('Grid Column Index', fontsize=20, fontweight='bold')
    ax.set_ylabel('Grid Row Index', fontsize=20, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(0, 20, 2))
    ax.set_xticklabels(np.arange(0, 20, 2))
    ax.set_yticks(np.arange(0, 20, 2))
    ax.set_yticklabels(np.arange(0, 20, 2))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Maximum Rate (Mbps, 20MHz BW)', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # 添加网格分隔线
    for i in range(1, 20):
        ax.axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # 强制设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # 强制刷新图表
    fig.canvas.draw()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/ground_grids_heatmap.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    plt.savefig(f"{output_dir}/ground_grids_heatmap.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='black')
    
    print(f"📊 地面网格热图已保存至: {output_dir}/ground_grids_heatmap.png")
    
    # 生成地面网格详细统计
    ground_performances = [perf_data['performance_value'] for perf_data in ground_grids.values() 
                          if perf_data['performance_value'] > 0]
    
    if ground_performances:
        print(f"\n🚗 地面网格专项分析:")
        print(f"  有效网格数: {len(ground_performances)}/{len(ground_grids)}")
        print(f"  平均性能: {np.mean(ground_performances):.3f} Mbps")
        print(f"  最大性能: {np.max(ground_performances):.3f} Mbps")
        print(f"  最小性能: {np.min(ground_performances):.3f} Mbps")
        print(f"  标准差: {np.std(ground_performances):.3f} Mbps")
        print(f"  性能范围: {np.max(ground_performances) - np.min(ground_performances):.3f} Mbps")
        
        # 找出最佳和最差的地面网格
        best_perf = np.max(ground_performances)
        worst_perf = np.min(ground_performances)
        
        for grid_id, perf_data in ground_grids.items():
            if perf_data['performance_value'] == best_perf:
                print(f"  🏆 最佳地面网格: ID{grid_id}, 位置{perf_data['center_position']}, 性能{best_perf:.3f} Mbps")
                break
        
        for grid_id, perf_data in ground_grids.items():
            if perf_data['performance_value'] == worst_perf:
                print(f"  📉 最差地面网格: ID{grid_id}, 位置{perf_data['center_position']}, 性能{worst_perf:.3f} Mbps")
                break
    
    return fig


def main():
    """主函数：生成网格性能热图"""
    print("🚀 网格性能热图生成工具")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = "grid_heatmap_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载优化数据
    print("📂 加载网格优化数据...")
    optimization_data = load_grid_optimization_data()
    
    if optimization_data is None:
        return
    
    # 提取网格性能数据
    grid_performance, grid_positions = extract_grid_performance_data(optimization_data)
    
    if grid_performance is None:
        print("❌ 无法提取网格性能数据")
        return
    
    # 生成详细分析
    analysis_summary = create_detailed_grid_analysis(grid_performance, output_dir)
    
    # 生成热图
    print(f"\n🎨 生成可视化图表...")
    
    # 创建增强热图（空中+地面）
    create_enhanced_grid_heatmap(grid_performance, output_dir)
    
    # 创建地面网格专用热图
    create_ground_grids_only_heatmap(grid_performance, output_dir)
    
    # 组织数据并创建标准热图
    visualization_matrix, grid_id_matrix = organize_grids_for_visualization(
        grid_performance, grid_positions)
    create_grid_performance_heatmap(visualization_matrix, grid_id_matrix, output_dir)
    
    print(f"\n🎉 网格性能分析完成！")
    print(f"📁 结果已保存至: {output_dir}/")
    print(f"   - enhanced_grid_heatmap.png/pdf: 增强网格热图（空中+地面）")
    print(f"   - ground_grids_heatmap.png/pdf: 地面网格专用热图")
    print(f"   - grid_performance_heatmap.png/pdf: 标准网格热图")
    print(f"   - grid_performance_analysis.json: 详细分析数据")
    print("=" * 80)


if __name__ == "__main__":
    main()
