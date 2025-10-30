

import numpy as np
import time
import os
from typing import List, Dict

from sixDMA_Environment_core_class import SystemParams, UserMobility
from grid_based_antenna_optimizer import GridBasedAntennaOptimizer


def demo_basic_optimization():
    """演示基于理论速率的基本网格优化功能 (3GPP标准)"""
    print("="*80)
    print("演示1: 基于理论速率的基本网格优化分析 (3GPP标准)")
    print("="*80)
    
    # 初始化系统参数 - 网格优化不需要指定用户数量
    params = SystemParams(
        environment_size=(300, 300, 100),
        air_height_range=(50.0, 100.0)
    )
    print(f"系统参数 (3GPP标准):")
    print(f"  环境尺寸: {params.environment_size}")
    print(f"  基站位置: {params.base_station_pos}")
    print(f"  天线表面数: {params.num_surfaces}")
    print(f"  载波频率: {params.fc/1e9:.1f} GHz")
    print(f"  空中高度范围: {params.air_height_range}")
    
    # 创建优化器
    optimizer = GridBasedAntennaOptimizer(
        params=params,
        enable_parallel=True,  # 启用并行计算
        cache_results=True     # 启用结果缓存
    )
    
    # 运行优化 (使用3GPP标准路径损耗模型和更新的空中网格范围)
    optimizer.sampling_config['hemisphere_samples'] = 80  # 80个采样点
    optimizer.sampling_config['users_per_grid'] = 20     # 每网格20个用户

    start_time = time.time()
    
    try:
        results = optimizer.run_complete_optimization(
            output_dir="demo_optimization_results"
        )
        
        total_time = time.time() - start_time
        
        if results:
            print(f"\n✅ 优化成功完成！")
            print(f"总耗时: {total_time:.2f}秒")
            
            # 显示主要结果
            summary = results['summary']
            print(f"\n📊 优化结果摘要:")
            print(f"  分析网格数: {summary['analyzed_grids']}/{summary['total_grids']}")
            print(f"  天线位置数: {summary['total_antenna_positions']}")
            
            # 显示最佳天线位置
            rankings = results['antenna_ranking'][:5]
            print(f"\n🏆 前5个最佳天线位置:")
            for i, rank in enumerate(rankings):
                print(f"  {i+1}. 位置索引 {rank['position_idx']}: "
                      f"得分 {rank['composite_score']:.3f}, "
                      f"覆盖 {rank['coverage_count']} 个网格")
            
            # 显示配置统计
            config_stats = results.get('config_statistics', {})
            if config_stats:
                print(f"\n📊 配置统计:")
                print(f"  分析的网格配置数: {len(config_stats.get('rate_distribution', []))}")
                if config_stats.get('rate_distribution'):
                    import numpy as np
                    rates = np.array(config_stats['rate_distribution'])
                    print(f"  平均理论速率: {np.mean(rates):.2f} Mbps")
                    print(f"  最大理论速率: {np.max(rates):.2f} Mbps")
            
            return results
        
        else:
            print("❌ 优化失败")
            return None
    
    except Exception as e:
        print(f"❌ 优化过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def demo_user_adaptive_optimization(optimization_results):
    """演示基于用户分布的自适应优化"""
    if not optimization_results:
        print("跳过演示2: 需要基本优化结果")
        return
    
    print("\n" + "="*80)
    print("演示2: 基于用户分布的自适应优化")
    print("="*80)
    
    # 生成实际的用户分布
    params = SystemParams()
    users = UserMobility.generate_user_positions(params, seed=42)
    user_positions = [user.position for user in users]
    
    print(f"生成用户分布:")
    print(f"  地面用户: {sum(1 for u in users if u.type == 'vehicle')}个")
    print(f"  空中用户: {sum(1 for u in users if u.type == 'UAV')}个")
    
    # 创建优化器并加载之前的结果
    optimizer = GridBasedAntennaOptimizer(
        params=params,
        enable_parallel=True,
        cache_results=True
    )
    
    # 从pickle文件加载完整数据
    try:
        import pickle
        with open("demo_optimization_results/complete_optimization_data.pkl", 'rb') as f:
            data = pickle.load(f)
            optimizer.grid_cells = data['grid_cells']
            optimizer.antenna_grid_gains = data['antenna_grid_gains']
        
        print(f"\n✅ 成功加载优化数据")
        
        # 基于实际用户分布计算自适应部署策略
        print(f"\n计算自适应部署策略...")
        adaptive_strategy = optimizer.get_deployment_strategy_for_user_distribution(user_positions)
        
        # 显示结果
        if adaptive_strategy:
            user_grid_mapping = adaptive_strategy.get('user_grid_mapping', {})
            grid_weights = adaptive_strategy.get('grid_weights', {})
            weighted_strategy = adaptive_strategy.get('weighted_deployment_strategy', {})
            
            print(f"\n📊 用户分布分析:")
            print(f"  总用户数: {adaptive_strategy.get('total_users', 0)}")
            print(f"  占用网格数: {len([g for g in grid_weights.values() if g > 0])}")
            print(f"  最大网格用户数: {max(grid_weights.values()) if grid_weights else 0}")
            
            weighted_ranking = weighted_strategy.get('weighted_ranking', [])
            if weighted_ranking:
                print(f"\n🎯 基于理论速率的自适应部署建议 (前5个位置):")
                for i, pos in enumerate(weighted_ranking[:5]):
                    print(f"  {i+1}. 位置索引 {pos['position_idx']}: "
                          f"加权速率得分 {pos['weighted_score']:.3f}")
                
                # 显示基于实际用户分布的最佳天线位置
                print(f"\n📊 基于实际用户分布的天线位置分析:")
                print(f"  最佳自适应位置 (前{params.num_surfaces}个):")
                for i, pos in enumerate(weighted_ranking[:params.num_surfaces]):
                    print(f"    {i+1}. 位置索引 {pos['position_idx']}: 加权速率得分 {pos['weighted_score']:.3f}")
            
            # 显示覆盖分析
            occupied_grids = len([g for g in grid_weights.values() if g > 0])
            total_users_in_occupied_grids = sum(g for g in grid_weights.values() if g > 0)
            print(f"\n📈 用户分布覆盖:")
            total_grids = optimizer.grid_config['total_grids']
            print(f"  有用户的网格: {occupied_grids}/{total_grids} ({occupied_grids/total_grids:.1%})")
            print(f"  这些网格中的用户: {total_users_in_occupied_grids}个")
        else:
            print(f"❌ 无法获取自适应策略结果")
        
    except Exception as e:
        print(f"❌ 自适应优化失败: {str(e)}")


def demo_computational_analysis():
    """演示计算复杂度分析"""
    print("\n" + "="*80)
    print("演示3: 计算复杂度和可行性分析")
    print("="*80)
    
    # 分析不同配置的计算复杂度（使用默认网格配置）
    params = SystemParams()
    default_optimizer = GridBasedAntennaOptimizer(params)
    default_grids = default_optimizer.grid_config['total_grids']
    
    configurations = [
        {"name": "快速模式", "grids": default_grids, "samples": 20, "users": 5},
        {"name": "标准模式", "grids": default_grids, "samples": 50, "users": 20},
        {"name": "高精度模式", "grids": default_grids, "samples": 100, "users": 50},
    ]
    
    print(f"计算复杂度分析 (基于理论速率计算):")
    print(f"{'模式':<12} {'网格数':<8} {'天线采样':<10} {'用户/网格':<10} {'总计算量':<12} {'预估时间'}")
    print("-" * 70)
    
    for config in configurations:
        total_computations = config['grids'] * config['samples'] * config['users']
        # 基于经验估算：每个速率计算约0.002秒（包括4×4信道矩阵和SINR计算）
        estimated_time = total_computations * 0.002 / 4  # 4核并行
        
        print(f"{config['name']:<12} {config['grids']:<8} {config['samples']:<10} "
              f"{config['users']:<10} {total_computations:<12} {estimated_time:.1f}秒")


def main():
    """主演示函数"""
    print("🚀 Grid-Based Antenna Optimization System Demo")
    print("网格天线优化系统完整演示")
    print("="*80)
    
    # 演示1: 基本优化功能
    optimization_results = demo_basic_optimization()
    
    # 演示2: 用户自适应优化
    demo_user_adaptive_optimization(optimization_results)

    
    print(f"\n" + "="*80)
    print(f"🎉 演示完成！")
    print(f"所有结果已保存至 'demo_optimization_results/' 目录")
    print(f"="*80)


if __name__ == "__main__":
    main()
