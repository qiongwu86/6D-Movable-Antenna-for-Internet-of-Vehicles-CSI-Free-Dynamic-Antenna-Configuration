import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 尝试导入sklearn，如果失败则使用简单采样
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，将使用简单的均匀采样方法")

# Import existing classes
from sixDMA_Environment_core_class import (
    SystemParams, ActionSpace, ChannelModel, Antenna, User
)


@dataclass
class GridCell:
    """网格单元类"""
    grid_id: int
    grid_type: str  # 'ground' or 'air'
    center_position: np.ndarray
    bounds: Dict[str, Tuple[float, float]]  # x, y, z boundaries
    user_positions: List[np.ndarray]  # 该网格内的用户位置
    
    def __post_init__(self):
        if len(self.user_positions) == 0:
            self.user_positions = []


@dataclass
class AntennaGainResult:
    """天线速率分析结果（保持类名兼容性）"""
    antenna_position_idx: int
    antenna_position: np.ndarray
    antenna_normal: np.ndarray
    rotation_type: str
    average_gain: float  # 实际存储平均速率(Mbps)，保持字段名兼容性
    max_gain: float      # 实际存储最大速率(Mbps)，保持字段名兼容性
    min_gain: float      # 实际存储最小速率(Mbps)，保持字段名兼容性
    gain_variance: float # 实际存储速率方差，保持字段名兼容性
    user_gains: List[float] # 实际存储每个用户的速率(Mbps)，保持字段名兼容性


class GridBasedAntennaOptimizer:
    """基于网格的天线优化器"""
    
    def __init__(self, params: SystemParams, enable_parallel: bool = True, cache_results: bool = True):
        self.params = params
        self.enable_parallel = enable_parallel
        self.cache_results = cache_results
        
        # 初始化动作空间管理器
        self.action_space_manager = ActionSpace(params)
        
        # 网格配置
        self.grid_config = {
            'total_grids': 800,
            'ground_grids': 400,
            'air_grids': 400,
            'ground_grid_size': (20, 20),  # 20x20 ground grids
            'air_grid_size': (20, 20),     # 20x20 air grids
            'ground_height': 1.5,          # 车辆高度
            'air_height_range': params.air_height_range
        }
        
        # 采样配置
        self.sampling_config = {
            'hemisphere_samples': 80,
            'users_per_grid': 20,  # 每个网格内模拟的用户数量
            'neighbor_expansion_radius': 2  # 邻居扩展半径
        }
        
        # 存储结构
        self.grid_cells: List[GridCell] = []
        self.antenna_grid_gains: Dict[int, Dict[int, List[AntennaGainResult]]] = {}
        self.optimization_cache = {}
        
        # 性能统计
        self.computation_stats = {
            'total_time': 0,
            'grid_generation_time': 0,
            'hemisphere_sampling_time': 0,
            'channel_computation_time': 0,
            'gain_analysis_time': 0
        }
        
        print(f"Grid-Based Antenna Optimizer 初始化完成:")
        print(f"  网格配置: {self.grid_config['total_grids']}个网格 (地面{self.grid_config['ground_grids']} + 空中{self.grid_config['air_grids']})")
        print(f"  采样配置: 每网格{self.sampling_config['hemisphere_samples']}个天线位置, {self.sampling_config['users_per_grid']}个用户")
        print(f"  并行计算: {'启用' if enable_parallel else '禁用'}")
        print(f"  结果缓存: {'启用' if cache_results else '禁用'}")
    
    def run_complete_optimization(self, output_dir: str = "antenna_optimization_results") -> Dict:
        """运行完整的天线优化分析"""
        print(f"\n{'='*80}")
        print(f"开始基于网格的天线优化分析")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 步骤1: 生成网格空间
            print(f"\n步骤1: 生成{self.grid_config['total_grids']}个网格空间...")
            self._generate_grid_space()
            
            # 步骤2: 为每个网格生成用户分布
            print(f"\n步骤2: 为每个网格生成用户分布...")
            self._generate_users_for_grids()
            
            # 步骤3: 执行天线-网格增益分析
            print(f"\n步骤3: 执行天线-网格增益分析...")
            self._perform_antenna_grid_analysis()
            
            # 步骤4: 分析和总结结果
            print(f"\n步骤4: 分析和总结结果...")
            analysis_results = self._analyze_optimization_results()
            
            # 步骤4.5: 分析前10配置统计
            print(f"\n步骤4.5: 分析前10配置统计...")
            config_stats = self.analyze_top_configs_statistics(analysis_results)
            analysis_results['config_statistics'] = config_stats
            
            # 步骤5: 保存结果
            print(f"\n步骤5: 保存优化结果...")
            self._save_results(analysis_results, output_dir)
            
            # 步骤6: 生成可视化
            print(f"\n步骤6: 生成可视化结果...")
            self._generate_visualizations(analysis_results, output_dir)
            
            total_time = time.time() - start_time
            self.computation_stats['total_time'] = total_time
            
            print(f"\n{'='*80}")
            print(f"天线优化分析完成!")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"结果保存至: {output_dir}/")
            
            # 步骤7: 显示统计摘要
            print(f"\n步骤7: 显示前10配置统计摘要...")
            self._print_config_statistics_summary(analysis_results.get('config_statistics', {}))
            
            print(f"{'='*80}")
            
            return analysis_results
            
        except Exception as e:
            print(f"优化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _generate_grid_space(self):
        """生成网格空间 (地面 + 空中)"""
        start_time = time.time()
        
        env_size = self.params.environment_size
        grid_cells = []
        
        # 地面网格尺寸
        ground_x_step = env_size[0] / self.grid_config['ground_grid_size'][0]
        ground_y_step = env_size[1] / self.grid_config['ground_grid_size'][1]
        
        # 空中网格尺寸
        air_x_step = env_size[0] / self.grid_config['air_grid_size'][0]
        air_y_step = env_size[1] / self.grid_config['air_grid_size'][1]
        air_z_step = (self.grid_config['air_height_range'][1] - self.grid_config['air_height_range'][0]) / 10
        
        grid_id = 0
        
        # 生成地面网格
        ground_x_size, ground_y_size = self.grid_config['ground_grid_size']
        print(f"  生成地面网格: {ground_x_size}x{ground_y_size} = {ground_x_size * ground_y_size}个")
        for i in range(ground_x_size):
            for j in range(ground_y_size):
                x_center = (i + 0.5) * ground_x_step
                y_center = (j + 0.5) * ground_y_step
                z_center = self.grid_config['ground_height']
                
                bounds = {
                    'x': (i * ground_x_step, (i + 1) * ground_x_step),
                    'y': (j * ground_y_step, (j + 1) * ground_y_step),
                    'z': (z_center - 0.5, z_center + 0.5)  # 车辆高度范围
                }
                
                grid_cell = GridCell(
                    grid_id=grid_id,
                    grid_type='ground',
                    center_position=np.array([x_center, y_center, z_center]),
                    bounds=bounds,
                    user_positions=[]
                )
                
                grid_cells.append(grid_cell)
                grid_id += 1
        
        # 生成空中网格
        air_x_size, air_y_size = self.grid_config['air_grid_size']
        print(f"  生成空中网格: {air_x_size}x{air_y_size} = {air_x_size * air_y_size}个")
        for i in range(air_x_size):
            for j in range(air_y_size):
                x_center = (i + 0.5) * air_x_step
                y_center = (j + 0.5) * air_y_step
                
                # 空中网格的高度中心设为高度范围的中点
                z_center = (self.grid_config['air_height_range'][0] + 
                           self.grid_config['air_height_range'][1]) / 2
                
                # 所有空中网格都覆盖整个空中高度范围
                bounds = {
                    'x': (i * air_x_step, (i + 1) * air_x_step),
                    'y': (j * air_y_step, (j + 1) * air_y_step),
                    'z': (self.grid_config['air_height_range'][0], 
                          self.grid_config['air_height_range'][1])
                }
                
                grid_cell = GridCell(
                    grid_id=grid_id,
                    grid_type='air',
                    center_position=np.array([x_center, y_center, z_center]),
                    bounds=bounds,
                    user_positions=[]
                )
                
                grid_cells.append(grid_cell)
                grid_id += 1
        
        self.grid_cells = grid_cells
        
        generation_time = time.time() - start_time
        self.computation_stats['grid_generation_time'] = generation_time
        
        print(f"  网格生成完成: {len(grid_cells)}个网格, 耗时: {generation_time:.3f}秒")
        print(f"  地面网格: {sum(1 for g in grid_cells if g.grid_type == 'ground')}个")
        print(f"  空中网格: {sum(1 for g in grid_cells if g.grid_type == 'air')}个")
    
    def _generate_users_for_grids(self):
        """为每个网格生成均匀分布的用户"""
        print(f"  为每个网格生成{self.sampling_config['users_per_grid']}个用户...")
        
        total_users = 0
        
        for grid_cell in self.grid_cells:
            user_positions = []
            
            for _ in range(self.sampling_config['users_per_grid']):
                if grid_cell.grid_type == 'ground':
                    # 地面用户: 只在x,y平面均匀分布，z固定为车辆高度
                    x = np.random.uniform(grid_cell.bounds['x'][0], grid_cell.bounds['x'][1])
                    y = np.random.uniform(grid_cell.bounds['y'][0], grid_cell.bounds['y'][1])
                    z = self.grid_config['ground_height']
                    
                else:  # air grid
                    # 空中用户: 在三维空间均匀分布
                    x = np.random.uniform(grid_cell.bounds['x'][0], grid_cell.bounds['x'][1])
                    y = np.random.uniform(grid_cell.bounds['y'][0], grid_cell.bounds['y'][1])
                    z = np.random.uniform(grid_cell.bounds['z'][0], grid_cell.bounds['z'][1])
                
                user_positions.append(np.array([x, y, z]))
            
            grid_cell.user_positions = user_positions
            total_users += len(user_positions)
        
        print(f"  用户生成完成: 总共{total_users}个用户 ({total_users//self.grid_config['total_grids']}个/网格)")
    
    def _perform_antenna_grid_analysis(self):
        """执行天线-网格增益分析"""
        start_time = time.time()
        
        print(f"  开始分析{len(self.grid_cells)}个网格的天线增益...")
        
        if self.enable_parallel:
            # 并行处理
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for grid_cell in self.grid_cells:
                    future = executor.submit(self._analyze_single_grid, grid_cell)
                    futures.append((grid_cell.grid_id, future))
                
                completed = 0
                for grid_id, future in futures:
                    try:
                        grid_results = future.result()
                        self.antenna_grid_gains[grid_id] = grid_results
                        completed += 1
                        
                        if completed % 20 == 0:
                            print(f"    已完成: {completed}/{len(self.grid_cells)}个网格")
                    
                    except Exception as e:
                        print(f"    网格{grid_id}分析失败: {str(e)}")
                        self.antenna_grid_gains[grid_id] = {}
        else:
            # 串行处理
            for i, grid_cell in enumerate(self.grid_cells):
                try:
                    grid_results = self._analyze_single_grid(grid_cell)
                    self.antenna_grid_gains[grid_cell.grid_id] = grid_results
                    
                    if (i + 1) % 20 == 0:
                        print(f"    已完成: {i+1}/{len(self.grid_cells)}个网格")
                
                except Exception as e:
                    print(f"    网格{grid_cell.grid_id}分析失败: {str(e)}")
                    self.antenna_grid_gains[grid_cell.grid_id] = {}
        
        analysis_time = time.time() - start_time
        self.computation_stats['channel_computation_time'] = analysis_time
        
        print(f"  天线-网格增益分析完成, 耗时: {analysis_time:.2f}秒")
        print(f"  成功分析: {len(self.antenna_grid_gains)}个网格")
    
    def _analyze_single_grid(self, grid_cell: GridCell) -> Dict[int, List[AntennaGainResult]]:
        """分析单个网格的天线增益"""
        base_station_pos = np.array(self.params.base_station_pos)
        grid_center = grid_cell.center_position
        
        # 计算网格中心与基站的连线方向
        connection_vector = grid_center - base_station_pos
        connection_vector = connection_vector / np.linalg.norm(connection_vector)
        
        # 获取面向网格的半球天线位置
        hemisphere_positions = self._get_hemisphere_antenna_positions(
            connection_vector, self.sampling_config['hemisphere_samples']
        )
        
        grid_results = {}
        
        # 第一轮：分析半球内的50个径向位置
        first_round_results = []
        for pos_idx, antenna_pos in hemisphere_positions:
            # 径向法向量
            radial_normal = (antenna_pos - base_station_pos) / np.linalg.norm(antenna_pos - base_station_pos)
            
            gain_result = self._calculate_antenna_grid_gain(
                antenna_pos, radial_normal, grid_cell.user_positions, 'radial'
            )
            gain_result.antenna_position_idx = pos_idx
            first_round_results.append(gain_result)
        
        # 选择前5个强增益位置
        first_round_results.sort(key=lambda x: x.average_gain, reverse=True)
        top_positions = first_round_results[:5]
        
        # 第二轮：分析邻居位置
        neighbor_results = []
        processed_positions = set()
        
        for top_result in top_positions:
            pos_idx = top_result.antenna_position_idx
            # 传入连线向量，只获取半球内的邻居
            neighbor_indices = self._get_neighbor_position_indices(pos_idx, connection_vector)
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx not in processed_positions and neighbor_idx < len(self.action_space_manager.all_positions):
                    neighbor_pos = self.action_space_manager.all_positions[neighbor_idx]
                    radial_normal = (neighbor_pos - base_station_pos) / np.linalg.norm(neighbor_pos - base_station_pos)
                    
                    gain_result = self._calculate_antenna_grid_gain(
                        neighbor_pos, radial_normal, grid_cell.user_positions, 'radial_neighbor'
                    )
                    gain_result.antenna_position_idx = neighbor_idx
                    neighbor_results.append(gain_result)
                    processed_positions.add(neighbor_idx)
        
        # 合并第一轮和第二轮结果
        combined_results = first_round_results + neighbor_results
        combined_results.sort(key=lambda x: x.average_gain, reverse=True)
        
        # 选择最佳位置进行8种旋转分析
        best_positions = combined_results[:min(10, len(combined_results))]
        
        # 第三轮：8种旋转分析
        final_results = []
        for best_result in best_positions:
            pos_idx = best_result.antenna_position_idx
            antenna_pos = self.action_space_manager.all_positions[pos_idx]
            
            # 获取该位置的9种旋转（包括径向）
            rotations = self._get_position_rotations(pos_idx, antenna_pos)
            
            for rotation_matrix, normal, rotation_type in rotations:
                gain_result = self._calculate_antenna_grid_gain(
                    antenna_pos, normal, grid_cell.user_positions, rotation_type
                )
                gain_result.antenna_position_idx = pos_idx
                final_results.append(gain_result)
        
        # 按位置索引组织结果
        for result in final_results:
            pos_idx = result.antenna_position_idx
            if pos_idx not in grid_results:
                grid_results[pos_idx] = []
            grid_results[pos_idx].append(result)
        
        return grid_results
    
    def _get_hemisphere_antenna_positions(self, connection_vector: np.ndarray, 
                                        num_samples: int) -> List[Tuple[int, np.ndarray]]:
        """获取面向网格的半球内的天线位置"""
        base_station_pos = np.array(self.params.base_station_pos)
        all_positions = self.action_space_manager.all_positions
        
        hemisphere_positions = []
        
        for pos_idx, antenna_pos in enumerate(all_positions):
            # 计算天线位置相对于基站的方向
            antenna_direction = (antenna_pos - base_station_pos) / np.linalg.norm(antenna_pos - base_station_pos)
            
            # 检查是否在面向网格的半球内（点积 > 0）
            dot_product = np.dot(antenna_direction, connection_vector)
            if dot_product > 0:
                hemisphere_positions.append((pos_idx, antenna_pos))
        
        # 如果半球内位置太多，进行均匀采样
        if len(hemisphere_positions) > num_samples:
            # 使用球面均匀采样
            sampled_positions = self._uniform_sphere_sampling(hemisphere_positions, num_samples)
            return sampled_positions
        else:
            return hemisphere_positions
    
    def _uniform_sphere_sampling(self, positions: List[Tuple[int, np.ndarray]], 
                                num_samples: int) -> List[Tuple[int, np.ndarray]]:
        """在球面上进行均匀采样"""
        if len(positions) <= num_samples:
            return positions
        
        # 提取位置坐标
        coords = np.array([pos[1] for pos in positions])
        base_station_pos = np.array(self.params.base_station_pos)
        
        # 转换为球坐标
        relative_coords = coords - base_station_pos
        distances = np.linalg.norm(relative_coords, axis=1)
        
        # 归一化到单位球面
        unit_coords = relative_coords / distances[:, np.newaxis]
        
        # 使用k-means聚类进行均匀采样
        if SKLEARN_AVAILABLE:
            try:
                kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
                kmeans.fit(unit_coords)
                
                # 为每个聚类中心找到最近的实际位置
                sampled_positions = []
                for center in kmeans.cluster_centers_:
                    distances_to_center = np.linalg.norm(unit_coords - center, axis=1)
                    closest_idx = np.argmin(distances_to_center)
                    sampled_positions.append(positions[closest_idx])
                
                return sampled_positions
            except Exception as e:
                print(f"K-means聚类失败: {e}，使用简单采样")
        
        # 如果没有sklearn或聚类失败，使用简单的等间距采样
        step = len(positions) // num_samples
        return positions[::step][:num_samples]
    
    def _get_neighbor_position_indices(self, position_idx: int, connection_vector: np.ndarray = None) -> List[int]:
        """获取位置的邻居索引，可选择性过滤半球外的邻居"""
        if position_idx not in self.action_space_manager.neighbors:
            return []
        
        neighbors = self.action_space_manager.neighbors[position_idx]
        neighbor_indices = []
        base_station_pos = np.array(self.params.base_station_pos)
        
        for direction, neighbor_idx in neighbors.items():
            if neighbor_idx is not None and neighbor_idx != -1:
                # 如果提供了连线向量，检查邻居是否在对应半球内
                if connection_vector is not None:
                    neighbor_pos = self.action_space_manager.all_positions[neighbor_idx]
                    neighbor_direction = (neighbor_pos - base_station_pos) / np.linalg.norm(neighbor_pos - base_station_pos)
                    
                    # 只有面向网格的半球内的邻居才被包含
                    if np.dot(neighbor_direction, connection_vector) > 0:
                        neighbor_indices.append(neighbor_idx)
                else:
                    # 如果没有提供连线向量，包含所有邻居
                    neighbor_indices.append(neighbor_idx)
        
        return neighbor_indices
    
    def _get_position_rotations(self, pos_idx: int, position: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """获取位置的9种旋转"""
        return self.action_space_manager._generate_9_rotations(pos_idx, position)
    
    def _calculate_antenna_grid_gain(self, antenna_pos: np.ndarray, antenna_normal: np.ndarray,
                                   user_positions: List[np.ndarray], rotation_type: str) -> AntennaGainResult:
        """计算4天线阵列对网格内用户的理论极限速率"""
        if len(user_positions) == 0:
            return AntennaGainResult(
                antenna_position_idx=-1,
                antenna_position=antenna_pos,
                antenna_normal=antenna_normal,
                rotation_type=rotation_type,
                average_gain=0.0,
                max_gain=0.0,
                min_gain=0.0,
                gain_variance=0.0,
                user_gains=[]
            )
        
        # 生成4天线矩形阵列位置
        antenna_array_positions = self._generate_4_antenna_array(antenna_pos, antenna_normal)
        
        # 构建4天线阵列的信道矩阵
        num_users = len(user_positions)
        num_antennas = 4
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        # 创建用户对象列表
        users = []
        for user_pos in user_positions:
            user = User(
                id=len(users),
                type='vehicle' if user_pos[2] < 10 else 'UAV',
                position=user_pos,
                height=user_pos[2]
            )
            users.append(user)
        
        # 计算4天线阵列的信道矩阵
        for ant_idx, ant_pos in enumerate(antenna_array_positions):
            # 创建天线对象
            antenna = Antenna(
                surface_id=0,
                global_id=ant_idx,
                local_id=ant_idx,
                position=ant_pos,
                normal=antenna_normal,  # 阵列内所有天线使用相同法向量
                surface_center=antenna_pos  # 阵列中心
            )
            
            for user_idx, user in enumerate(users):
                # 计算到表面中心的距离（用于路径损耗）
                # 根据你的要求：表面内不同天线考虑相同的路径损耗
                distance = np.linalg.norm(user.position - antenna_pos)  # 使用阵列中心距离
                
                # 计算天线增益
                antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                    antenna, user, self.params
                )
                
                # 计算信道系数
                if user.type == 'vehicle':
                    channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                        distance, antenna_gain_linear, antenna, user, self.params
                    )
                else:
                    channel_coeff = ChannelModel.uav_channel_model_v2(
                        distance, antenna_gain_linear, user, self.params
                    )
                
                H[ant_idx, user_idx] = channel_coeff
        
        # 使用environment中的速率计算函数计算理论极限速率
        user_rates = self._calculate_theoretical_rates_vectorized(H)
        
        # 计算统计指标（现在是速率而不是增益）
        user_gains = user_rates  # 重命名为rates更合适，但保持兼容性
        
        return AntennaGainResult(
            antenna_position_idx=-1,
            antenna_position=antenna_pos,
            antenna_normal=antenna_normal,
            rotation_type=rotation_type,
            average_gain=np.mean(user_gains),
            max_gain=np.max(user_gains),
            min_gain=np.min(user_gains),
            gain_variance=np.var(user_gains),
            user_gains=user_gains.tolist()
        )
    
    def _generate_4_antenna_array(self, center_pos: np.ndarray, normal: np.ndarray) -> List[np.ndarray]:
        """生成4天线矩形阵列位置（2x2配置）"""
        spacing = self.params.antenna_spacing
        
        # 构建局部坐标系
        if abs(normal[2]) < 0.9:
            ref_vec = np.array([0, 0, 1])
        else:
            ref_vec = np.array([1, 0, 0])
        
        # 计算局部坐标系的两个切向量
        u = np.cross(normal, ref_vec)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # 2x2阵列的本地位置（以中心为原点）
        local_positions = np.array([
            [-spacing/2, -spacing/2, 0],  # 天线0: 左下
            [spacing/2, -spacing/2, 0],   # 天线1: 右下  
            [-spacing/2, spacing/2, 0],   # 天线2: 左上
            [spacing/2, spacing/2, 0]     # 天线3: 右上
        ])
        
        # 转换到全局坐标系
        antenna_positions = []
        for local_pos in local_positions:
            global_offset = local_pos[0] * u + local_pos[1] * v + local_pos[2] * normal
            global_pos = center_pos + global_offset
            antenna_positions.append(global_pos)
        
        return antenna_positions
    
    def _calculate_theoretical_rates_vectorized(self, H: np.ndarray) -> np.ndarray:
        """向量化计算理论速率（从environment中复制）"""
        # 系统参数
        noise_power_dBm = -174
        bandwidth_MHz = 20
        noise_figure_dB = 7
        transmit_power_dBm = 23  # 3GPP标准: 车辆和UAV上行链路功率
        
        # 噪声功率
        total_noise_dBm = noise_power_dBm + 10 * np.log10(bandwidth_MHz * 1e6) + noise_figure_dB
        noise_power_W = 10 ** ((total_noise_dBm - 30) / 10)
        
        # 发射功率
        transmit_power_W = 10 ** ((transmit_power_dBm - 30) / 10)
        
        num_users = H.shape[1]
        
        # 向量化计算信号功率
        signal_powers = transmit_power_W * np.abs(np.einsum('au,au->u', H.conj(), H))
        
        # 向量化计算干扰功率
        H_conj_H = np.abs(H.conj().T @ H) ** 2  # [num_users, num_users]
        H_norm_squared = np.abs(np.einsum('au,au->u', H.conj(), H))  # [num_users]
        
        interference_powers = np.zeros(num_users)
        for k in range(num_users):
            interference_mask = np.arange(num_users) != k
            interference_powers[k] = transmit_power_W * np.sum(
                H_conj_H[k, interference_mask] / (H_norm_squared[interference_mask] + 1e-10)
            )
        
        # SINR计算
        sinr = signal_powers / (interference_powers + noise_power_W)
        sinr = np.clip(sinr, 1e-6, 1e6)
        
        # 香农容量
        rates = np.log2(1 + sinr) * bandwidth_MHz
        
        return rates
    
    def _analyze_optimization_results(self) -> Dict:
        """分析优化结果"""
        print(f"  分析优化结果...")
        
        analysis_results = {
            'summary': {
                'total_grids': len(self.grid_cells),
                'analyzed_grids': len(self.antenna_grid_gains),
                'total_antenna_positions': len(self.action_space_manager.all_positions),
                'computation_stats': self.computation_stats.copy()
            },
            'grid_analysis': {},
            'antenna_ranking': {},
            'deployment_recommendations': {}
        }
        
        # 分析每个网格的最佳天线配置
        for grid_id, grid_results in self.antenna_grid_gains.items():
            if not grid_results:
                continue
            
            grid_cell = self.grid_cells[grid_id]
            
            # 找到该网格的最佳天线配置
            all_results = []
            for pos_idx, results_list in grid_results.items():
                all_results.extend(results_list)
            
            if all_results:
                # 按平均速率排序
                all_results.sort(key=lambda x: x.average_gain, reverse=True)
                best_config = all_results[0]
                
                analysis_results['grid_analysis'][grid_id] = {
                    'grid_type': grid_cell.grid_type,
                    'grid_center': grid_cell.center_position.tolist(),
                    'best_antenna_position_idx': best_config.antenna_position_idx,
                    'best_antenna_position': best_config.antenna_position.tolist(),
                    'best_antenna_normal': best_config.antenna_normal.tolist(),
                    'best_rotation_type': best_config.rotation_type,
                    'best_average_rate_mbps': best_config.average_gain,  # 现在是平均速率(Mbps)
                    'best_max_rate_mbps': best_config.max_gain,          # 现在是最大速率(Mbps)
                    'rate_improvement_ratio': best_config.max_gain / (best_config.min_gain + 1e-10),
                    'top_10_configs': [
                        {
                            'position_idx': result.antenna_position_idx,
                            'position': result.antenna_position.tolist(),
                            'normal': result.antenna_normal.tolist(),
                            'rotation_type': result.rotation_type,
                            'average_rate_mbps': result.average_gain,    # 现在是平均速率(Mbps)
                            'max_rate_mbps': result.max_gain,            # 现在是最大速率(Mbps)
                            'min_rate_mbps': result.min_gain,            # 现在是最小速率(Mbps)
                            'rate_variance': result.gain_variance,       # 现在是速率方差
                            'user_rates_mbps': result.user_gains         # 现在是每个用户的速率
                        }
                        for result in all_results[:10]
                    ]
                }
        
        # 全局天线位置排名
        position_scores = {}
        for grid_id, grid_results in self.antenna_grid_gains.items():
            for pos_idx, results_list in grid_results.items():
                if pos_idx not in position_scores:
                    position_scores[pos_idx] = []
                
                for result in results_list:
                    position_scores[pos_idx].append(result.average_gain)
        
        # 计算每个位置的综合得分
        position_rankings = []
        for pos_idx, rates in position_scores.items():
            avg_rate = np.mean(rates)
            max_rate = np.max(rates)
            coverage_count = len(rates)  # 该位置能有效服务的网格数量
            
            # 综合得分：平均速率 × 覆盖网格数 + 最大速率
            composite_score = avg_rate * coverage_count + max_rate * 0.1
            
            position_rankings.append({
                'position_idx': pos_idx,
                'position': self.action_space_manager.all_positions[pos_idx].tolist(),
                'average_rate_mbps': avg_rate,
                'max_rate_mbps': max_rate,
                'coverage_count': coverage_count,
                'composite_score': composite_score
            })
        
        # 按综合得分排序
        position_rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        analysis_results['antenna_ranking'] = position_rankings[:20]  # 前20个位置
        
        # 简化：移除复杂的部署建议，专注于网格-天线增益映射
        print(f"  专注于网格-天线增益映射，跳过部署建议生成")
        
        print(f"  结果分析完成:")
        print(f"    成功分析网格: {len(analysis_results['grid_analysis'])}")
        print(f"    天线位置排名: {len(position_rankings)}")
        print(f"    最佳位置综合得分: {position_rankings[0]['composite_score']:.3f}")
        
        return analysis_results
    
    def analyze_top_configs_statistics(self, analysis_results: Dict) -> Dict:
        """分析前10配置的统计信息"""
        print(f"  分析前10配置统计信息...")
        
        grid_analysis = analysis_results.get('grid_analysis', {})
        
        # 统计信息
        stats = {
            'total_grids_analyzed': len(grid_analysis),
            'position_frequency': {},     # 位置出现频次
            'rotation_frequency': {},     # 旋转类型频次
            'rate_distribution': [],      # 速率分布
            'top_positions_summary': [],  # 最频繁的位置汇总
            'rotation_effectiveness': {}  # 各旋转类型的效果
        }
        
        # 收集所有配置数据
        all_configs = []
        for grid_id, grid_info in grid_analysis.items():
            top_configs = grid_info.get('top_10_configs', [])
            for rank, config in enumerate(top_configs):
                config_data = {
                    'grid_id': grid_id,
                    'rank': rank + 1,
                    'position_idx': config['position_idx'],
                    'rotation_type': config['rotation_type'],
                    'average_rate': config['average_rate_mbps'],  # 现在是速率
                    'max_rate': config['max_rate_mbps']           # 现在是速率
                }
                all_configs.append(config_data)
        
        # 统计位置频次
        for config in all_configs:
            pos_idx = config['position_idx']
            if pos_idx not in stats['position_frequency']:
                stats['position_frequency'][pos_idx] = {
                    'count': 0,
                    'total_rate': 0,
                    'avg_rank': 0,
                    'grids_served': set()
                }
            
            stats['position_frequency'][pos_idx]['count'] += 1
            stats['position_frequency'][pos_idx]['total_rate'] += config['average_rate']
            stats['position_frequency'][pos_idx]['avg_rank'] += config['rank']
            stats['position_frequency'][pos_idx]['grids_served'].add(config['grid_id'])
        
        # 计算平均值
        for pos_idx, pos_data in stats['position_frequency'].items():
            count = pos_data['count']
            pos_data['avg_rate'] = pos_data['total_rate'] / count
            pos_data['avg_rank'] = pos_data['avg_rank'] / count
            pos_data['grids_served'] = len(pos_data['grids_served'])
        
        # 统计旋转类型频次
        for config in all_configs:
            rot_type = config['rotation_type']
            if rot_type not in stats['rotation_frequency']:
                stats['rotation_frequency'][rot_type] = {
                    'count': 0,
                    'total_rate': 0,
                    'avg_rank': 0
                }
            
            stats['rotation_frequency'][rot_type]['count'] += 1
            stats['rotation_frequency'][rot_type]['total_rate'] += config['average_rate']
            stats['rotation_frequency'][rot_type]['avg_rank'] += config['rank']
        
        # 计算旋转类型平均值
        for rot_type, rot_data in stats['rotation_frequency'].items():
            count = rot_data['count']
            rot_data['avg_rate'] = rot_data['total_rate'] / count
            rot_data['avg_rank'] = rot_data['avg_rank'] / count
        
        # 速率分布
        stats['rate_distribution'] = [config['average_rate'] for config in all_configs]
        
        # 找出最频繁的前20个位置
        top_positions = sorted(
            stats['position_frequency'].items(),
            key=lambda x: (x[1]['count'], x[1]['avg_rate']),
            reverse=True
        )[:20]
        
        stats['top_positions_summary'] = [
            {
                'position_idx': pos_idx,
                'position': self.action_space_manager.all_positions[pos_idx].tolist(),
                'frequency': pos_data['count'],
                'avg_rate_mbps': pos_data['avg_rate'],
                'avg_rank': pos_data['avg_rank'],
                'grids_served': pos_data['grids_served']
            }
            for pos_idx, pos_data in top_positions
        ]
        
        # 旋转类型效果排序
        rotation_ranking = sorted(
            stats['rotation_frequency'].items(),
            key=lambda x: x[1]['avg_rate'],
            reverse=True
        )
        
        stats['rotation_effectiveness'] = [
            {
                'rotation_type': rot_type,
                'frequency': rot_data['count'],
                'avg_rate_mbps': rot_data['avg_rate'],
                'avg_rank': rot_data['avg_rank']
            }
            for rot_type, rot_data in rotation_ranking
        ]
        
        return stats
    
    def _print_config_statistics_summary(self, config_stats: Dict):
        """打印前10配置统计摘要"""
        if not config_stats:
            print("  无统计数据可显示")
            return
        
        print(f"  📊 前10配置统计摘要:")
        print(f"    总分析网格数: {config_stats['total_grids_analyzed']}")
        print(f"    总配置数: {len(config_stats.get('rate_distribution', []))}")
        
        # 显示最频繁的前5个位置
        top_positions = config_stats.get('top_positions_summary', [])[:5]
        if top_positions:
            print(f"\n  🏆 最频繁的前5个天线位置:")
            for i, pos in enumerate(top_positions):
                print(f"    {i+1}. 位置{pos['position_idx']}: "
                      f"出现{pos['frequency']}次, "
                      f"平均速率{pos['avg_rate_mbps']:.2f}Mbps, "
                      f"服务{pos['grids_served']}个网格")
        
        # 显示旋转类型效果
        rotation_eff = config_stats.get('rotation_effectiveness', [])[:5]
        if rotation_eff:
            print(f"\n  🔄 最有效的前5种旋转类型:")
            for i, rot in enumerate(rotation_eff):
                print(f"    {i+1}. {rot['rotation_type']}: "
                      f"平均速率{rot['avg_rate_mbps']:.2f}Mbps, "
                      f"出现{rot['frequency']}次, "
                      f"平均排名{rot['avg_rank']:.1f}")
        
        # 速率分布统计
        rate_dist = config_stats.get('rate_distribution', [])
        if rate_dist:
            import numpy as np
            rates = np.array(rate_dist)
            print(f"\n  📈 速率分布统计:")
            print(f"    最大速率: {np.max(rates):.2f} Mbps")
            print(f"    最小速率: {np.min(rates):.2f} Mbps")
            print(f"    平均速率: {np.mean(rates):.2f} Mbps")
            print(f"    中位数速率: {np.median(rates):.2f} Mbps")
            print(f"    标准差: {np.std(rates):.2f} Mbps")
    
    def _save_results(self, analysis_results: Dict, output_dir: str):
        """保存优化结果"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式的分析结果
        with open(f"{output_dir}/optimization_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # 保存pickle格式的完整数据
        with open(f"{output_dir}/complete_optimization_data.pkl", 'wb') as f:
            pickle.dump({
                'grid_cells': self.grid_cells,
                'antenna_grid_gains': self.antenna_grid_gains,
                'analysis_results': analysis_results,
                'computation_stats': self.computation_stats
            }, f)
        
        # 保存核心的网格-天线速率映射摘要
        grid_antenna_summary = {
            'total_grids': len(analysis_results.get('grid_analysis', {})),
            'top_antenna_positions': analysis_results.get('antenna_ranking', [])[:10],
            'config_statistics_summary': {
                'total_configs': len(analysis_results.get('config_statistics', {}).get('rate_distribution', [])),
                'top_positions': analysis_results.get('config_statistics', {}).get('top_positions_summary', [])[:10],
                'rotation_effectiveness': analysis_results.get('config_statistics', {}).get('rotation_effectiveness', [])[:5]
            }
        }
        
        with open(f"{output_dir}/grid_antenna_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(grid_antenna_summary, f, indent=2, ensure_ascii=False)
        
        print(f"  结果已保存至: {output_dir}/")
        print(f"    - optimization_analysis.json: 完整的网格-天线速率分析")
        print(f"    - grid_antenna_mapping.json: 网格-天线映射摘要")
        print(f"    - complete_optimization_data.pkl: 完整数据(可重新加载)")
    
    def _generate_visualizations(self, analysis_results: Dict, output_dir: str):
        """生成可视化结果"""
        try:
            # 1. 网格和最优天线位置3D可视化
            self._plot_3d_optimization_results(analysis_results, output_dir)
            
            # 2. 天线位置排名图
            self._plot_antenna_ranking(analysis_results, output_dir)
            
            # 3. 覆盖分析热图
            self._plot_coverage_heatmap(analysis_results, output_dir)
            
            print(f"  可视化结果已生成至: {output_dir}/")
        
        except Exception as e:
            print(f"  可视化生成失败: {str(e)}")
    
    def _plot_3d_optimization_results(self, analysis_results: Dict, output_dir: str):
        """绘制3D优化结果"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制基站
        base_pos = self.params.base_station_pos
        ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], 
                  c='red', s=200, marker='^', label='Base Station')
        
        # 绘制网格中心点
        ground_grids = [g for g in self.grid_cells if g.grid_type == 'ground']
        air_grids = [g for g in self.grid_cells if g.grid_type == 'air']
        
        if ground_grids:
            ground_centers = np.array([g.center_position for g in ground_grids])
            ax.scatter(ground_centers[:, 0], ground_centers[:, 1], ground_centers[:, 2],
                      c='blue', s=20, alpha=0.6, label='Ground Grids')
        
        if air_grids:
            air_centers = np.array([g.center_position for g in air_grids])
            ax.scatter(air_centers[:, 0], air_centers[:, 1], air_centers[:, 2],
                      c='cyan', s=20, alpha=0.6, label='Air Grids')
        
        # 绘制最频繁的天线位置（来自统计分析）
        config_stats = analysis_results.get('config_statistics', {})
        top_positions = config_stats.get('top_positions_summary', [])[:10]
        if top_positions:
            top_pos = np.array([pos['position'] for pos in top_positions])
            ax.scatter(top_pos[:, 0], top_pos[:, 1], top_pos[:, 2],
                      c='gold', s=100, marker='*', label='Most Frequent Antennas')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Grid-Based Antenna Optimization Results')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3d_optimization_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_antenna_ranking(self, analysis_results: Dict, output_dir: str):
        """绘制天线位置排名"""
        rankings = analysis_results['antenna_ranking'][:15]  # 前15个
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 综合得分排名
        scores = [r['composite_score'] for r in rankings]
        positions = [f"Pos {r['position_idx']}" for r in rankings]
        
        ax1.barh(range(len(scores)), scores)
        ax1.set_yticks(range(len(scores)))
        ax1.set_yticklabels(positions)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Top 15 Antenna Positions by Composite Score')
        ax1.invert_yaxis()
        
        # 覆盖网格数量
        coverage_counts = [r['coverage_count'] for r in rankings]
        
        ax2.barh(range(len(coverage_counts)), coverage_counts, color='orange')
        ax2.set_yticks(range(len(coverage_counts)))
        ax2.set_yticklabels(positions)
        ax2.set_xlabel('Coverage Count (Grids)')
        ax2.set_title('Coverage Count by Antenna Position')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/antenna_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coverage_heatmap(self, analysis_results: Dict, output_dir: str):
        """绘制覆盖分析热图"""
        grid_analysis = analysis_results['grid_analysis']
        
        # 提取地面网格和空中网格的增益数据
        ground_gains = []
        air_gains = []
        
        for grid_id, grid_info in grid_analysis.items():
            if grid_info['grid_type'] == 'ground':
                ground_gains.append(grid_info['best_average_gain'])
            else:
                air_gains.append(grid_info['best_average_gain'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 地面网格热图
        if ground_gains:
            ground_gains_2d = np.array(ground_gains).reshape(10, 10)
            im1 = ax1.imshow(ground_gains_2d, cmap='viridis', aspect='auto')
            ax1.set_title('Ground Grids - Best Average Gain')
            ax1.set_xlabel('Grid X Index')
            ax1.set_ylabel('Grid Y Index')
            plt.colorbar(im1, ax=ax1, label='Average Gain')
        
        # 空中网格热图
        if air_gains:
            air_gains_2d = np.array(air_gains).reshape(10, 10)
            im2 = ax2.imshow(air_gains_2d, cmap='plasma', aspect='auto')
            ax2.set_title('Air Grids - Best Average Gain')
            ax2.set_xlabel('Grid X Index')
            ax2.set_ylabel('Grid Y Index')
            plt.colorbar(im2, ax=ax2, label='Average Gain')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/coverage_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_optimization_results(self, result_file: str) -> Dict:
        """加载已保存的优化结果"""
        try:
            if result_file.endswith('.pkl'):
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
                    self.grid_cells = data['grid_cells']
                    self.antenna_grid_gains = data['antenna_grid_gains']
                    self.computation_stats = data['computation_stats']
                    return data['analysis_results']
            
            elif result_file.endswith('.json'):
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            else:
                raise ValueError("Unsupported file format. Use .pkl or .json")
        
        except Exception as e:
            print(f"加载优化结果失败: {str(e)}")
            return {}
    
    def get_deployment_strategy_for_user_distribution(self, user_positions: List[np.ndarray]) -> Dict:
        """根据实际用户分布获取部署策略"""
        if not hasattr(self, 'antenna_grid_gains') or not self.antenna_grid_gains:
            raise ValueError("请先运行优化分析或加载已有结果")
        
        # 将用户分配到网格
        user_grid_mapping = self._assign_users_to_grids(user_positions)
        
        # 根据用户分布计算网格权重
        grid_weights = {}
        for grid_id in range(len(self.grid_cells)):
            user_count = len(user_grid_mapping.get(grid_id, []))
            grid_weights[grid_id] = user_count
        
        # 基于权重选择天线位置
        weighted_strategy = self._calculate_weighted_deployment_strategy(grid_weights)
        
        return {
            'user_grid_mapping': user_grid_mapping,
            'grid_weights': grid_weights,
            'weighted_deployment_strategy': weighted_strategy,
            'total_users': len(user_positions)
        }
    
    def _assign_users_to_grids(self, user_positions: List[np.ndarray]) -> Dict[int, List[int]]:
        """将用户分配到网格"""
        user_grid_mapping = {}
        
        for user_idx, user_pos in enumerate(user_positions):
            # 找到用户所属的网格
            assigned_grid = None
            
            for grid_cell in self.grid_cells:
                bounds = grid_cell.bounds
                if (bounds['x'][0] <= user_pos[0] <= bounds['x'][1] and
                    bounds['y'][0] <= user_pos[1] <= bounds['y'][1] and
                    bounds['z'][0] <= user_pos[2] <= bounds['z'][1]):
                    assigned_grid = grid_cell.grid_id
                    break
            
            if assigned_grid is not None:
                if assigned_grid not in user_grid_mapping:
                    user_grid_mapping[assigned_grid] = []
                user_grid_mapping[assigned_grid].append(user_idx)
        
        return user_grid_mapping
    
    def _calculate_weighted_deployment_strategy(self, grid_weights: Dict[int, int]) -> Dict:
        """计算基于权重的部署策略"""
        if not hasattr(self, 'antenna_grid_gains'):
            return {}
        
        # 计算每个天线位置的加权得分
        position_weighted_scores = {}
        
        # 先收集所有增益值来计算归一化因子
        all_gains = []
        for grid_id, user_count in grid_weights.items():
            if user_count == 0 or grid_id not in self.antenna_grid_gains:
                continue
            
            grid_results = self.antenna_grid_gains[grid_id]
            for pos_idx, results_list in grid_results.items():
                best_gain = max([r.average_gain for r in results_list])
                all_gains.append(best_gain)
        
        # 计算增益的归一化因子
        if all_gains:
            max_gain = max(all_gains)
            min_gain = min(all_gains)
            gain_range = max_gain - min_gain if max_gain > min_gain else 1.0
        else:
            max_gain, min_gain, gain_range = 1.0, 0.0, 1.0
        
        # 计算加权得分
        for grid_id, user_count in grid_weights.items():
            if user_count == 0 or grid_id not in self.antenna_grid_gains:
                continue
            
            grid_results = self.antenna_grid_gains[grid_id]
            
            for pos_idx, results_list in grid_results.items():
                if pos_idx not in position_weighted_scores:
                    position_weighted_scores[pos_idx] = 0
                
                # 使用归一化的增益乘以用户数量作为权重
                best_gain = max([r.average_gain for r in results_list])
                normalized_gain = (best_gain - min_gain) / gain_range
                position_weighted_scores[pos_idx] += normalized_gain * user_count
        
        # 排序并选择最佳位置
        sorted_positions = sorted(
            position_weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'weighted_ranking': [
                {
                    'position_idx': pos_idx,
                    'weighted_score': score,
                    'position': self.action_space_manager.all_positions[pos_idx].tolist()
                }
                for pos_idx, score in sorted_positions[:20]
            ],
            'recommended_deployment': sorted_positions[:self.params.num_surfaces]
        }


def main():
    """主函数 - 演示网格天线优化系统的使用"""
    print("Grid-Based Antenna Optimizer Demo")
    print("="*50)
    
    # 初始化系统参数
    params = SystemParams()
    
    # 创建优化器
    optimizer = GridBasedAntennaOptimizer(
        params=params,
        enable_parallel=True,
        cache_results=True
    )
    
    # 运行完整优化
    results = optimizer.run_complete_optimization(
        output_dir="antenna_optimization_results"
    )
    
    if results:
        print("\n优化完成！主要结果:")
        print(f"- 分析网格数: {results['summary']['analyzed_grids']}")
        deployment_recommendations = results.get('deployment_recommendations', {})
        deployment_strategy = deployment_recommendations.get('deployment_strategy', {})
        coverage_analysis = deployment_recommendations.get('coverage_analysis', {})
        primary_positions = deployment_strategy.get('primary_positions', [])
        
        print(f"- 推荐天线位置数: {len(primary_positions)}")
        print(f"- 覆盖率: {coverage_analysis.get('coverage_ratio', 0):.1%}")
        
        # 展示前5个推荐位置
        print(f"\n前5个推荐天线位置:")
        for i, pos in enumerate(primary_positions[:5]):
            print(f"  {i+1}. 位置索引: {pos['position_idx']}, 覆盖网格: {pos.get('new_coverage', 0)}")


if __name__ == "__main__":
    main()
