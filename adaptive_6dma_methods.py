
import numpy as np
import os
import json
from typing import Dict, List, Tuple
from sixDMA_Environment_core_class import SystemParams, ChannelModel, Antenna, User, UserMobility
from sixdma_environment_optimized import OptimizedSixDMAEnvironment

# ============================================================================
# 🔧 配置参数
# ============================================================================

# 用户配置
NUM_GROUND_USERS = 30      # 地面车辆用户数量（基准值）
NUM_AIR_USERS = 0        # 空中无人机用户数量（固定）

# 测试配置
GROUND_USER_COUNTS = [30, 35, 40, 45, 50]  # 不同的车辆用户数量测试
POWER_RANGE_MW = [20, 40, 60, 80, 100, 120]  # 发射功率范围 (mW)
FIXED_USERS_FOR_POWER_TEST = (30, 0)  # 功率测试时固定的用户数量

# 场景配置
MAX_UPDATES = 25           # 动态场景更新次数
RANDOM_SEED = 42           # 随机种子
ENVIRONMENT_SIZE = (300, 300, 100)  # 环境尺寸
AIR_HEIGHT_RANGE = (50.0, 100.0)   # 空中用户高度范围

# 新方法特定配置
CIRCULAR_POSITIONS_M = 8      # 圆形路径上的离散位置数
CIRCULAR_RADIUS = 10.0        # 圆形路径半径(米)
DISCRETE_ROTATIONS_L = 4      # 每个角度维度的离散步数（水平和竖直各5个）
ROTATION_RANGE = 60.0         # 旋转范围：±60度（水平和竖直）

print(f"🚀 自适应6DMA方法测试")
print(f"🔧 配置参数:")
print(f"  圆形位置数: {CIRCULAR_POSITIONS_M}个")
print(f"  圆形半径: {CIRCULAR_RADIUS}米")
print(f"  位置选择策略: 随机选择4个不同位置")
print(f"  离散旋转数: {DISCRETE_ROTATIONS_L}×{DISCRETE_ROTATIONS_L}个 (水平×竖直)")
print(f"  旋转范围: ±{ROTATION_RANGE}° (水平和竖直)")
print(f"  旋转选择策略: 随机采样10种组合，选择最优")
print("=" * 80)


# 全局缓存的环境实例，避免重复初始化
_cached_env = None
_cached_params = None


def get_cached_environment(params: SystemParams):
    """获取缓存的环境实例，避免重复初始化ActionSpace"""
    global _cached_env, _cached_params
    
    # 检查参数是否变化，如果变化则重新创建
    if (_cached_env is None or 
        _cached_params is None or
        _cached_params.num_ground_users != params.num_ground_users or
        _cached_params.num_air_users != params.num_air_users):
        
        print("    🔧 创建新的环境实例...")
        _cached_env = OptimizedSixDMAEnvironment(params)
        _cached_params = params
    
    return _cached_env


def mw_to_dbm(power_mw):
    """将毫瓦转换为dBm"""
    return 10 * np.log10(power_mw)


def dbm_to_mw(power_dbm):
    """将dBm转换为毫瓦"""
    return 10 ** (power_dbm / 10)


class CircularPositionManager:
    """6DMA圆形离散位置管理器 - 4扇区沿圆形路径移动"""
    
    def __init__(self, params: SystemParams, transmit_power_dbm: float = 23.0):
        self.params = params
        self.transmit_power_dbm = transmit_power_dbm
        
        print("🔧 初始化6DMA圆形位置管理器...")
        
        # 基站基准位置
        self.base_station_pos = np.array(params.base_station_pos)
        
        # 生成圆形路径上的离散位置
        self.circular_positions = self._generate_circular_positions()
        print(f"  生成{len(self.circular_positions)}个圆形路径位置")
        
        # 初始化用户
        self.current_users = UserMobility.generate_user_positions(params, seed=RANDOM_SEED)
        print(f"  用户总数: {len(self.current_users)} (地面{params.num_ground_users}个, 空中{params.num_air_users}个)")
        
        # 统计数据
        self.stats = {
            'update_rates': [],
            'total_updates': 0,
            'position_history': [],  # 记录使用的位置
            'best_positions': []     # 记录每次更新的最佳位置
        }
        
        print(f"  ✅ 圆形位置6DMA配置完成: 4扇区×{CIRCULAR_POSITIONS_M}个位置")
    
    def _generate_circular_positions(self):
        """生成圆形路径上的M个离散位置"""
        positions = []
        
        for i in range(CIRCULAR_POSITIONS_M):
            angle = 2 * np.pi * i / CIRCULAR_POSITIONS_M
            # 在水平面上的圆形路径
            offset = np.array([
                CIRCULAR_RADIUS * np.cos(angle),
                CIRCULAR_RADIUS * np.sin(angle),
                0  # 保持相同高度
            ])
            positions.append(self.base_station_pos + offset)
        
        return positions
    
    def _generate_4sector_fpa_at_position(self, center_pos: np.ndarray):
        """在指定位置生成4扇区FPA配置"""
        sectors = []
        
        # 4个扇区的方位角 (北、东、南、西)
        sector_azimuths = [0, 90, 180, 270]  # 度
        sector_names = ['North', 'East', 'South', 'West']
        
        # 下倾角固定为15°
        downtilt_angle = 15.0  # 度
        
        # 每个扇区的4×4天线阵列参数
        array_spacing = 0.5 * self.params.lambda_wave  # 半波长间距
        
        for sector_idx, (azimuth, name) in enumerate(zip(sector_azimuths, sector_names)):
            sector_config = {
                'sector_id': sector_idx,
                'name': name,
                'azimuth': azimuth,
                'downtilt': downtilt_angle,
                'center_position': center_pos,
                'antennas': []
            }
            
            # 生成4×4天线阵列
            antennas = self._generate_4x4_antenna_array_at_position(
                center_pos, sector_idx, azimuth, downtilt_angle, array_spacing
            )
            
            sector_config['antennas'] = antennas
            sectors.append(sector_config)
        
        return sectors
    
    def _generate_4sector_fpa_at_position_combination(self, position_indices: List[int]):
        """在指定的4个位置组合上生成4扇区FPA配置
        
        Args:
            position_indices: 4个扇区对应的圆形位置索引 [北扇区位置, 东扇区位置, 南扇区位置, 西扇区位置]
        """
        sectors = []
        
        # 4个扇区的方位角 (北、东、南、西)
        sector_azimuths = [0, 90, 180, 270]  # 度
        sector_names = ['North', 'East', 'South', 'West']
        
        # 下倾角固定为15°
        downtilt_angle = 15.0  # 度
        
        # 每个扇区的4×4天线阵列参数
        array_spacing = 0.5 * self.params.lambda_wave  # 半波长间距
        
        for sector_idx, (azimuth, name) in enumerate(zip(sector_azimuths, sector_names)):
            # 获取该扇区的圆形位置
            sector_position = self.circular_positions[position_indices[sector_idx]]
            
            sector_config = {
                'sector_id': sector_idx,
                'name': name,
                'azimuth': azimuth,
                'downtilt': downtilt_angle,
                'center_position': sector_position,
                'position_index': position_indices[sector_idx],
                'antennas': []
            }
            
            # 生成4×4天线阵列
            antennas = self._generate_4x4_antenna_array_at_position(
                sector_position, sector_idx, azimuth, downtilt_angle, array_spacing
            )
            
            sector_config['antennas'] = antennas
            sectors.append(sector_config)
        
        return sectors
    
    def _generate_4x4_antenna_array_at_position(self, center_pos: np.ndarray, sector_id: int, 
                                               azimuth: float, downtilt: float, spacing: float):
        """在指定中心位置为单个扇区生成4×4天线阵列"""
        antennas = []
        
        # 转换角度为弧度
        azimuth_rad = np.radians(azimuth)
        downtilt_rad = np.radians(downtilt)
        
        # 计算局部坐标系
        u_vec = np.array([-np.cos(azimuth_rad), np.sin(azimuth_rad), 0])
        v_vec = np.array([0, 0, 1])  # 垂直向上
        
        # 天线法向量（指向扇区覆盖方向，考虑下倾）
        antenna_normal = np.array([
            np.cos(downtilt_rad) * np.sin(azimuth_rad),
            np.cos(downtilt_rad) * np.cos(azimuth_rad),
            -np.sin(downtilt_rad)
        ])
        
        # 生成4×4阵列位置
        for i in range(4):  # 行
            for j in range(4):  # 列
                u_offset = (j - 1.5) * spacing  # 列偏移
                v_offset = (i - 1.5) * spacing  # 行偏移
                
                antenna_pos = center_pos + u_offset * u_vec + v_offset * v_vec
                
                antenna = Antenna(
                    surface_id=sector_id,
                    global_id=sector_id * 16 + i * 4 + j,
                    local_id=i * 4 + j,
                    position=antenna_pos,
                    normal=antenna_normal,
                    surface_center=center_pos
                )
                
                antennas.append(antenna)
        
        return antennas
    
    def _find_random_position_combination_for_users(self):
        """随机选择4扇区位置组合
        
        从8个圆形位置中随机选择4个不同位置给4个扇区
        """
        # 设置随机种子确保可重现性
        np.random.seed(RANDOM_SEED + self.stats['total_updates'] + 1000)
        
        # 随机选择4个不同的位置索引
        selected_positions = np.random.choice(
            CIRCULAR_POSITIONS_M, 
            size=4, 
            replace=False  # 不重复选择
        ).tolist()
        
        print(f"    🎲 随机选择位置组合: {selected_positions}")
        
        # 生成该位置组合的4扇区配置
        sectors = self._generate_4sector_fpa_at_position_combination(selected_positions)
        
        # 计算该配置的系统速率
        total_rate = self._calculate_system_rate_for_sectors(sectors)
        
        # 显示选择的位置组合坐标
        positions_coords = [self.circular_positions[idx] for idx in selected_positions]
        print(f"    ✅ 随机位置组合详情:")
        print(f"      扇区0(北): 位置{selected_positions[0]} {positions_coords[0]}")
        print(f"      扇区1(东): 位置{selected_positions[1]} {positions_coords[1]}")
        print(f"      扇区2(南): 位置{selected_positions[2]} {positions_coords[2]}")
        print(f"      扇区3(西): 位置{selected_positions[3]} {positions_coords[3]}")
        print(f"      总速率: {total_rate:.2f} Mbps")
        
        return selected_positions, total_rate
    
    def _calculate_system_rate_for_sectors(self, sectors):
        """计算给定扇区配置的系统总速率"""
        if not self.current_users:
            return 0.0
        
        # 构建信道矩阵
        num_users = len(self.current_users)
        total_antennas = 4 * 16  # 4个扇区，每扇区16个天线
        
        H = np.zeros((total_antennas, num_users), dtype=complex)
        
        antenna_idx = 0
        for sector in sectors:
            for antenna in sector['antennas']:
                for user_idx, user in enumerate(self.current_users):
                    distance = np.linalg.norm(user.position - antenna.position)
                    antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                        antenna, user, self.params
                    )
                    
                    if user.type == 'vehicle':
                        channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                            distance, antenna_gain_linear, antenna, user, self.params
                        )
                    else:
                        channel_coeff = ChannelModel.uav_channel_model_v2(
                            distance, antenna_gain_linear, user, self.params
                        )
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 计算速率（使用全局缓存的环境实例）
        temp_env = get_cached_environment(self.params)
        user_rates = temp_env._calculate_theoretical_rates_vectorized(H, self.transmit_power_dbm)
        
        return np.sum(user_rates)
    
    def update_scenario(self, time_step: float):
        """更新圆形位置场景"""
        # 更新用户位置
        seed_for_update = RANDOM_SEED + self.stats['total_updates'] + 4000
        self.current_users = UserMobility.update_user_positions(self.current_users, time_step, random_seed=seed_for_update)
        
        # 随机选择位置组合
        selected_position_combination, total_rate = self._find_random_position_combination_for_users()
        
        # 记录结果
        self.stats['update_rates'].append(total_rate)
        self.stats['position_history'].append(selected_position_combination)
        self.stats['best_positions'].append(selected_position_combination)
        self.stats['total_updates'] += 1
        
        print(f"    圆形位置更新{self.stats['total_updates']}: 随机位置组合{selected_position_combination}, 总用户速率 {total_rate:.2f} Mbps")
        
        return total_rate
    
    def run_dynamic_scenario(self, max_updates: int):
        """运行动态圆形位置场景"""
        print(f"\n🔄 开始 {max_updates} 次圆形位置优化更新...")
        
        for update_count in range(max_updates):
            print(f"\n  --- 圆形位置更新 {update_count + 1}/{max_updates} ---")
            
            time_step = 1.0
            self.update_scenario(time_step)
        
        print(f"\n🔄 圆形位置方法完成 {max_updates} 次更新")


class DiscreteRotationManager:
    """6DMA离散旋转管理器 - 4扇区固定位置可旋转"""
    
    def __init__(self, params: SystemParams, transmit_power_dbm: float = 23.0):
        self.params = params
        self.transmit_power_dbm = transmit_power_dbm
        
        print("🔧 初始化6DMA离散旋转管理器...")
        
        # 基站位置（固定）
        self.base_station_pos = np.array(params.base_station_pos)
        
        # 生成离散旋转角度（水平和竖直）
        self.discrete_horizontal_angles, self.discrete_vertical_angles = self._generate_discrete_angles()
        print(f"  生成{len(self.discrete_horizontal_angles)}×{len(self.discrete_vertical_angles)}个离散旋转角度组合")
        
        # 初始化用户
        self.current_users = UserMobility.generate_user_positions(params, seed=RANDOM_SEED)
        print(f"  用户总数: {len(self.current_users)} (地面{params.num_ground_users}个, 空中{params.num_air_users}个)")
        
        # 统计数据
        self.stats = {
            'update_rates': [],
            'total_updates': 0,
            'rotation_history': [],    # 记录使用的旋转组合
            'best_rotations': []       # 记录每次更新的最佳旋转
        }
        
        print(f"  ✅ 离散旋转6DMA配置完成: 4扇区×{DISCRETE_ROTATIONS_L}×{DISCRETE_ROTATIONS_L}个角度组合")
    
    def _generate_discrete_angles(self):
        """生成水平和竖直的离散角度（±60度范围内）"""
        # 水平角度：在±60度范围内生成L个离散角度
        horizontal_angles = []
        for i in range(DISCRETE_ROTATIONS_L):
            angle = -ROTATION_RANGE + (2 * ROTATION_RANGE * i) / (DISCRETE_ROTATIONS_L - 1)
            horizontal_angles.append(angle)
        
        # 竖直角度：在±60度范围内生成L个离散角度
        vertical_angles = []
        for i in range(DISCRETE_ROTATIONS_L):
            angle = -ROTATION_RANGE + (2 * ROTATION_RANGE * i) / (DISCRETE_ROTATIONS_L - 1)
            vertical_angles.append(angle)
        
        return horizontal_angles, vertical_angles
    
    def _generate_4sector_fpa_with_rotations(self, rotation_indices: List[Tuple[int, int]]):
        """生成4扇区FPA配置，每个扇区使用指定的水平和竖直旋转角度
        
        Args:
            rotation_indices: 4个扇区的旋转索引，每个为(水平索引, 竖直索引)
        """
        sectors = []
        
        # 4个扇区的基准方位角和下倾角
        base_azimuths = [0, 90, 180, 270]  # 北、东、南、西
        sector_names = ['North', 'East', 'South', 'West']
        base_downtilt = 15.0  # 基准下倾角
        
        # 每个扇区的4×4天线阵列参数
        array_spacing = 0.5 * self.params.lambda_wave
        
        for sector_idx, (base_azimuth, name) in enumerate(zip(base_azimuths, sector_names)):
            h_idx, v_idx = rotation_indices[sector_idx]
            
            # 应用水平和竖直旋转偏移
            horizontal_offset = self.discrete_horizontal_angles[h_idx]
            vertical_offset = self.discrete_vertical_angles[v_idx]
            
            # 计算实际的方位角和下倾角
            actual_azimuth = (base_azimuth + horizontal_offset) % 360
            actual_downtilt = base_downtilt + vertical_offset
            
            # 限制下倾角范围（避免过度上倾或下倾）
            actual_downtilt = np.clip(actual_downtilt, -45.0, 75.0)
            
            sector_config = {
                'sector_id': sector_idx,
                'name': name,
                'base_azimuth': base_azimuth,
                'base_downtilt': base_downtilt,
                'horizontal_offset': horizontal_offset,
                'vertical_offset': vertical_offset,
                'actual_azimuth': actual_azimuth,
                'actual_downtilt': actual_downtilt,
                'antennas': []
            }
            
            # 生成4×4天线阵列
            antennas = self._generate_4x4_antenna_array_with_rotation(
                sector_idx, actual_azimuth, actual_downtilt, array_spacing
            )
            
            sector_config['antennas'] = antennas
            sectors.append(sector_config)
        
        return sectors
    
    def _generate_4x4_antenna_array_with_rotation(self, sector_id: int, azimuth: float, 
                                                 downtilt: float, spacing: float):
        """为单个扇区生成4×4天线阵列（考虑水平和竖直旋转）"""
        antennas = []
        
        # 转换角度为弧度
        azimuth_rad = np.radians(azimuth)
        downtilt_rad = np.radians(downtilt)
        
        # 计算局部坐标系（考虑旋转后的方位角）
        u_vec = np.array([-np.cos(azimuth_rad), np.sin(azimuth_rad), 0])
        v_vec = np.array([0, 0, 1])  # 垂直向上
        
        # 天线法向量（考虑调整后的方位角和下倾角）
        antenna_normal = np.array([
            np.cos(downtilt_rad) * np.sin(azimuth_rad),
            np.cos(downtilt_rad) * np.cos(azimuth_rad),
            -np.sin(downtilt_rad)  # 下倾为负z方向
        ])
        
        # 生成4×4阵列位置
        for i in range(4):  # 行
            for j in range(4):  # 列
                u_offset = (j - 1.5) * spacing  # 列偏移
                v_offset = (i - 1.5) * spacing  # 行偏移
                
                antenna_pos = self.base_station_pos + u_offset * u_vec + v_offset * v_vec
                
                antenna = Antenna(
                    surface_id=sector_id,
                    global_id=sector_id * 16 + i * 4 + j,
                    local_id=i * 4 + j,
                    position=antenna_pos,
                    normal=antenna_normal,
                    surface_center=self.base_station_pos
                )
                
                antennas.append(antenna)
        
        return antennas
    
    def _find_best_from_random_rotation_samples(self):
        """随机采样10种旋转组合，选择效果最好的一种
        
        策略：
        1. 随机生成10种不同的旋转角度组合
        2. 评估每种组合的系统速率
        3. 选择速率最高的组合
        """
        # 设置随机种子确保可重现性
        np.random.seed(RANDOM_SEED + self.stats['total_updates'] + 2000)
        
        num_samples = 10
        best_rotations = [(0, 0), (0, 0), (0, 0), (0, 0)]
        best_rate = 0.0
        all_samples = []
        
        print(f"    🎲 随机采样{num_samples}种旋转组合，选择最优...")
        
        # 生成并评估10种随机旋转组合
        for sample_idx in range(num_samples):
            # 为4个扇区随机选择旋转角度
            sample_rotations = []
            for sector_idx in range(4):
                h_idx = np.random.randint(0, DISCRETE_ROTATIONS_L)
                v_idx = np.random.randint(0, DISCRETE_ROTATIONS_L)
                sample_rotations.append((h_idx, v_idx))
            
            # 生成该旋转组合的扇区配置
            sectors = self._generate_4sector_fpa_with_rotations(sample_rotations)
            
            # 计算该配置的系统速率
            sample_rate = self._calculate_system_rate_for_sectors(sectors)
            
            # 记录样本
            all_samples.append({
                'rotations': sample_rotations,
                'rate': sample_rate,
                'angles': self._format_rotation_angles(sample_rotations)
            })
            
            # 更新最优配置
            if sample_rate > best_rate:
                best_rate = sample_rate
                best_rotations = sample_rotations.copy()
            
            print(f"      样本{sample_idx+1}: {self._format_rotation_angles(sample_rotations)} -> {sample_rate:.2f} Mbps")
        
        # 显示最优选择
        print(f"    ✅ 最优旋转组合 (从{num_samples}个样本中选择):")
        for sector_idx, (h_idx, v_idx) in enumerate(best_rotations):
            h_angle = self.discrete_horizontal_angles[h_idx]
            v_angle = self.discrete_vertical_angles[v_idx]
            sector_names = ['北', '东', '南', '西']
            print(f"      扇区{sector_idx}({sector_names[sector_idx]}): ({h_angle:+.1f}°, {v_angle:+.1f}°)")
        print(f"      最优速率: {best_rate:.2f} Mbps")
        
        # 显示采样统计
        sample_rates = [s['rate'] for s in all_samples]
        print(f"      采样统计: 平均{np.mean(sample_rates):.2f} ± {np.std(sample_rates):.2f} Mbps, 范围[{np.min(sample_rates):.2f}, {np.max(sample_rates):.2f}]")
        
        return best_rotations, best_rate
    
    def _format_rotation_angles(self, rotation_indices: List[Tuple[int, int]]):
        """格式化旋转角度显示"""
        angles = []
        for h_idx, v_idx in rotation_indices:
            h_angle = self.discrete_horizontal_angles[h_idx]
            v_angle = self.discrete_vertical_angles[v_idx]
            angles.append(f"({h_angle:+.1f}°,{v_angle:+.1f}°)")
        return angles
    
    def _calculate_system_rate_for_sectors(self, sectors):
        """计算给定扇区配置的系统总速率"""
        if not self.current_users:
            return 0.0
        
        # 构建信道矩阵
        num_users = len(self.current_users)
        total_antennas = 4 * 16  # 4个扇区，每扇区16个天线
        
        H = np.zeros((total_antennas, num_users), dtype=complex)
        
        antenna_idx = 0
        for sector in sectors:
            for antenna in sector['antennas']:
                for user_idx, user in enumerate(self.current_users):
                    distance = np.linalg.norm(user.position - antenna.position)
                    antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                        antenna, user, self.params
                    )
                    
                    if user.type == 'vehicle':
                        channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                            distance, antenna_gain_linear, antenna, user, self.params
                        )
                    else:
                        channel_coeff = ChannelModel.uav_channel_model_v2(
                            distance, antenna_gain_linear, user, self.params
                        )
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 计算速率（使用全局缓存的环境实例）
        temp_env = get_cached_environment(self.params)
        user_rates = temp_env._calculate_theoretical_rates_vectorized(H, self.transmit_power_dbm)
        
        return np.sum(user_rates)
    
    def update_scenario(self, time_step: float):
        """更新离散旋转场景"""
        # 更新用户位置
        seed_for_update = RANDOM_SEED + self.stats['total_updates'] + 5000
        self.current_users = UserMobility.update_user_positions(self.current_users, time_step, random_seed=seed_for_update)
        
        # 随机采样选择最优旋转组合
        best_rotations, best_rate = self._find_best_from_random_rotation_samples()
        
        # 记录结果
        self.stats['update_rates'].append(best_rate)
        self.stats['best_rotations'].append(best_rotations)
        self.stats['total_updates'] += 1
        
        print(f"    离散旋转更新{self.stats['total_updates']}: 最优旋转{self._format_rotation_angles(best_rotations)}, 总用户速率 {best_rate:.2f} Mbps")
        
        return best_rate
    
    def run_dynamic_scenario(self, max_updates: int):
        """运行动态离散旋转场景"""
        print(f"\n🔄 开始 {max_updates} 次离散旋转优化更新...")
        
        for update_count in range(max_updates):
            print(f"\n  --- 离散旋转更新 {update_count + 1}/{max_updates} ---")
            
            time_step = 1.0
            self.update_scenario(time_step)
        
        print(f"\n🔄 离散旋转方法完成 {max_updates} 次更新")


def run_circular_position_test(transmit_power_dbm: float = 23.0):
    """运行圆形位置方法测试"""
    print("\n🔄 圆形离散位置方法")
    print("=" * 80)
    
    # 创建系统参数
    params = SystemParams(
        num_ground_users=NUM_GROUND_USERS,
        num_air_users=NUM_AIR_USERS,
        num_surfaces=16,  # 4扇区×4天线表面
        environment_size=ENVIRONMENT_SIZE,
        air_height_range=AIR_HEIGHT_RANGE
    )
    
    # 创建圆形位置管理器
    circular_manager = CircularPositionManager(params, transmit_power_dbm)
    
    # 运行动态场景
    circular_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    
    return circular_manager


def run_discrete_rotation_test(transmit_power_dbm: float = 23.0):
    """运行离散旋转方法测试"""
    print("\n🔄 离散旋转方法")
    print("=" * 80)
    
    # 创建系统参数
    params = SystemParams(
        num_ground_users=NUM_GROUND_USERS,
        num_air_users=NUM_AIR_USERS,
        num_surfaces=16,  # 4扇区×4天线表面
        environment_size=ENVIRONMENT_SIZE,
        air_height_range=AIR_HEIGHT_RANGE
    )
    
    # 创建离散旋转管理器
    rotation_manager = DiscreteRotationManager(params, transmit_power_dbm)
    
    # 运行动态场景
    rotation_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    
    return rotation_manager


# 组合测试相关函数已删除 - 回到简单的分开测试模式


def run_user_count_tests_with_new_methods():
    """运行包含新方法的用户数量测试"""
    print("🚀 开始包含新方法的用户数量测试")
    print("=" * 80)
    
    test_results = {}
    
    for ground_user_count in GROUND_USER_COUNTS:
        print(f"\n📊 测试车辆用户数量: {ground_user_count}个 (空中用户: {NUM_AIR_USERS}个)")
        print("-" * 60)
        
        # 更新全局用户数量配置
        global NUM_GROUND_USERS
        NUM_GROUND_USERS = ground_user_count
        
        # 运行两种新方法
        print(f"  🔄 运行圆形位置方法...")
        circular_manager = run_circular_position_test()
        
        print(f"  🔄 运行离散旋转方法...")
        rotation_manager = run_discrete_rotation_test()
        
        # 计算统计结果
        circular_rates = circular_manager.stats['update_rates']
        rotation_rates = rotation_manager.stats['update_rates']
        
        circular_avg = np.mean(circular_rates)
        rotation_avg = np.mean(rotation_rates)
        
        result = {
            'ground_users': ground_user_count,
            'air_users': NUM_AIR_USERS,
            'total_users': ground_user_count + NUM_AIR_USERS,
            'circular_avg_rate': circular_avg,
            'circular_std': np.std(circular_rates),
            'circular_max': np.max(circular_rates),
            'circular_min': np.min(circular_rates),
            'rotation_avg_rate': rotation_avg,
            'rotation_std': np.std(rotation_rates),
            'rotation_max': np.max(rotation_rates),
            'rotation_min': np.min(rotation_rates),
            'rotation_vs_circular_improvement': ((rotation_avg - circular_avg) / circular_avg) * 100
        }
        
        test_results[ground_user_count] = result
        
        # 打印当前测试结果
        print(f"\n  📈 {ground_user_count}车用户测试结果:")
        print(f"    🔄 圆形位置: {circular_avg:.2f} ± {np.std(circular_rates):.2f} Mbps")
        print(f"    🔄 离散旋转: {rotation_avg:.2f} ± {np.std(rotation_rates):.2f} Mbps (相对圆形: {result['rotation_vs_circular_improvement']:+.1f}%)")
    
    # 打印汇总结果
    print(f"\n📊 新方法用户数量测试汇总")
    print("=" * 80)
    print(f"{'用户数':<8} {'总用户':<8} {'圆形(Mbps)':<12} {'旋转(Mbps)':<12} {'旋转提升%':<10}")
    print("-" * 60)
    
    for ground_count, result in test_results.items():
        circular_str = f"{result['circular_avg_rate']:.1f}"
        rotation_str = f"{result['rotation_avg_rate']:.1f}"
        improvement_str = f"{result['rotation_vs_circular_improvement']:+.1f}"
        
        print(f"{ground_count:<8} {result['total_users']:<8} {circular_str:<12} {rotation_str:<12} {improvement_str:<10}")
    
    # 保存测试结果
    output_dir = "adaptive_6dma_user_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/user_count_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 新方法测试结果已保存至: {output_dir}/user_count_test_results.json")
    
    return test_results


def run_power_tests_with_new_methods():
    """运行包含新方法的功率测试"""
    print("🚀 开始包含新方法的功率测试")
    print("=" * 80)
    
    ground_users, air_users = FIXED_USERS_FOR_POWER_TEST
    print(f"固定用户数量: {ground_users}个车辆 + {air_users}个空中用户")
    print(f"测试功率范围: {POWER_RANGE_MW} mW")
    
    # 更新全局用户数量配置
    global NUM_GROUND_USERS, NUM_AIR_USERS
    NUM_GROUND_USERS = ground_users
    NUM_AIR_USERS = air_users
    
    power_test_results = {}
    
    for power_mw in POWER_RANGE_MW:
        power_dbm = mw_to_dbm(power_mw)
        print(f"\n📊 测试发射功率: {power_mw} mW ({power_dbm:.1f} dBm)")
        print("-" * 60)
        
        # 运行两种新方法
        print(f"  🔄 运行圆形位置方法...")
        circular_manager = run_circular_position_test(transmit_power_dbm=power_dbm)
        
        print(f"  🔄 运行离散旋转方法...")
        rotation_manager = run_discrete_rotation_test(transmit_power_dbm=power_dbm)
        
        # 计算统计结果
        circular_rates = circular_manager.stats['update_rates']
        rotation_rates = rotation_manager.stats['update_rates']
        
        circular_avg = np.mean(circular_rates)
        rotation_avg = np.mean(rotation_rates)
        
        result = {
            'power_mw': power_mw,
            'power_dbm': power_dbm,
            'ground_users': ground_users,
            'air_users': air_users,
            'total_users': ground_users + air_users,
            'circular_avg_rate': circular_avg,
            'circular_std': np.std(circular_rates),
            'circular_max': np.max(circular_rates),
            'circular_min': np.min(circular_rates),
            'rotation_avg_rate': rotation_avg,
            'rotation_std': np.std(rotation_rates),
            'rotation_max': np.max(rotation_rates),
            'rotation_min': np.min(rotation_rates),
            'rotation_vs_circular_improvement': ((rotation_avg - circular_avg) / circular_avg) * 100
        }
        
        power_test_results[power_mw] = result
        
        # 打印当前测试结果
        print(f"\n  📈 {power_mw}mW功率测试结果:")
        print(f"    🔄 圆形位置: {circular_avg:.2f} ± {np.std(circular_rates):.2f} Mbps")
        print(f"    🔄 离散旋转: {rotation_avg:.2f} ± {np.std(rotation_rates):.2f} Mbps (相对圆形: {result['rotation_vs_circular_improvement']:+.1f}%)")
    
    # 打印汇总结果
    print(f"\n📊 新方法功率测试汇总")
    print("=" * 80)
    print(f"{'功率(mW)':<10} {'功率(dBm)':<10} {'圆形(Mbps)':<12} {'旋转(Mbps)':<12} {'旋转提升%':<10}")
    print("-" * 70)
    
    for power_mw, result in power_test_results.items():
        power_dbm_str = f"{result['power_dbm']:.1f}"
        circular_str = f"{result['circular_avg_rate']:.1f}"
        rotation_str = f"{result['rotation_avg_rate']:.1f}"
        improvement_str = f"{result['rotation_vs_circular_improvement']:+.1f}"
        
        print(f"{power_mw:<10} {power_dbm_str:<10} {circular_str:<12} {rotation_str:<12} {improvement_str:<10}")
    
    # 保存测试结果
    output_dir = "adaptive_6dma_power_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/power_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(power_test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 新方法功率测试结果已保存至: {output_dir}/power_test_results.json")
    
    return power_test_results


def main():
    """主函数：选择测试类型"""
    print("🚀 自适应6DMA方法性能测试")
    print("  🔄 6DMA圆形离散位置")
    print("  🔄 6DMA离散旋转")
    print()
    print("请选择测试类型:")
    print("1. 用户数量测试 (30-50个车辆用户)")
    print("2. 发射功率测试 (20-120mW)")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            print(f"\n📊 用户数量测试")
            print(f"  测试车辆用户数量: {GROUND_USER_COUNTS}")
            print(f"  固定空中用户数量: {NUM_AIR_USERS}")
            test_results = run_user_count_tests_with_new_methods()
            print(f"\n🎉 用户数量测试完成！")
            
        elif choice == "2":
            print(f"\n⚡ 发射功率测试")
            print(f"  测试功率范围: {POWER_RANGE_MW} mW")
            print(f"  固定用户数量: {FIXED_USERS_FOR_POWER_TEST[0]}车 + {FIXED_USERS_FOR_POWER_TEST[1]}空")
            test_results = run_power_tests_with_new_methods()
            print(f"\n🎉 发射功率测试完成！")
            
        else:
            print("❌ 无效选择，默认运行用户数量测试")
            test_results = run_user_count_tests_with_new_methods()
            print(f"\n🎉 用户数量测试完成！")
            
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
