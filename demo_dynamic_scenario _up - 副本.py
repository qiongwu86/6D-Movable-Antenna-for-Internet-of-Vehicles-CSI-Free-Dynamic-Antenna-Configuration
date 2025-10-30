import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from collections import defaultdict


from sixDMA_Environment_core_class import SystemParams, ActionSpace, ChannelModel, Antenna, User
# 导入修复后的_up版本
import importlib.util
import sys

# 导入修复后的_up版本
spec = importlib.util.spec_from_file_location("dynamic_scenario_manager", "dynamic_scenario_manager _up.py")
dynamic_scenario_manager_module = importlib.util.module_from_spec(spec)
sys.modules["dynamic_scenario_manager"] = dynamic_scenario_manager_module
spec.loader.exec_module(dynamic_scenario_manager_module)

from dynamic_scenario_manager import DynamicScenarioManager


# ============================================================================
# 🚀 增强的动态场景管理器（支持预测性部署）
# ============================================================================

class EnhancedDynamicScenarioManager(DynamicScenarioManager):
    """增强的动态场景管理器 - 支持预测性部署"""
    
    def __init__(self, params, optimization_results_path=None, 
                 enable_adaptive_mapping=False, stack_size=5, random_seed=42,
                 transmit_power_dbm=23.0, antenna_update_steps=10):
        
        # 调用父类初始化
        super().__init__(params, optimization_results_path, enable_adaptive_mapping, 
                        stack_size, random_seed, transmit_power_dbm)
        
        # 新增：异步更新配置
        self.vehicle_update_interval = 1.0   # 车辆更新间隔（秒）
        self.antenna_update_interval = 10.0  # 天线更新间隔（秒）
        self.antenna_update_steps = antenna_update_steps  # 可配置的预测步数
        
        # 新增：运动模型和预测器
        self.motion_model = VehicleMotionModel()
        self.motion_predictor = UserMotionPredictor(
            prediction_horizon=self.antenna_update_steps,
            dt=self.vehicle_update_interval
        )
        
        # 新增：时间管理
        self.current_time = 0.0
        self.last_antenna_update_time = 0.0
        self.vehicle_update_count = 0
        self.antenna_update_count = 0
        
        # 新增：累积密度历史
        self.cumulative_density_history = []
        
        # 新增：每次车辆更新的速率记录
        self.vehicle_update_rates = []  # 每次车辆更新后的速率
        self.antenna_update_avg_rates = []  # 每次天线更新周期的平均速率
        
        print(f"🚀 增强动态场景管理器初始化完成")
        print(f"  预测步数: {self.antenna_update_steps}步")
        print(f"  车辆更新间隔: {self.vehicle_update_interval}秒")
        print(f"  天线更新间隔: {self.antenna_update_interval}秒")
        
    def update_scenario_with_prediction(self, time_step: float):
        """带预测的场景更新（分离车辆和天线更新）"""
        self.current_time += time_step
        self.stats['total_updates'] += 1
        
        # 1. 更新车辆位置（每个时间步）
        self._update_vehicle_positions_with_variable_speed(time_step)
        self.vehicle_update_count += 1
        
        # 2. 🆕 每次车辆更新后计算速率
        current_rate = self._calculate_current_rate()
        if current_rate is not None:
            self.vehicle_update_rates.append(current_rate)
            # 只在每5次车辆更新时输出，避免信息过多
            if self.vehicle_update_count % 5 == 0:
                print(f"  车辆更新{self.vehicle_update_count}: 速率 {current_rate:.2f} Mbps")
        
        # 3. 检查是否需要更新天线配置（每10个时间步）
        if self.current_time - self.last_antenna_update_time >= self.antenna_update_interval:
            print(f"\n🔄 第{self.antenna_update_count + 1}次天线更新（时间: {self.current_time:.1f}s）")
            
            # 计算本周期的平均速率
            if len(self.vehicle_update_rates) >= self.antenna_update_steps:
                recent_rates = self.vehicle_update_rates[-self.antenna_update_steps:]
                avg_rate = np.mean(recent_rates)
                self.antenna_update_avg_rates.append(avg_rate)
                print(f"  📊 本周期平均速率: {avg_rate:.2f} Mbps (基于{len(recent_rates)}次车辆更新)")
            
            # 执行天线更新
            self._update_antenna_with_prediction()
            self.last_antenna_update_time = self.current_time
            self.antenna_update_count += 1
            
    def _update_vehicle_positions_with_variable_speed(self, dt: float):
        """使用变速模型更新车辆位置（保留原有十字路口逻辑）"""
        seed_offset = int(self.current_time * 1000)  # 基于时间的种子偏移
        
        # 🔍 调试：记录更新前的位置（仅第一个用户）
        if self.current_users and self.vehicle_update_count % 10 == 1:
            first_user = self.current_users[0]
            if first_user.type == 'vehicle':
                old_pos = first_user.position.copy()
        
        for user in self.current_users:
            if user.type == 'vehicle':
                # 记录更新前位置
                old_position = user.position.copy()
                
                # 🆕 采样新速度（变速模型）
                user_seed = self.random_seed + user.id + seed_offset
                new_speed = self.motion_model.sample_speed(seed=user_seed)
                user.velocity = new_speed
                
                # 记录速度历史（可选）
                if not hasattr(user, 'velocity_history'):
                    user.velocity_history = []
                user.velocity_history.append(new_speed)
                
                # 🔄 使用原有的位置更新和边界处理逻辑
                # 车辆沿道路移动
                displacement = user.velocity * dt * user.direction
                user.position += displacement

                # 原有的边界处理逻辑（保持十字路口环境）
                if user.lane in ['north_bound', 'south_bound']:
                    if user.position[1] > 300:
                        user.position[1] = 0
                    elif user.position[1] < 0:
                        user.position[1] = 300
                else:
                    if user.position[0] > 300:
                        user.position[0] = 0
                    elif user.position[0] < 0:
                        user.position[0] = 300
                
                # 🔍 验证位置确实更新了
                if self.vehicle_update_count % 10 == 1 and user.id == 1:  # 只检查第一个用户
                    displacement_actual = np.linalg.norm(user.position - old_position)
                    expected_displacement = user.velocity * dt
                    print(f"    🔍 用户{user.id}位置更新: {old_position[:2]} -> {user.position[:2]}")
                    print(f"      实际位移: {displacement_actual:.2f}m, 预期位移: {expected_displacement:.2f}m")
                        
            elif user.type == 'UAV':
                # 无人机使用原有的更新逻辑
                self._update_uav_position(user, dt)
    
    def _calculate_current_rate(self):
        """计算当前天线配置下的用户速率（直接使用父类方法）"""
        try:
            # 如果还没有天线分配，返回None
            if not hasattr(self, 'antenna_allocations') or not self.antenna_allocations:
                if self.vehicle_update_count % 10 == 1:
                    print(f"    ⚠️  没有天线分配，无法计算速率")
                return None
                
            # 检查用户数量
            if not self.current_users:
                print(f"    ⚠️  没有用户，无法计算速率")
                return None
            
            # 🔍 调试信息：显示当前状态
            if self.vehicle_update_count % 10 == 1:  # 每10次更新显示一次详细信息
                print(f"    🔍 速率计算详情:")
                print(f"      用户数: {len(self.current_users)} (地面: {sum(1 for u in self.current_users if u.type == 'vehicle')}, 空中: {sum(1 for u in self.current_users if u.type == 'UAV')})")
                print(f"      天线分配数: {len(self.antenna_allocations)}")
                
                # 显示部分用户位置
                for i, user in enumerate(self.current_users[:3]):  # 只显示前3个用户
                    print(f"      用户{user.id}: 位置{user.position[:2]}, 速度{user.velocity:.1f}m/s, 车道{getattr(user, 'lane', 'N/A')}")
            
            # 🔄 直接使用父类的速率计算方法（已经实现了完整的逻辑）
            total_rate = self._calculate_total_user_rate()
            
            # 验证速率计算结果
            if total_rate is None or total_rate <= 0:
                if self.vehicle_update_count % 10 == 1:
                    print(f"    ⚠️  速率计算结果异常: {total_rate}")
                return 0.0
            
            return total_rate
            
        except Exception as e:
            print(f"    ⚠️  速率计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None
                
    def _update_uav_position(self, user, dt: float):
        """更新无人机位置（使用原有逻辑）"""
        # 检查是否已初始化轨道参数
        if not hasattr(user, 'orbit_radius') or user.orbit_radius == 0.0:
            # 从当前位置推导轨道参数
            center_2d = np.array([150, 150])
            current_pos_2d = user.position[:2]
            user.orbit_center = np.array([150, 150, user.height])
            user.orbit_radius = max(np.linalg.norm(current_pos_2d - center_2d), 30)
            user.orbit_angle = np.arctan2(current_pos_2d[1] - center_2d[1], 
                                        current_pos_2d[0] - center_2d[0])
        
        # 更新轨道角度（角速度 = 线速度 / 半径）
        angular_velocity = user.velocity / user.orbit_radius
        user.orbit_angle += angular_velocity * dt
        
        # 保持角度在[0, 2π]范围内
        user.orbit_angle = user.orbit_angle % (2 * np.pi)
        
        # 更新水平位置
        user.position[0] = user.orbit_center[0] + user.orbit_radius * np.cos(user.orbit_angle)
        user.position[1] = user.orbit_center[1] + user.orbit_radius * np.sin(user.orbit_angle)
        
        # 垂直运动处理（简化版本）
        if hasattr(user, 'vertical_velocity') and abs(user.vertical_velocity) > 0.01:
            user.height += user.vertical_velocity * dt
            user.position[2] = user.height
            
            # 简单的高度限制
            if user.height >= 100 or user.height <= 50:
                user.vertical_velocity = -user.vertical_velocity
                
    def _update_antenna_with_prediction(self):
        """基于预测的累积密度更新天线配置"""
        
        # 1. 计算累积网格密度（10次预测叠加）
        print(f"  📊 计算未来{self.antenna_update_steps}步的累积用户分布...")
        cumulative_density = self.motion_predictor.calculate_cumulative_grid_density(
            self.current_users, 
            self.grid_cells
        )
        
        # 保存累积密度历史
        self.cumulative_density_history.append({
            'time': self.current_time,
            'density': dict(cumulative_density)
        })
        
        print(f"  📈 累积密度统计: {len(cumulative_density)}个网格有用户分布")
        
        # 2. 基于累积密度更新网格用户信息
        self._update_grid_info_from_cumulative_density(cumulative_density)
        
        # 3. 执行天线分配
        if self.enable_adaptive_mapping:
            self._allocate_antennas_with_adaptive_mapping()
        else:
            self._allocate_antennas_with_optimization()
            
        print(f"  ✅ 天线配置更新完成，共分配{len(self.antenna_allocations)}个天线")
        
    def _update_grid_info_from_cumulative_density(self, cumulative_density: Dict[int, float]):
        """基于累积密度更新网格信息（用于天线分配）"""
        # 清空当前网格用户信息
        for grid_id in self.grid_user_info:
            self.grid_user_info[grid_id].user_count = 0
            self.grid_user_info[grid_id].user_ids = []
            
        # 根据累积密度更新
        total_density = 0
        for grid_id, density in cumulative_density.items():
            if grid_id in self.grid_user_info:
                # 使用累积密度的平均值作为"等效用户数"
                avg_users = density / self.antenna_update_steps
                self.grid_user_info[grid_id].user_count = max(1, int(avg_users))
                
                # 保存累积密度信息
                if not hasattr(self.grid_user_info[grid_id], 'cumulative_density'):
                    self.grid_user_info[grid_id].cumulative_density = density
                else:
                    self.grid_user_info[grid_id].cumulative_density = density
                    
                total_density += density
                
        print(f"  📊 总累积密度: {total_density:.1f}, 平均每步: {total_density/self.antenna_update_steps:.1f}用户")
        
    def get_final_statistics(self):
        """获取最终统计信息"""
        # 使用天线更新周期的平均速率作为主要统计
        if self.antenna_update_avg_rates:
            self.stats['update_rates'] = self.antenna_update_avg_rates.copy()
            self.stats['avg_update_rate'] = np.mean(self.antenna_update_avg_rates)
        else:
            # 如果没有天线更新周期数据，使用车辆更新速率
            self.stats['update_rates'] = self.vehicle_update_rates.copy()
            self.stats['avg_update_rate'] = np.mean(self.vehicle_update_rates) if self.vehicle_update_rates else 0
            
        # 添加详细统计信息
        self.stats['vehicle_updates'] = self.vehicle_update_count
        self.stats['antenna_updates'] = self.antenna_update_count
        self.stats['vehicle_update_rates'] = self.vehicle_update_rates.copy()
        self.stats['antenna_avg_rates'] = self.antenna_update_avg_rates.copy()
        
        return self.stats


# ============================================================================
# 🚗 车辆运动模型类
# ============================================================================

class VehicleMotionModel:
    """车辆运动模型 - 截断高斯分布速度"""
    def __init__(self, speed_mean: float = 15.0, speed_std: float = 2.5, 
                 speed_min: float = 10.0, speed_max: float = 20.0):
        self.speed_mean = speed_mean  # 速度分布中心 (m/s)
        self.speed_std = speed_std    # 速度标准差
        self.speed_min = speed_min    # 最小速度
        self.speed_max = speed_max    # 最大速度
        
    def sample_speed(self, seed: int = None) -> float:
        """采样截断高斯分布的速度"""
        if seed is not None:
            np.random.seed(seed)
        
        # 使用截断正态分布
        max_attempts = 100  # 防止无限循环
        for _ in range(max_attempts):
            speed = np.random.normal(self.speed_mean, self.speed_std)
            if self.speed_min <= speed <= self.speed_max:
                return speed
        
        # 如果采样失败，返回均值
        return self.speed_mean
                
    def get_prediction_speed(self) -> float:
        """返回预测速度（使用分布中心）"""
        return self.speed_mean


class UserMotionPredictor:
    """用户运动预测器"""
    def __init__(self, prediction_horizon: int = 10, dt: float = 1.0):
        self.prediction_horizon = prediction_horizon  # 预测步数
        self.dt = dt  # 时间步长
        self.motion_model = VehicleMotionModel()
        
    def predict_vehicle_positions(self, user, prediction_steps: int) -> List[np.ndarray]:
        """预测车辆未来位置（保留原有十字路口边界处理）"""
        predicted_positions = []
        current_pos = user.position.copy()
        
        # 使用分布中心速度进行预测
        predicted_speed = self.motion_model.get_prediction_speed()
        
        for t in range(prediction_steps):
            # 计算位移
            displacement = predicted_speed * self.dt * user.direction
            current_pos = current_pos + displacement
            
            # 🔄 使用与原始代码完全一致的边界处理逻辑
            if user.lane in ['north_bound', 'south_bound']:
                if current_pos[1] > 300:
                    current_pos[1] = 0  # 直接重置到起点
                elif current_pos[1] < 0:
                    current_pos[1] = 300  # 直接重置到终点
            else:  # east_bound, west_bound
                if current_pos[0] > 300:
                    current_pos[0] = 0  # 直接重置到起点
                elif current_pos[0] < 0:
                    current_pos[0] = 300  # 直接重置到终点
                    
            predicted_positions.append(current_pos.copy())
            
        return predicted_positions
    
    def predict_uav_positions(self, user, prediction_steps: int) -> List[np.ndarray]:
        """预测无人机未来位置（简化预测）"""
        predicted_positions = []
        
        # 当前轨道参数
        current_angle = user.orbit_angle if hasattr(user, 'orbit_angle') else 0
        current_height = user.height
        orbit_radius = user.orbit_radius if hasattr(user, 'orbit_radius') else 50
        orbit_center = user.orbit_center[:2] if hasattr(user, 'orbit_center') else np.array([150, 150])
        
        # 角速度
        angular_velocity = user.velocity / orbit_radius
        
        # 垂直速度（如果有）
        vertical_velocity = user.vertical_velocity if hasattr(user, 'vertical_velocity') else 0
        target_height = user.target_height if hasattr(user, 'target_height') else current_height
        
        for t in range(prediction_steps):
            # 预测水平位置（环绕运动）
            predicted_angle = current_angle + angular_velocity * self.dt * (t + 1)
            predicted_angle = predicted_angle % (2 * np.pi)
            
            x = orbit_center[0] + orbit_radius * np.cos(predicted_angle)
            y = orbit_center[1] + orbit_radius * np.sin(predicted_angle)
            
            # 预测垂直位置（简化：线性插值到目标高度）
            if abs(vertical_velocity) > 0.01:
                predicted_height = current_height + vertical_velocity * self.dt * (t + 1)
                # 限制在目标高度
                if vertical_velocity > 0:
                    predicted_height = min(predicted_height, target_height)
                else:
                    predicted_height = max(predicted_height, target_height)
            else:
                predicted_height = current_height
                
            # 确保高度在合理范围内
            predicted_height = np.clip(predicted_height, 50, 100)
            
            predicted_positions.append(np.array([x, y, predicted_height]))
            
        return predicted_positions
    
    def predict_all_users_positions(self, users: List) -> Dict[int, List[np.ndarray]]:
        """预测所有用户的未来位置"""
        predictions = {}
        
        for user in users:
            if user.type == 'vehicle':
                predictions[user.id] = self.predict_vehicle_positions(user, self.prediction_horizon)
            else:  # UAV
                predictions[user.id] = self.predict_uav_positions(user, self.prediction_horizon)
                
        return predictions
    
    def calculate_cumulative_grid_density(self, users: List, grid_cells: List, 
                                         grid_size: float = 15.0) -> Dict[int, float]:
        """计算累积网格密度（10次预测叠加）"""
        cumulative_density = defaultdict(float)
        
        # 获取所有用户的预测位置
        all_predictions = self.predict_all_users_positions(users)
        
        # 对每个预测时间步
        for t in range(self.prediction_horizon):
            # 计算该时间步的网格密度
            grid_density = defaultdict(int)
            
            for user_id, predicted_positions in all_predictions.items():
                if t < len(predicted_positions):
                    pos = predicted_positions[t]
                    
                    # 找到用户所在的网格
                    grid_id = self._find_grid_for_position(pos, grid_cells, grid_size)
                    if grid_id is not None:
                        grid_density[grid_id] += 1
            
            # 累加到总密度
            for grid_id, density in grid_density.items():
                cumulative_density[grid_id] += density
                
        return cumulative_density
    
    def _find_grid_for_position(self, position: np.ndarray, grid_cells: List, 
                               grid_size: float) -> int:
        """根据位置找到对应的网格ID"""
        x, y, z = position
        
        # 确保坐标在有效范围内
        if x < 0 or x >= 300 or y < 0 or y >= 300:
            return None
        
        # 判断是地面还是空中
        if z < 10:  # 地面用户
            col_idx = int(x / grid_size)
            row_idx = int(y / grid_size)
            grid_id = row_idx * 20 + col_idx  # 假设20x20网格
            
            # 确保网格ID在有效范围内
            if 0 <= grid_id < 400:
                return grid_id
        else:  # 空中用户
            col_idx = int(x / grid_size)
            row_idx = int(y / grid_size)
            grid_id = 400 + row_idx * 20 + col_idx  # 空中网格从400开始
            
            # 确保网格ID在有效范围内
            if 400 <= grid_id < 800:
                return grid_id
                
        return None


# ============================================================================
# 🔧 配置参数 - 在这里快速调整测试参数
# ============================================================================

# 用户配置
NUM_GROUND_USERS = 30      # 地面车辆用户数量
NUM_AIR_USERS = 0       # 空中无人机用户数量

# 天线配置  
NUM_ANTENNA_SURFACES = 16  # 天线表面数量

# 测试配置
GROUND_USER_COUNTS = [30, 35, 40, 45, 50]  # 不同的车辆用户数量测试
POWER_RANGE_MW = [20, 40, 60, 80, 100, 120]  # 发射功率范围 (mW)
FIXED_USERS_FOR_POWER_TEST = (30, 5)  # 功率测试时固定的用户数量 (地面, 空中)

# 场景配置
MAX_UPDATES = 50           # 动态场景更新次数（增加到50）
RANDOM_SEED = 42           # 随机种子（保证结果可复现）
ENABLE_ADAPTIVE_MAPPING = True  # 是否启用自适应网格-天线映射
STACK_SIZE = 5            # 每个网格的预存天线位置数（可快速调整）

# 🆕 预测性部署配置
VEHICLE_UPDATE_INTERVAL = 1.0   # 车辆更新间隔（秒）
ANTENNA_UPDATE_INTERVAL = 10.0  # 天线更新间隔（秒）  
ANTENNA_UPDATE_STEPS = 10       # 天线预测步数（可调整）

# 环境配置
ENVIRONMENT_SIZE = (300, 300, 100)  # 环境尺寸 (长, 宽, 高)
AIR_HEIGHT_RANGE = (50.0, 100.0)   # 空中用户高度范围

print(f"🚀 预测性6DMA部署性能测试")
print(f"🔧 配置参数:")
print(f"  基准地面用户: {NUM_GROUND_USERS}个")
print(f"  空中用户: {NUM_AIR_USERS}个") 
print(f"  天线表面: {NUM_ANTENNA_SURFACES}个")
print(f"  更新次数: {MAX_UPDATES}次")
print(f"  自适应映射: {'✅ 启用' if ENABLE_ADAPTIVE_MAPPING else '❌ 禁用'}")
if ENABLE_ADAPTIVE_MAPPING:
    print(f"  堆栈大小: {STACK_SIZE}个天线配置/网格")
print(f"🆕 预测性部署配置:")
print(f"  车辆更新间隔: {VEHICLE_UPDATE_INTERVAL}秒")
print(f"  天线更新间隔: {ANTENNA_UPDATE_INTERVAL}秒")
print(f"  预测步数: {ANTENNA_UPDATE_STEPS}步")
print(f"📊 测试配置:")
print(f"  用户数量测试: {GROUND_USER_COUNTS}")
print(f"  功率测试: {POWER_RANGE_MW} mW")
print("=" * 80)


# ============================================================================
# 🔧 辅助函数
# ============================================================================

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
        from sixdma_environment_optimized import OptimizedSixDMAEnvironment
        _cached_env = OptimizedSixDMAEnvironment(params)
        _cached_params = params
    
    return _cached_env

def mw_to_dbm(power_mw: float) -> float:
    """将功率从mW转换为dBm"""
    return 10 * np.log10(power_mw)

def dbm_to_mw(power_dbm: float) -> float:
    """将功率从dBm转换为mW"""
    return 10 ** (power_dbm / 10)


# ============================================================================
# 🔧 FPA和随机天线管理器（从test_dynamic_scenario.py移植）
# ============================================================================

class FPAManager:
    """传统FPA固定相控阵管理器"""
    def __init__(self, params: SystemParams, transmit_power_dbm: float = 23.0):
        self.params = params
        self.transmit_power_dbm = transmit_power_dbm
        self.stats = {
            'total_updates': 0,
            'update_rates': [],
            'avg_update_rate': 0
        }
        
    def run_dynamic_scenario(self, max_updates: int = MAX_UPDATES):
        """运行FPA动态场景"""
        from sixDMA_Environment_core_class import UserMobility
        from sixdma_environment_optimized import OptimizedSixDMAEnvironment
        
        # 生成用户
        users = UserMobility.generate_user_positions(self.params, seed=RANDOM_SEED)
        
        # 创建环境
        env = OptimizedSixDMAEnvironment(self.params)
        
        # FPA配置：4扇区，每扇区90°覆盖，4x4矩形阵列，15°下倾
        fpa_antennas = self._generate_fpa_configuration()
        
        for update in range(max_updates):
            # 更新用户位置
            seed_for_update = RANDOM_SEED + update + 1000
            users = UserMobility.update_user_positions(users, dt=1.0, random_seed=seed_for_update)
            
            # 计算速率（使用test_dynamic_scenario.py中的方法）
            total_rate = calculate_antenna_config_rate(fpa_antennas, users, self.params, self.transmit_power_dbm)
            self.stats['update_rates'].append(total_rate)
            self.stats['total_updates'] += 1
        
        self.stats['avg_update_rate'] = np.mean(self.stats['update_rates'])
        
    def _generate_fpa_configuration(self):
        """生成FPA天线配置"""
        # 基站中心位置
        base_station_center = np.array([150, 150, 30])
        
        # 4个扇区的方位角
        sector_azimuths = [0, 90, 180, 270]  # 度
        downtilt = 15  # 下倾角度
        
        antenna_configs = []
        
        for azimuth in sector_azimuths:
            # 计算法向量（考虑方位角和下倾角）
            azimuth_rad = np.radians(azimuth)
            downtilt_rad = np.radians(downtilt)
            
            normal = np.array([
                np.cos(downtilt_rad) * np.cos(azimuth_rad),
                np.cos(downtilt_rad) * np.sin(azimuth_rad),
                -np.sin(downtilt_rad)
            ])
            
            antenna_configs.append({
                'position': base_station_center,
                'normal': normal,
                'rotation_type': 'fixed',
                'surface_id': len(antenna_configs)  # 添加surface_id字段
            })
        
        return antenna_configs


class RandomAntennaManager:
    """随机天线配置管理器"""
    def __init__(self, params: SystemParams, transmit_power_dbm: float = 23.0):
        self.params = params
        self.transmit_power_dbm = transmit_power_dbm
        self.stats = {
            'total_updates': 0,
            'update_rates': [],
            'avg_update_rate': 0
        }
        
    def run_dynamic_scenario(self, max_updates: int = MAX_UPDATES):
        """运行随机天线动态场景"""
        from sixDMA_Environment_core_class import UserMobility
        from sixdma_environment_optimized import OptimizedSixDMAEnvironment
        
        # 生成用户
        users = UserMobility.generate_user_positions(self.params, seed=RANDOM_SEED)
        
        # 创建环境
        env = OptimizedSixDMAEnvironment(self.params)
        
        for update in range(max_updates):
            # 更新用户位置
            seed_for_update = RANDOM_SEED + update + 1000
            users = UserMobility.update_user_positions(users, dt=1.0, random_seed=seed_for_update)
            
            # 生成随机天线配置
            random_antennas = self._generate_random_configuration(update)
            
            # 计算速率（使用test_dynamic_scenario.py中的方法）
            total_rate = calculate_antenna_config_rate(random_antennas, users, self.params, self.transmit_power_dbm)
            self.stats['update_rates'].append(total_rate)
            self.stats['total_updates'] += 1
        
        self.stats['avg_update_rate'] = np.mean(self.stats['update_rates'])
        
    def _generate_random_configuration(self, seed_offset: int = 0):
        """生成随机天线配置"""
        np.random.seed(RANDOM_SEED + seed_offset + 2000)
        
        antenna_configs = []
        
        for i in range(self.params.num_surfaces):
            # 随机位置（在环境范围内）
            position = np.array([
                np.random.uniform(0, self.params.environment_size[0]),
                np.random.uniform(0, self.params.environment_size[1]),
                np.random.uniform(10, self.params.environment_size[2])
            ])
            
            # 随机方向
            theta = np.random.uniform(0, 2 * np.pi)  # 方位角
            phi = np.random.uniform(0, np.pi/3)      # 俯仰角（限制在60度内）
            
            normal = np.array([
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                -np.sin(phi)
            ])
            
            antenna_configs.append({
                'position': position,
                'normal': normal,
                'rotation_type': 'random',
                'surface_id': i  # 添加surface_id字段
            })
        
        return antenna_configs


def calculate_antenna_config_rate(antenna_configs: List[Dict], users: List[User], params: SystemParams, transmit_power_dbm: float = 23.0) -> float:
    """计算给定天线配置的系统总速率"""
    try:
        num_users = len(users)
        num_antennas = len(antenna_configs) * 4  # 每个表面4个天线
        
        if num_antennas == 0 or num_users == 0:
            return 0.0
        
        # 构建信道矩阵
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        antenna_idx = 0
        for config in antenna_configs:
            # 生成4天线阵列位置
            antenna_array_positions = generate_4_antenna_array(
                config['position'], config['normal'], params
            )
            
            for ant_pos in antenna_array_positions:
                # 创建天线对象
                antenna = Antenna(
                    surface_id=antenna_idx // 4,
                    global_id=antenna_idx,
                    local_id=antenna_idx % 4,
                    position=ant_pos,
                    normal=config['normal'],
                    surface_center=config['position']
                )
                
                # 计算该天线对所有用户的信道系数
                for user_idx, user in enumerate(users):
                    distance = np.linalg.norm(user.position - config['position'])
                    antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                        antenna, user, params
                    )
                    
                    if user.type == 'vehicle':
                        channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                            distance, antenna_gain_linear, antenna, user, params
                        )
                    else:
                        channel_coeff = ChannelModel.uav_channel_model_v2(
                            distance, antenna_gain_linear, user, params
                        )
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 计算理论速率（使用缓存的环境实例）
        temp_env = get_cached_environment(params)
        user_rates = temp_env._calculate_theoretical_rates_vectorized(H, transmit_power_dbm)
        return np.sum(user_rates)
        
    except Exception as e:
        print(f"计算天线配置速率时出错: {e}")
        return 0.0


def generate_4_antenna_array(center_pos: np.ndarray, normal: np.ndarray, params: SystemParams) -> List[np.ndarray]:
    """生成4天线矩形阵列位置（2x2配置）"""
    spacing = params.antenna_spacing
    
    # 构建局部坐标系
    if abs(normal[2]) < 0.9:
        ref_vec = np.array([0, 0, 1])
    else:
        ref_vec = np.array([1, 0, 0])
    
    # 计算局部坐标系的两个切向量
    u_vec = np.cross(normal, ref_vec)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    
    # 生成2x2阵列的4个位置（相对于中心的偏移）
    offsets = [
        (-spacing/2, -spacing/2),  # 左下
        ( spacing/2, -spacing/2),  # 右下
        (-spacing/2,  spacing/2),  # 左上
        ( spacing/2,  spacing/2)   # 右上
    ]
    
    antenna_positions = []
    for u_offset, v_offset in offsets:
        position = center_pos + u_offset * u_vec + v_offset * v_vec
        antenna_positions.append(position)
    
    return antenna_positions


def run_four_method_test(ground_users: int = NUM_GROUND_USERS, 
                        air_users: int = NUM_AIR_USERS, 
                        transmit_power_dbm: float = 23.0):
    """运行四种方法的对比测试"""
    print(f"\n📊 四种方法对比测试 - 用户: {ground_users}+{air_users}, 功率: {transmit_power_dbm:.1f}dBm")
    print("=" * 80)
    
    # 检查优化结果
    optimization_path = "demo_optimization_results"
    if not os.path.exists(f"{optimization_path}/complete_optimization_data.pkl"):
        print("⚠️  未找到优化结果，无法运行动态场景")
        print("   请先运行 python demo_grid_optimization.py 生成优化结果")
        return None
    
    # 创建系统参数
    params = SystemParams(
        num_ground_users=ground_users,
        num_air_users=air_users,
        num_surfaces=NUM_ANTENNA_SURFACES,
        environment_size=ENVIRONMENT_SIZE,
        air_height_range=AIR_HEIGHT_RANGE
    )
    
    results = {}
    
    # 1. 运行FPA方法
    print(f"  📡 运行FPA方法...")
    fpa_manager = FPAManager(params, transmit_power_dbm)
    fpa_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    fpa_rates = fpa_manager.stats['update_rates']
    fpa_avg = np.mean(fpa_rates)
    results['fpa'] = {
        'avg_rate': fpa_avg,
        'std_rate': np.std(fpa_rates),
        'max_rate': np.max(fpa_rates),
        'min_rate': np.min(fpa_rates),
        'rates': fpa_rates
    }
    print(f"    ✅ FPA完成: 平均速率 {fpa_avg:.2f} Mbps")
    
    # 2. 运行随机天线方法
    print(f"  🎲 运行随机天线方法...")
    random_manager = RandomAntennaManager(params, transmit_power_dbm)
    random_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    random_rates = random_manager.stats['update_rates']
    random_avg = np.mean(random_rates)
    results['random'] = {
        'avg_rate': random_avg,
        'std_rate': np.std(random_rates),
        'max_rate': np.max(random_rates),
        'min_rate': np.min(random_rates),
        'rates': random_rates
    }
    print(f"    ✅ 随机完成: 平均速率 {random_avg:.2f} Mbps")
    
    # 3. 运行优化天线方法（传统）
    print(f"  🎯 运行优化天线方法...")
    scenario_manager = DynamicScenarioManager(
        params=params,
        optimization_results_path=optimization_path,
        enable_adaptive_mapping=ENABLE_ADAPTIVE_MAPPING,
        stack_size=STACK_SIZE,
        random_seed=RANDOM_SEED,
        transmit_power_dbm=transmit_power_dbm
    )
    scenario_manager.initialize_scenario()
    scenario_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    optimized_rates = scenario_manager.stats['update_rates']
    optimized_avg = np.mean(optimized_rates)
    results['optimized'] = {
        'avg_rate': optimized_avg,
        'std_rate': np.std(optimized_rates),
        'max_rate': np.max(optimized_rates),
        'min_rate': np.min(optimized_rates),
        'rates': optimized_rates
    }
    print(f"    ✅ 优化完成: 平均速率 {optimized_avg:.2f} Mbps")
    
    # 4. 运行预测性部署方法
    print(f"  🆕 运行预测性部署方法...")
    predictive_manager = EnhancedDynamicScenarioManager(
        params=params,
        optimization_results_path=optimization_path,
        enable_adaptive_mapping=ENABLE_ADAPTIVE_MAPPING,
        stack_size=STACK_SIZE,
        random_seed=RANDOM_SEED,
        transmit_power_dbm=transmit_power_dbm,
        antenna_update_steps=ANTENNA_UPDATE_STEPS
    )
    
    # 初始化场景
    predictive_manager.initialize_scenario()
    predictive_manager._update_grid_user_mapping()
    predictive_manager._allocate_antennas_with_adaptive_mapping()
    
    # 运行预测性部署仿真
    total_simulation_time = MAX_UPDATES * VEHICLE_UPDATE_INTERVAL
    current_time = 0
    
    while current_time < total_simulation_time:
        predictive_manager.update_scenario_with_prediction(VEHICLE_UPDATE_INTERVAL)
        current_time += VEHICLE_UPDATE_INTERVAL
    
    # 整理预测性部署结果
    predictive_manager.get_final_statistics()
    predictive_rates = predictive_manager.stats['update_rates']
    
    if predictive_rates:
        predictive_avg = np.mean(predictive_rates)
        results['predictive'] = {
            'avg_rate': predictive_avg,
            'std_rate': np.std(predictive_rates),
            'max_rate': np.max(predictive_rates),
            'min_rate': np.min(predictive_rates),
            'rates': predictive_rates,
            'antenna_updates': predictive_manager.antenna_update_count,
            'vehicle_updates': predictive_manager.vehicle_update_count
        }
        print(f"    ✅ 预测性部署完成: 平均速率 {predictive_avg:.2f} Mbps")
    else:
        print(f"    ❌ 预测性部署失败：没有速率数据")
        results['predictive'] = None
    
    # 计算相对改进
    if results['predictive']:
        results['random_vs_fpa_improvement'] = ((random_avg - fpa_avg) / fpa_avg) * 100
        results['optimized_vs_fpa_improvement'] = ((optimized_avg - fpa_avg) / fpa_avg) * 100
        results['predictive_vs_fpa_improvement'] = ((predictive_avg - fpa_avg) / fpa_avg) * 100
        results['predictive_vs_optimized_improvement'] = ((predictive_avg - optimized_avg) / optimized_avg) * 100
    
    # 添加测试参数
    results['ground_users'] = ground_users
    results['air_users'] = air_users
    results['transmit_power_dbm'] = transmit_power_dbm
    
    return results


def run_user_count_tests():
    """运行不同用户数量的四种方法测试"""
    print(f"\n📊 用户数量测试（四种方法对比）")
    print(f"  测试车辆用户数量: {GROUND_USER_COUNTS}")
    print(f"  固定空中用户数量: {NUM_AIR_USERS}")
    print("=" * 80)
    
    user_count_results = {}
    
    for ground_count in GROUND_USER_COUNTS:
        print(f"\n🚗 测试 {ground_count} 个地面用户 + {NUM_AIR_USERS} 个空中用户")
        
        result = run_four_method_test(
            ground_users=ground_count,
            air_users=NUM_AIR_USERS,
            transmit_power_dbm=23.0  # 默认功率
        )
        
        if result:
            user_count_results[ground_count] = result
            print(f"\n  📈 测试结果汇总:")
            print(f"    📡 FPA: {result['fpa']['avg_rate']:.2f} Mbps")
            print(f"    🎲 随机: {result['random']['avg_rate']:.2f} Mbps (相对FPA: {result['random_vs_fpa_improvement']:+.1f}%)")
            print(f"    🎯 优化: {result['optimized']['avg_rate']:.2f} Mbps (相对FPA: {result['optimized_vs_fpa_improvement']:+.1f}%)")
            if result['predictive']:
                print(f"    🆕 预测: {result['predictive']['avg_rate']:.2f} Mbps (相对FPA: {result['predictive_vs_fpa_improvement']:+.1f}%, 相对优化: {result['predictive_vs_optimized_improvement']:+.1f}%)")
        else:
            print(f"  ❌ 失败")
    
    # 打印汇总结果
    print(f"\n📊 用户数量测试汇总结果")
    print("=" * 120)
    print(f"{'用户数':<8} {'FPA(Mbps)':<12} {'随机(Mbps)':<12} {'优化(Mbps)':<12} {'预测(Mbps)':<12} {'随机提升%':<10} {'优化提升%':<10} {'预测提升%':<10}")
    print("-" * 120)
    
    for ground_count, result in user_count_results.items():
        user_str = f"{ground_count}+{result['air_users']}"
        fpa_str = f"{result['fpa']['avg_rate']:.2f}"
        random_str = f"{result['random']['avg_rate']:.2f}"
        opt_str = f"{result['optimized']['avg_rate']:.2f}"
        
        if result['predictive']:
            pred_str = f"{result['predictive']['avg_rate']:.2f}"
            pred_imp_str = f"{result['predictive_vs_fpa_improvement']:+.1f}"
        else:
            pred_str = "N/A"
            pred_imp_str = "N/A"
        
        random_imp_str = f"{result['random_vs_fpa_improvement']:+.1f}"
        opt_imp_str = f"{result['optimized_vs_fpa_improvement']:+.1f}"
        
        print(f"{user_str:<8} {fpa_str:<12} {random_str:<12} {opt_str:<12} {pred_str:<12} {random_imp_str:<10} {opt_imp_str:<10} {pred_imp_str:<10}")
    
    # 保存测试结果
    output_dir = "predictive_deployment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    with open(f"{output_dir}/user_count_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(user_count_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 用户数量测试结果已保存至: {output_dir}/user_count_test_results.json")
    
    return user_count_results


def run_power_tests():
    """运行不同发射功率的四种方法测试"""
    print(f"\n⚡ 发射功率测试（四种方法对比）")
    print(f"  测试功率范围: {POWER_RANGE_MW} mW")
    print(f"  固定用户数量: {FIXED_USERS_FOR_POWER_TEST[0]}车 + {FIXED_USERS_FOR_POWER_TEST[1]}空")
    print("=" * 80)
    
    power_test_results = {}
    
    for power_mw in POWER_RANGE_MW:
        power_dbm = mw_to_dbm(power_mw)
        print(f"\n⚡ 测试功率: {power_mw}mW ({power_dbm:.1f}dBm)")
        
        result = run_four_method_test(
            ground_users=FIXED_USERS_FOR_POWER_TEST[0],
            air_users=FIXED_USERS_FOR_POWER_TEST[1],
            transmit_power_dbm=power_dbm
        )
        
        if result:
            result['power_mw'] = power_mw
            result['power_dbm'] = power_dbm
            power_test_results[power_mw] = result
            
            print(f"\n  📈 测试结果汇总:")
            print(f"    📡 FPA: {result['fpa']['avg_rate']:.2f} Mbps")
            print(f"    🎲 随机: {result['random']['avg_rate']:.2f} Mbps (相对FPA: {result['random_vs_fpa_improvement']:+.1f}%)")
            print(f"    🎯 优化: {result['optimized']['avg_rate']:.2f} Mbps (相对FPA: {result['optimized_vs_fpa_improvement']:+.1f}%)")
            if result['predictive']:
                print(f"    🆕 预测: {result['predictive']['avg_rate']:.2f} Mbps (相对FPA: {result['predictive_vs_fpa_improvement']:+.1f}%, 相对优化: {result['predictive_vs_optimized_improvement']:+.1f}%)")
        else:
            print(f"  ❌ 失败")
    
    # 打印汇总结果
    print(f"\n📊 发射功率测试汇总结果")
    print("=" * 130)
    print(f"{'功率(mW)':<10} {'功率(dBm)':<10} {'FPA(Mbps)':<12} {'随机(Mbps)':<12} {'优化(Mbps)':<12} {'预测(Mbps)':<12} {'随机提升%':<10} {'优化提升%':<10} {'预测提升%':<10}")
    print("-" * 130)
    
    for power_mw, result in power_test_results.items():
        power_dbm_str = f"{result['power_dbm']:.1f}"
        fpa_str = f"{result['fpa']['avg_rate']:.2f}"
        random_str = f"{result['random']['avg_rate']:.2f}"
        opt_str = f"{result['optimized']['avg_rate']:.2f}"
        
        if result['predictive']:
            pred_str = f"{result['predictive']['avg_rate']:.2f}"
            pred_imp_str = f"{result['predictive_vs_fpa_improvement']:+.1f}"
        else:
            pred_str = "N/A"
            pred_imp_str = "N/A"
        
        random_imp_str = f"{result['random_vs_fpa_improvement']:+.1f}"
        opt_imp_str = f"{result['optimized_vs_fpa_improvement']:+.1f}"
        
        print(f"{power_mw:<10} {power_dbm_str:<10} {fpa_str:<12} {random_str:<12} {opt_str:<12} {pred_str:<12} {random_imp_str:<10} {opt_imp_str:<10} {pred_imp_str:<10}")
    
    # 保存测试结果
    output_dir = "predictive_deployment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    with open(f"{output_dir}/power_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(power_test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 功率测试结果已保存至: {output_dir}/power_test_results.json")
    
    return power_test_results




def create_rate_visualizations(scenario_manager):
    """创建速率可视化图表"""
    if not scenario_manager or not scenario_manager.stats['update_rates']:
        print("❌ 没有速率数据可用于可视化")
        return
    
    # 获取数据
    update_rates = scenario_manager.stats['update_rates']
    avg_rate = np.mean(update_rates)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形 - 修改为1行3列
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # 图1：每次更新的总速率变化
    ax1.plot(range(1, len(update_rates) + 1), update_rates, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, label=f'平均值: {avg_rate:.1f} Mbps')
    ax1.set_xlabel('更新次数')
    ax1.set_ylabel('总用户速率 (Mbps)')
    ax1.set_title(f'动态场景速率变化 ({MAX_UPDATES}次更新)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加数值标注（每5个点标注一次）
    for i in range(0, len(update_rates), 5):
        ax1.annotate(f'{update_rates[i]:.0f}', 
                    (i+1, update_rates[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # 图2：最后一次更新的用户速率分布直方图
    last_rates = None
    user_distances = None
    
    if hasattr(scenario_manager, '_temp_env') and hasattr(scenario_manager, 'current_users'):
        # 计算最后一次的个人用户速率和距离
        try:
            # 重新计算最后一次的信道矩阵和用户速率
            last_rates = get_individual_user_rates(scenario_manager)
            
            # 计算用户到基站的距离
            base_station_pos = scenario_manager.params.base_station_pos
            user_distances = []
            for user in scenario_manager.current_users:
                distance = np.linalg.norm(user.position - base_station_pos)
                user_distances.append(distance)
            user_distances = np.array(user_distances)
            
            if last_rates is not None and len(last_rates) > 0:
                ax2.hist(last_rates, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                user_avg_rate = np.mean(last_rates)
                ax2.axvline(x=user_avg_rate, color='red', linestyle='--', linewidth=2, 
                           label=f'平均用户速率: {user_avg_rate:.1f} Mbps')
                ax2.set_xlabel('用户速率 (Mbps)')
                ax2.set_ylabel('用户数量')
                ax2.set_title(f'最后一次更新的用户速率分布 ({len(last_rates)}个用户)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # 添加统计信息
                ax2.text(0.05, 0.95, f'最大: {np.max(last_rates):.1f} Mbps\n最小: {np.min(last_rates):.1f} Mbps\n标准差: {np.std(last_rates):.1f} Mbps', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax2.text(0.5, 0.5, '无法获取用户速率数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('用户速率分布 (数据不可用)')
        except Exception as e:
            ax2.text(0.5, 0.5, f'计算用户速率时出错:\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('用户速率分布 (计算错误)')
    else:
        ax2.text(0.5, 0.5, '无个人用户速率数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('用户速率分布 (数据不可用)')
    
    # 图3：用户距离与速率的关系散点图
    if last_rates is not None and user_distances is not None and len(last_rates) > 0:
        try:
            # 区分地面用户和空中用户
            ground_users = []
            air_users = []
            ground_rates = []
            air_rates = []
            ground_distances = []
            air_distances = []
            
            for i, user in enumerate(scenario_manager.current_users):
                if user.type == 'vehicle':
                    ground_users.append(i)
                    ground_rates.append(last_rates[i])
                    ground_distances.append(user_distances[i])
                else:
                    air_users.append(i)
                    air_rates.append(last_rates[i])
                    air_distances.append(user_distances[i])
            
            # 绘制散点图
            if ground_rates:
                ax3.scatter(ground_distances, ground_rates, c='blue', alpha=0.6, s=50, label=f'地面用户 ({len(ground_rates)}个)')
            if air_rates:
                ax3.scatter(air_distances, air_rates, c='red', alpha=0.6, s=50, label=f'空中用户 ({len(air_rates)}个)')
            
            # 计算并绘制趋势线（所有用户）
            if len(user_distances) > 1:
                z = np.polyfit(user_distances, last_rates, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(user_distances), max(user_distances), 100)
                ax3.plot(x_trend, p(x_trend), 'g--', alpha=0.8, linewidth=2, label=f'趋势线 (斜率: {z[0]:.2f})')
            
            # 添加平均速率线
            user_avg_rate = np.mean(last_rates)
            ax3.axhline(y=user_avg_rate, color='red', linestyle=':', linewidth=1, alpha=0.7, 
                       label=f'平均速率: {user_avg_rate:.1f} Mbps')
            
            ax3.set_xlabel('用户到基站距离 (m)')
            ax3.set_ylabel('用户速率 (Mbps)')
            ax3.set_title('用户距离与速率关系')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 添加统计信息
            correlation = np.corrcoef(user_distances, last_rates)[0, 1]
            ax3.text(0.05, 0.95, f'相关系数: {correlation:.3f}\n用户数量: {len(last_rates)}\n距离范围: {min(user_distances):.0f}-{max(user_distances):.0f}m', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'计算距离-速率关系时出错:\n{str(e)}', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('距离-速率关系 (计算错误)')
    else:
        ax3.text(0.5, 0.5, '无距离-速率数据', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('距离-速率关系 (数据不可用)')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "rate_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/rate_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📊 可视化结果已保存至: {output_dir}/rate_analysis.png")
    
    # 显示统计摘要
    print(f"\n📈 速率统计摘要:")
    print(f"  总更新次数: {len(update_rates)}")
    print(f"  平均总速率: {avg_rate:.2f} Mbps")
    print(f"  最大总速率: {max(update_rates):.2f} Mbps")
    print(f"  最小总速率: {min(update_rates):.2f} Mbps")
    print(f"  速率标准差: {np.std(update_rates):.2f} Mbps")
    print(f"  速率变异系数: {np.std(update_rates)/avg_rate:.2%}")
    
    # 如果有距离-速率数据，显示相关分析
    if last_rates is not None and user_distances is not None and len(last_rates) > 0:
        correlation = np.corrcoef(user_distances, last_rates)[0, 1]
        print(f"\n📍 距离-速率关系分析:")
        print(f"  用户总数: {len(last_rates)}个")
        print(f"  平均用户速率: {np.mean(last_rates):.2f} Mbps")
        print(f"  平均距离: {np.mean(user_distances):.1f} m")
        print(f"  距离范围: {min(user_distances):.0f} - {max(user_distances):.0f} m")
        print(f"  距离-速率相关系数: {correlation:.3f}")
        
        if abs(correlation) > 0.5:
            corr_strength = "强" if abs(correlation) > 0.7 else "中等"
            corr_direction = "负" if correlation < 0 else "正"
            print(f"  相关性评价: {corr_strength}{corr_direction}相关")
        else:
            print(f"  相关性评价: 弱相关")
    
    plt.show()


class FPAManagerOld:
    """传统固定相控阵(FPA)管理器 - 4扇区固定部署（旧版本，已弃用）"""
    
    def __init__(self, params: SystemParams):
        self.params = params
        
        print("🔧 初始化传统FPA管理器...")
        
        # 基站位置
        self.base_station_pos = np.array(params.base_station_pos)
        
        # 生成4扇区FPA配置
        self._generate_fpa_configuration()
        
        # 初始化用户和统计
        from sixDMA_Environment_core_class import UserMobility
        self.current_users = UserMobility.generate_user_positions(params, seed=RANDOM_SEED)
        print(f"  用户总数: {len(self.current_users)} (地面{params.num_ground_users}个, 空中{params.num_air_users}个)")
        
        # 统计数据
        self.stats = {
            'update_rates': [],
            'total_updates': 0
        }
        
        print(f"  ✅ FPA配置完成: 4个扇区，每扇区16个天线单元 (4×4阵列)")
    
    def _generate_fpa_configuration(self):
        """生成4扇区FPA配置"""
        self.fpa_sectors = []
        
        # 4个扇区的方位角 (北、东、南、西)
        sector_azimuths = [0, 90, 180, 270]  # 度
        sector_names = ['North', 'East', 'South', 'West']
        
        # 下倾角
        downtilt_angle = 15.0  # 度
        
        # 每个扇区的4×4天线阵列参数
        array_spacing = 0.5 * self.params.lambda_wave  # 半波长间距
        
        for sector_idx, (azimuth, name) in enumerate(zip(sector_azimuths, sector_names)):
            sector_config = {
                'sector_id': sector_idx,
                'name': name,
                'azimuth': azimuth,
                'downtilt': downtilt_angle,
                'antennas': []
            }
            
            # 计算扇区中心方向
            azimuth_rad = np.radians(azimuth)
            downtilt_rad = np.radians(downtilt_angle)
            
            # 扇区主方向向量（考虑下倾）
            main_direction = np.array([
                np.cos(downtilt_rad) * np.sin(azimuth_rad),  # x
                np.cos(downtilt_rad) * np.cos(azimuth_rad),  # y
                -np.sin(downtilt_rad)  # z (向下)
            ])
            
            # 生成4×4天线阵列
            antennas = self._generate_4x4_antenna_array(
                sector_idx, azimuth, downtilt_angle, array_spacing
            )
            
            sector_config['antennas'] = antennas
            sector_config['main_direction'] = main_direction
            
            self.fpa_sectors.append(sector_config)
            
            print(f"    扇区 {name} (方位角{azimuth}°): {len(antennas)}个天线单元")
    
    def _generate_4x4_antenna_array(self, sector_id: int, azimuth: float, downtilt: float, spacing: float):
        """为单个扇区生成4×4天线阵列"""
        antennas = []
        
        # 转换角度为弧度
        azimuth_rad = np.radians(azimuth)
        downtilt_rad = np.radians(downtilt)
        
        # 计算局部坐标系
        # u向量：水平方向，垂直于主方向
        u_vec = np.array([-np.cos(azimuth_rad), np.sin(azimuth_rad), 0])
        # v向量：垂直方向，在主方向和水平面的垂直平面内
        main_dir_horizontal = np.array([np.sin(azimuth_rad), np.cos(azimuth_rad), 0])
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
                # 相对于阵列中心的偏移
                u_offset = (j - 1.5) * spacing  # 列偏移
                v_offset = (i - 1.5) * spacing  # 行偏移
                
                # 计算天线位置
                antenna_pos = (self.base_station_pos + 
                              u_offset * u_vec + 
                              v_offset * v_vec)
                
                # 创建天线对象
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
    
    def update_scenario(self, time_step: float):
        """更新FPA场景 - 与其他方法同步"""
        # 更新用户位置
        from sixDMA_Environment_core_class import UserMobility
        # 为每次更新生成确定性但不同的种子
        seed_for_update = RANDOM_SEED + self.stats['total_updates'] + 3000
        self.current_users = UserMobility.update_user_positions(self.current_users, time_step, random_seed=seed_for_update)
        
        # 计算当前FPA配置的总速率
        total_rate = self._calculate_fpa_system_rate()
        self.stats['update_rates'].append(total_rate)
        self.stats['total_updates'] += 1
        
        print(f"    FPA更新{self.stats['total_updates']}: 总用户速率 {total_rate:.2f} Mbps")
        
        return total_rate
    
    def _calculate_fpa_system_rate(self) -> float:
        """计算FPA系统的总用户速率"""
        if not self.current_users:
            return 0.0
        
        # 构建完整的FPA系统信道矩阵
        num_users = len(self.current_users)
        total_antennas = 4 * 16  # 4个扇区，每扇区16个天线
        
        H = np.zeros((total_antennas, num_users), dtype=complex)
        
        # 计算每个扇区的天线对所有用户的信道系数
        antenna_idx = 0
        for sector in self.fpa_sectors:
            for antenna in sector['antennas']:
                for user_idx, user in enumerate(self.current_users):
                    # 计算距离
                    distance = np.linalg.norm(user.position - antenna.position)
                    
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
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 使用6DMA环境中的速率计算函数
        from sixdma_environment_optimized import OptimizedSixDMAEnvironment
        temp_env = OptimizedSixDMAEnvironment(self.params)
        user_rates = temp_env._calculate_theoretical_rates_vectorized(H)
        
        return np.sum(user_rates)
    
    def run_dynamic_scenario(self, max_updates: int):
        """运行动态FPA场景"""
        print(f"\n📡 开始 {max_updates} 次FPA固定配置更新...")
        
        for update_count in range(max_updates):
            print(f"\n  --- FPA更新 {update_count + 1}/{max_updates} ---")
            
            # 执行场景更新
            time_step = 1.0  # 与其他方法保持一致
            self.update_scenario(time_step)
        
        print(f"\n📡 FPA方法完成 {max_updates} 次更新")


def run_fpa_comparison():
    """运行传统FPA固定相控阵的对比方法"""
    print("\n📡 传统FPA固定相控阵对比方法")
    print("=" * 80)
    
    # 创建系统参数
    params = SystemParams(
        num_ground_users=NUM_GROUND_USERS,
        num_air_users=NUM_AIR_USERS,
        num_surfaces=NUM_ANTENNA_SURFACES,
        environment_size=ENVIRONMENT_SIZE,
        air_height_range=AIR_HEIGHT_RANGE
    )
    
    # 创建FPA管理器
    fpa_manager = FPAManager(params)
    
    # 运行动态FPA场景
    fpa_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    
    return fpa_manager


class RandomAntennaManagerOld:
    """随机天线位置管理器 - 与贪婪方法同步运行（旧版本，已弃用）"""
    
    def __init__(self, params: SystemParams):
        self.params = params
        
        # 生成离散位置
        print("🔧 初始化随机天线管理器...")
        self.action_space = ActionSpace(params)
        self.all_positions = self.action_space.all_positions
        self.position_rotation_pairs = self.action_space.position_rotation_pairs
        
        print(f"  总离散位置数: {len(self.all_positions)}")
        print(f"  总动作空间大小: {len(self.position_rotation_pairs)} (位置 × 9种旋转)")
        
        # 初始化用户和统计
        from sixDMA_Environment_core_class import UserMobility
        self.current_users = UserMobility.generate_user_positions(params, seed=RANDOM_SEED)
        print(f"  用户总数: {len(self.current_users)} (地面{params.num_ground_users}个, 空中{params.num_air_users}个)")
        
        # 统计数据
        self.stats = {
            'update_rates': [],
            'total_updates': 0
        }
        
        # 随机种子计数器（确保每次更新都有不同的随机配置）
        self.random_seed_counter = RANDOM_SEED + 1
    
    def generate_random_antenna_config(self):
        """生成随机天线配置"""
        np.random.seed(self.random_seed_counter)
        self.random_seed_counter += 1
        
        # 随机选择位置索引（不重复）
        selected_position_indices = np.random.choice(
            len(self.all_positions), 
            size=self.params.num_surfaces, 
            replace=False
        )
        
        random_antenna_configs = []
        for i, pos_idx in enumerate(selected_position_indices):
            # 随机选择该位置的9种旋转中的一种
            rotation_idx = np.random.randint(0, 9)
            
            # 找到对应的动作
            action_idx = pos_idx * 9 + rotation_idx
            action = self.position_rotation_pairs[action_idx]
            
            config = {
                'surface_id': i,
                'position_idx': pos_idx,
                'rotation_idx': rotation_idx,
                'position': action['position'].copy(),
                'normal': action['normal'].copy(),
                'type': action['type']
            }
            random_antenna_configs.append(config)
        
        return random_antenna_configs
    
    def update_scenario(self, time_step: float):
        """更新随机场景 - 与贪婪方法同步"""
        # 更新用户位置
        from sixDMA_Environment_core_class import UserMobility
        # 为每次更新生成确定性但不同的种子
        seed_for_update = RANDOM_SEED + self.stats['total_updates'] + 2000
        self.current_users = UserMobility.update_user_positions(self.current_users, time_step, random_seed=seed_for_update)
        
        # 重新生成随机天线配置
        random_antenna_configs = self.generate_random_antenna_config()
        
        # 计算当前配置的总速率
        total_rate = calculate_antenna_config_rate(random_antenna_configs, self.current_users, self.params)
        self.stats['update_rates'].append(total_rate)
        self.stats['total_updates'] += 1
        
        print(f"    随机更新{self.stats['total_updates']}: 总用户速率 {total_rate:.2f} Mbps")
        
        return total_rate
    
    def run_dynamic_scenario(self, max_updates: int):
        """运行动态随机场景"""
        print(f"\n🎲 开始 {max_updates} 次随机天线配置更新...")
        
        for update_count in range(max_updates):
            print(f"\n  --- 随机更新 {update_count + 1}/{max_updates} ---")
            
            # 执行场景更新
            time_step = 1.0  # 与贪婪方法保持一致
            self.update_scenario(time_step)
        
        print(f"\n🎲 随机方法完成 {max_updates} 次更新")


def run_random_antenna_comparison():
    """运行随机天线位置选择的对比方法"""
    print("\n🎲 随机天线位置对比方法")
    print("=" * 80)
    
    # 创建系统参数
    params = SystemParams(
        num_ground_users=NUM_GROUND_USERS,
        num_air_users=NUM_AIR_USERS,
        num_surfaces=NUM_ANTENNA_SURFACES,
        environment_size=ENVIRONMENT_SIZE,
        air_height_range=AIR_HEIGHT_RANGE
    )
    
    # 创建随机天线管理器
    random_manager = RandomAntennaManager(params)
    
    # 运行动态随机场景
    random_manager.run_dynamic_scenario(max_updates=MAX_UPDATES)
    
    return random_manager


def calculate_antenna_config_rate_old(antenna_configs: List[Dict], users: List[User], params: SystemParams) -> float:
    """计算给定天线配置的系统总速率"""
    try:
        num_users = len(users)
        num_antennas = len(antenna_configs) * 4  # 每个表面4个天线
        
        if num_antennas == 0 or num_users == 0:
            return 0.0
        
        # 构建信道矩阵
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        antenna_idx = 0
        for config in antenna_configs:
            # 生成4天线阵列位置（复用DynamicScenarioManager中的函数）
            antenna_array_positions = generate_4_antenna_array(
                config['position'], config['normal'], params
            )
            
            for ant_pos in antenna_array_positions:
                # 创建天线对象
                antenna = Antenna(
                    surface_id=config['surface_id'],
                    global_id=antenna_idx,
                    local_id=antenna_idx % 4,
                    position=ant_pos,
                    normal=config['normal'],
                    surface_center=config['position']
                )
                
                # 计算该天线对所有用户的信道系数
                for user_idx, user in enumerate(users):
                    distance = np.linalg.norm(user.position - config['position'])
                    antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                        antenna, user, params
                    )
                    
                    if user.type == 'vehicle':
                        channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                            distance, antenna_gain_linear, antenna, user, params
                        )
                    else:
                        channel_coeff = ChannelModel.uav_channel_model_v2(
                            distance, antenna_gain_linear, user, params
                        )
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 计算理论速率（复用环境中的函数）
        from sixdma_environment_optimized import OptimizedSixDMAEnvironment
        temp_env = OptimizedSixDMAEnvironment(params)
        user_rates = temp_env._calculate_theoretical_rates_vectorized(H)
        return np.sum(user_rates)
        
    except Exception as e:
        print(f"计算天线配置速率时出错: {e}")
        return 0.0


def generate_4_antenna_array(center_pos: np.ndarray, normal: np.ndarray, params: SystemParams) -> List[np.ndarray]:
    """生成4天线矩形阵列位置（2x2配置）- 复用DynamicScenarioManager中的逻辑"""
    spacing = params.antenna_spacing
    
    # 构建局部坐标系
    if abs(normal[2]) < 0.9:
        ref_vec = np.array([0, 0, 1])
    else:
        ref_vec = np.array([1, 0, 0])
    
    # 计算局部坐标系的两个切向量
    u_vec = np.cross(normal, ref_vec)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    
    # 生成2x2阵列的4个位置（相对于中心的偏移）
    offsets = [
        (-spacing/2, -spacing/2),  # 左下
        ( spacing/2, -spacing/2),  # 右下
        (-spacing/2,  spacing/2),  # 左上
        ( spacing/2,  spacing/2)   # 右上
    ]
    
    antenna_positions = []
    for u_offset, v_offset in offsets:
        position = center_pos + u_offset * u_vec + v_offset * v_vec
        antenna_positions.append(position)
    
    return antenna_positions


def get_individual_user_rates(scenario_manager):
    """获取最后一次更新的个人用户速率"""
    try:
        if not scenario_manager.antenna_allocations or not scenario_manager.current_users:
            return None
        
        # 构建完整的系统信道矩阵
        num_users = len(scenario_manager.current_users)
        num_antennas = len(scenario_manager.antenna_allocations) * 4  # 每个表面4个天线
        
        if num_antennas == 0:
            return None
            
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        # 计算信道矩阵（复用_calculate_total_user_rate中的逻辑）
        antenna_idx = 0
        for allocation in scenario_manager.antenna_allocations:
            antenna_array_positions = scenario_manager._generate_4_antenna_array(
                allocation.antenna_position, allocation.antenna_normal
            )
            
            for ant_pos in antenna_array_positions:
                from sixDMA_Environment_core_class import Antenna, ChannelModel
                antenna = Antenna(
                    surface_id=allocation.surface_id,
                    global_id=antenna_idx,
                    local_id=antenna_idx % 4,
                    position=ant_pos,
                    normal=allocation.antenna_normal,
                    surface_center=allocation.antenna_position
                )
                
                for user_idx, user in enumerate(scenario_manager.current_users):
                    distance = np.linalg.norm(user.position - allocation.antenna_position)
                    antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                        antenna, user, scenario_manager.params
                    )
                    
                    if user.type == 'vehicle':
                        channel_coeff = ChannelModel.vehicle_channel_model_simplified(
                            distance, antenna_gain_linear, antenna, user, scenario_manager.params
                        )
                    else:
                        channel_coeff = ChannelModel.uav_channel_model_v2(
                            distance, antenna_gain_linear, user, scenario_manager.params
                        )
                    
                    H[antenna_idx, user_idx] = channel_coeff
                
                antenna_idx += 1
        
        # 计算个人用户速率
        if hasattr(scenario_manager, '_temp_env'):
            user_rates = scenario_manager._temp_env._calculate_theoretical_rates_vectorized(H)
            return user_rates
        else:
            return None
            
    except Exception as e:
        print(f"计算个人用户速率时出错: {e}")
        return None


def create_comparison_visualizations(random_manager, scenario_manager):
    """创建两种方法的对比可视化图表"""
    print("📊 生成对比可视化图表...")
    
    # 获取数据
    random_rates = random_manager.stats['update_rates']
    optimized_rates = scenario_manager.stats['update_rates']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 图1：两种方法的速率变化对比
    updates = range(1, len(random_rates) + 1)
    ax1.plot(updates, random_rates, 'r-o', linewidth=2, markersize=4, label='随机天线配置', alpha=0.8)
    ax1.plot(updates, optimized_rates, 'b-s', linewidth=2, markersize=4, label='优化天线配置', alpha=0.8)
    
    # 添加平均值线
    random_avg = np.mean(random_rates)
    optimized_avg = np.mean(optimized_rates)
    ax1.axhline(y=random_avg, color='red', linestyle='--', alpha=0.6, label=f'随机平均: {random_avg:.1f} Mbps')
    ax1.axhline(y=optimized_avg, color='blue', linestyle='--', alpha=0.6, label=f'优化平均: {optimized_avg:.1f} Mbps')
    
    ax1.set_xlabel('更新次数')
    ax1.set_ylabel('总用户速率 (Mbps)')
    ax1.set_title(f'两种方法速率对比 ({MAX_UPDATES}次更新)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2：性能提升百分比
    improvement_rates = [(opt - rand) / rand * 100 for opt, rand in zip(optimized_rates, random_rates)]
    ax2.plot(updates, improvement_rates, 'g-^', linewidth=2, markersize=4, label='性能提升%')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=np.mean(improvement_rates), color='green', linestyle='--', alpha=0.7, 
                label=f'平均提升: {np.mean(improvement_rates):.1f}%')
    
    ax2.set_xlabel('更新次数')
    ax2.set_ylabel('性能提升 (%)')
    ax2.set_title('优化方法相对随机方法的性能提升')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3：速率分布对比（箱线图）
    ax3.boxplot([random_rates, optimized_rates], 
                labels=['随机方法', '优化方法'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    
    ax3.set_ylabel('总用户速率 (Mbps)')
    ax3.set_title('两种方法速率分布对比')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax3.text(0.02, 0.98, f'随机方法:\n  均值: {np.mean(random_rates):.1f}\n  标准差: {np.std(random_rates):.1f}\n  最大: {np.max(random_rates):.1f}\n  最小: {np.min(random_rates):.1f}', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax3.text(0.52, 0.98, f'优化方法:\n  均值: {np.mean(optimized_rates):.1f}\n  标准差: {np.std(optimized_rates):.1f}\n  最大: {np.max(optimized_rates):.1f}\n  最小: {np.min(optimized_rates):.1f}', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 图4：累积性能对比
    random_cumsum = np.cumsum(random_rates)
    optimized_cumsum = np.cumsum(optimized_rates)
    ax4.plot(updates, random_cumsum, 'r-', linewidth=2, label='随机方法累积')
    ax4.plot(updates, optimized_cumsum, 'b-', linewidth=2, label='优化方法累积')
    
    ax4.set_xlabel('更新次数')
    ax4.set_ylabel('累积总用户速率 (Mbps)')
    ax4.set_title('累积性能对比')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 填充区域显示差异
    ax4.fill_between(updates, random_cumsum, optimized_cumsum, 
                     where=(optimized_cumsum >= random_cumsum), 
                     color='green', alpha=0.3, interpolate=True, label='优化优势')
    ax4.fill_between(updates, random_cumsum, optimized_cumsum, 
                     where=(optimized_cumsum < random_cumsum), 
                     color='red', alpha=0.3, interpolate=True, label='随机优势')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "rate_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 对比可视化结果已保存至: {output_dir}/method_comparison.png")
    
    plt.show()


def create_three_method_comparison(fpa_manager, random_manager, scenario_manager):
    """创建三种方法的对比可视化图表"""
    print("📊 生成三种方法对比可视化图表...")
    
    # 获取数据
    fpa_rates = fpa_manager.stats['update_rates']
    random_rates = random_manager.stats['update_rates']
    optimized_rates = scenario_manager.stats['update_rates']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比图形 (2x2布局)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 图1：三种方法的速率变化对比
    updates = range(1, len(fpa_rates) + 1)
    ax1.plot(updates, fpa_rates, 'g-^', linewidth=2, markersize=4, label='传统FPA', alpha=0.8)
    ax1.plot(updates, random_rates, 'r-o', linewidth=2, markersize=4, label='随机天线配置', alpha=0.8)
    ax1.plot(updates, optimized_rates, 'b-s', linewidth=2, markersize=4, label='优化天线配置', alpha=0.8)
    
    # 添加平均值线
    fpa_avg = np.mean(fpa_rates)
    random_avg = np.mean(random_rates)
    optimized_avg = np.mean(optimized_rates)
    ax1.axhline(y=fpa_avg, color='green', linestyle='--', alpha=0.6, label=f'FPA平均: {fpa_avg:.1f} Mbps')
    ax1.axhline(y=random_avg, color='red', linestyle='--', alpha=0.6, label=f'随机平均: {random_avg:.1f} Mbps')
    ax1.axhline(y=optimized_avg, color='blue', linestyle='--', alpha=0.6, label=f'优化平均: {optimized_avg:.1f} Mbps')
    
    ax1.set_xlabel('更新次数')
    ax1.set_ylabel('总用户速率 (Mbps)')
    ax1.set_title(f'三种方法速率对比 ({MAX_UPDATES}次更新)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2：性能提升对比（以FPA为基准）
    fpa_improvement = [0] * len(fpa_rates)  # FPA作为基准
    random_improvement = [(rand - fpa) / fpa * 100 for rand, fpa in zip(random_rates, fpa_rates)]
    optimized_improvement = [(opt - fpa) / fpa * 100 for opt, fpa in zip(optimized_rates, fpa_rates)]
    
    ax2.plot(updates, fpa_improvement, 'g-^', linewidth=2, markersize=4, label='传统FPA (基准)', alpha=0.8)
    ax2.plot(updates, random_improvement, 'r-o', linewidth=2, markersize=4, label='随机方法提升', alpha=0.8)
    ax2.plot(updates, optimized_improvement, 'b-s', linewidth=2, markersize=4, label='优化方法提升', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=np.mean(random_improvement), color='red', linestyle='--', alpha=0.7, 
                label=f'随机平均提升: {np.mean(random_improvement):.1f}%')
    ax2.axhline(y=np.mean(optimized_improvement), color='blue', linestyle='--', alpha=0.7, 
                label=f'优化平均提升: {np.mean(optimized_improvement):.1f}%')
    
    ax2.set_xlabel('更新次数')
    ax2.set_ylabel('相对FPA性能提升 (%)')
    ax2.set_title('相对传统FPA的性能提升对比')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3：速率分布对比（箱线图）
    ax3.boxplot([fpa_rates, random_rates, optimized_rates], 
                labels=['传统FPA', '随机方法', '优化方法'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    
    ax3.set_ylabel('总用户速率 (Mbps)')
    ax3.set_title('三种方法速率分布对比')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"""传统FPA:
均值: {np.mean(fpa_rates):.1f}
标准差: {np.std(fpa_rates):.1f}
最大: {np.max(fpa_rates):.1f}
最小: {np.min(fpa_rates):.1f}

随机方法:
均值: {np.mean(random_rates):.1f}
标准差: {np.std(random_rates):.1f}
最大: {np.max(random_rates):.1f}
最小: {np.min(random_rates):.1f}

优化方法:
均值: {np.mean(optimized_rates):.1f}
标准差: {np.std(optimized_rates):.1f}
最大: {np.max(optimized_rates):.1f}
最小: {np.min(optimized_rates):.1f}"""
    
    ax3.text(1.05, 0.5, stats_text, transform=ax3.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=9)
    
    # 图4：累积性能对比
    fpa_cumsum = np.cumsum(fpa_rates)
    random_cumsum = np.cumsum(random_rates)
    optimized_cumsum = np.cumsum(optimized_rates)
    
    ax4.plot(updates, fpa_cumsum, 'g-^', linewidth=2, markersize=3, label='传统FPA累积', alpha=0.8)
    ax4.plot(updates, random_cumsum, 'r-o', linewidth=2, markersize=3, label='随机方法累积', alpha=0.8)
    ax4.plot(updates, optimized_cumsum, 'b-s', linewidth=2, markersize=3, label='优化方法累积', alpha=0.8)
    
    ax4.set_xlabel('更新次数')
    ax4.set_ylabel('累积总用户速率 (Mbps)')
    ax4.set_title('三种方法累积性能对比')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 填充区域显示优势
    ax4.fill_between(updates, fpa_cumsum, optimized_cumsum, 
                     where=(optimized_cumsum >= fpa_cumsum), 
                     color='blue', alpha=0.3, interpolate=True, label='优化优于FPA')
    ax4.fill_between(updates, fpa_cumsum, random_cumsum, 
                     where=(random_cumsum >= fpa_cumsum), 
                     color='red', alpha=0.3, interpolate=True, label='随机优于FPA')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "rate_visualization_results_three_methods"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/three_methods_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 三种方法对比可视化结果已保存至: {output_dir}/three_methods_comparison.png")
    
    plt.show()
    
    # 打印详细对比统计
    print(f"\n📈 三种方法详细对比统计:")
    print("=" * 80)
    print(f"📡 传统FPA固定相控阵:")
    print(f"  平均速率: {fpa_avg:.2f} Mbps")
    print(f"  速率范围: {np.min(fpa_rates):.2f} - {np.max(fpa_rates):.2f} Mbps")
    print(f"  标准差: {np.std(fpa_rates):.2f} Mbps")
    print(f"  变异系数: {np.std(fpa_rates)/fpa_avg:.2%}")
    
    print(f"\n🎲 随机天线配置:")
    print(f"  平均速率: {random_avg:.2f} Mbps")
    print(f"  相对FPA提升: {(random_avg - fpa_avg)/fpa_avg:.2%}")
    print(f"  速率范围: {np.min(random_rates):.2f} - {np.max(random_rates):.2f} Mbps")
    print(f"  标准差: {np.std(random_rates):.2f} Mbps")
    print(f"  变异系数: {np.std(random_rates)/random_avg:.2%}")
    
    print(f"\n🎯 优化天线配置:")
    print(f"  平均速率: {optimized_avg:.2f} Mbps")
    print(f"  相对FPA提升: {(optimized_avg - fpa_avg)/fpa_avg:.2%}")
    print(f"  相对随机提升: {(optimized_avg - random_avg)/random_avg:.2%}")
    print(f"  速率范围: {np.min(optimized_rates):.2f} - {np.max(optimized_rates):.2f} Mbps")
    print(f"  标准差: {np.std(optimized_rates):.2f} Mbps")
    print(f"  变异系数: {np.std(optimized_rates)/optimized_avg:.2%}")


def main():
    """主函数：运行预测性部署测试"""
    print("🚀 预测性6DMA部署性能测试")
    print("=" * 80)
    print("请选择测试类型:")
    print("1. 📊 用户数量测试 (30-50个车辆用户)")
    print("2. ⚡ 发射功率测试 (20-120mW)")
    print("3. 🔄 运行两种测试")
    
    try:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            print(f"\n📊 用户数量测试")
            user_results = run_user_count_tests()
            print(f"\n🎉 用户数量测试完成！")
            
        elif choice == "2":
            print(f"\n⚡ 发射功率测试")
            power_results = run_power_tests()
            print(f"\n🎉 发射功率测试完成！")
            
        elif choice == "3":
            print(f"\n🔄 运行完整测试套件")
            
            # 运行用户数量测试
            print(f"\n📊 第一阶段：用户数量测试")
            user_results = run_user_count_tests()
            
            # 运行功率测试
            print(f"\n⚡ 第二阶段：发射功率测试")
            power_results = run_power_tests()
            
            print(f"\n🎉 完整测试套件完成！")
            print(f"  用户数量测试: {len(user_results)}个配置")
            print(f"  发射功率测试: {len(power_results)}个配置")
            
        else:
            print("❌ 无效选择，默认运行用户数量测试")
            user_results = run_user_count_tests()
            print(f"\n🎉 用户数量测试完成！")
            
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)


if __name__ == "__main__":
    main()