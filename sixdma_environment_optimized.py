import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

from sixDMA_Environment_core_class import SystemParams, ActionSpace, UserMobility, Surface, Antenna, ChannelModel
#from enhanced_reward_system import EnhancedRewardCalculator, TrainingDiagnostics


class OptimizedChannelCache:
    """优化的信道缓存管理器"""
    
    def __init__(self, max_cache_size: int = 20000, enable_parallel: bool = True):
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'computations': 0}
        self.max_cache_size = max_cache_size
        self.enable_parallel = enable_parallel
        self._lock = threading.Lock()
        
        # 线程池用于并行计算
        if enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=4)
        else:
            self.executor = None
    
    def _generate_cache_key(self, antenna_pos: np.ndarray, user_pos: np.ndarray, 
                          antenna_normal: np.ndarray, user_type: str) -> str:
        """生成缓存键"""
        # 对位置进行网格化以增加缓存命中率
        grid_resolution = 0.5  # 0.5米网格
        
        antenna_grid = tuple(np.round(antenna_pos / grid_resolution) * grid_resolution)
        user_grid = tuple(np.round(user_pos / grid_resolution) * grid_resolution)
        normal_grid = tuple(np.round(antenna_normal, 2))
        
        key_str = f"{antenna_grid}_{user_grid}_{normal_grid}_{user_type}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get_channel_coefficient(self, antenna: Antenna, user, params: SystemParams) -> complex:
        """获取信道系数，使用缓存优化"""
        cache_key = self._generate_cache_key(
            antenna.position, user.position, antenna.normal, user.type
        )
        
        # 检查缓存
        with self._lock:
            if cache_key in self.cache:
                self.cache_stats['hits'] += 1
                return self.cache[cache_key]
            
            self.cache_stats['misses'] += 1
        
        # 计算信道系数
        coefficient = self._compute_channel_coefficient(antenna, user, params)
        
        # 缓存结果
        with self._lock:
            if len(self.cache) >= self.max_cache_size:
                # 简单的FIFO缓存替换策略
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = coefficient
            self.cache_stats['computations'] += 1
        
        return coefficient
    
    def _compute_channel_coefficient(self, antenna: Antenna, user, params: SystemParams) -> complex:
        """计算信道系数"""
        distance = np.linalg.norm(user.position - antenna.position)
        
        # 计算天线增益
        antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
            antenna, user, params)
        
        # 根据用户类型选择信道模型
        if user.type == 'vehicle':
            return ChannelModel.vehicle_channel_model_simplified(
                distance, antenna_gain_linear, antenna, user, params)
        else:
            return ChannelModel.uav_channel_model_v2(
                distance, antenna_gain_linear, user, params)
    
    def compute_channel_matrix_batch(self, antennas: List[Antenna], users: List, 
                                   params: SystemParams) -> np.ndarray:
        """批量计算信道矩阵"""
        num_antennas = len(antennas)
        num_users = len(users)
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        if self.enable_parallel and self.executor:
            # 并行计算
            futures = []
            for u, user in enumerate(users):
                for a, antenna in enumerate(antennas):
                    future = self.executor.submit(
                        self.get_channel_coefficient, antenna, user, params
                    )
                    futures.append((a, u, future))
            
            for a, u, future in futures:
                H[a, u] = future.result()
        else:
            # 串行计算
            for u, user in enumerate(users):
                for a, antenna in enumerate(antennas):
                    H[a, u] = self.get_channel_coefficient(antenna, user, params)
        
        return H
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.cache_stats = {'hits': 0, 'misses': 0, 'computations': 0}
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
            return {
                'cache_size': len(self.cache),
                'hit_rate': hit_rate,
                'total_accesses': total,
                **self.cache_stats
            }


class VectorizedStateManager:
    """向量化状态管理器 - 智能体为中心的状态表示"""
    
    def __init__(self, params: SystemParams):
        self.params = params
        
        # 状态组件大小
        self.grid_size = params.grid_x * params.grid_y * params.grid_z
        self.neighbor_size = 8
        self.surface_state_size = params.num_surfaces * 6  # 每个表面6个特征
        
        # 总状态大小保持不变，但内部重新排序
        self.total_state_size = self.grid_size + self.neighbor_size + self.surface_state_size
        
        # 预分配数组以避免重复分配
        self.user_density_grid = np.zeros((params.grid_x, params.grid_y, params.grid_z))
        self.state_buffer = np.zeros((params.num_surfaces, self.total_state_size))
        
        print(f"启用智能体为中心的状态表示，状态大小: {self.total_state_size}")
    
    def compute_all_states_vectorized(self, users: List, surfaces: List[Surface], 
                                    occupied_positions: set, 
                                    current_position_indices: List[int],
                                    action_space_manager: ActionSpace) -> np.ndarray:
        """向量化计算所有智能体状态"""
        # 重置缓冲区
        self.user_density_grid.fill(0)
        self.state_buffer.fill(0)
        
        # 1. 计算用户密度网格（向量化）
        user_positions = np.array([user.position for user in users])
        if len(user_positions) > 0:
            self._compute_user_density_vectorized(user_positions)
        
        # 2. 为所有智能体计算状态
        for agent_id in range(self.params.num_surfaces):
            state = self._get_agent_state_optimized(
                agent_id, surfaces, occupied_positions, 
                current_position_indices, action_space_manager
            )
            self.state_buffer[agent_id] = state
        
        return self.state_buffer.copy()
    
    def _compute_user_density_vectorized(self, user_positions: np.ndarray):
        """向量化计算用户密度网格"""
        # 网格尺寸
        x_step = self.params.environment_size[0] / self.params.grid_x
        y_step = self.params.environment_size[1] / self.params.grid_y
        z_step = self.params.environment_size[2] / self.params.grid_z
        
        # 向量化计算网格索引
        grid_indices = np.floor(user_positions / np.array([x_step, y_step, z_step])).astype(int)
        
        # 限制索引范围
        grid_indices[:, 0] = np.clip(grid_indices[:, 0], 0, self.params.grid_x - 1)
        grid_indices[:, 1] = np.clip(grid_indices[:, 1], 0, self.params.grid_y - 1)
        grid_indices[:, 2] = np.clip(grid_indices[:, 2], 0, self.params.grid_z - 1)
        
        # 累加用户到网格
        for idx in grid_indices:
            self.user_density_grid[idx[0], idx[1], idx[2]] += 1
        
        # 归一化
        max_users_per_grid = 2
        self.user_density_grid = np.clip(self.user_density_grid / max_users_per_grid, 0, 1)
    
    def _get_agent_state_optimized(self, agent_id: int, surfaces: List[Surface],
                                 occupied_positions: set, current_position_indices: List[int],
                                 action_space_manager: ActionSpace) -> np.ndarray:
        """智能体为中心的状态计算 - 当前智能体信息排在前面"""
        state_components = []
        
        # 1. 用户密度网格（全局信息，所有智能体共享）
        state_components.append(self.user_density_grid.flatten())
        
        # 2. 邻近位置占用状态（个体信息）
        neighbor_occupancy = self._get_neighbor_occupancy_fast(
            agent_id, occupied_positions, current_position_indices, action_space_manager
        )
        state_components.append(neighbor_occupancy)
        
        # 3. 智能体为中心重排序的表面状态
        agent_centric_surface_state = self._get_agent_centric_surface_state(agent_id, surfaces)
        state_components.append(agent_centric_surface_state)
        
        # 拼接状态
        full_state = np.concatenate(state_components)
        
        # 确保状态维度正确
        if len(full_state) < self.total_state_size:
            padding = np.zeros(self.total_state_size - len(full_state))
            full_state = np.concatenate([full_state, padding])
        elif len(full_state) > self.total_state_size:
            full_state = full_state[:self.total_state_size]
        
        # 稀疏状态预处理：增强非零特征
        processed_state = self._preprocess_sparse_state(full_state)
        
        return processed_state.astype(np.float32)
    
    def _preprocess_sparse_state(self, state: np.ndarray) -> np.ndarray:
        """预处理稀疏状态：增强重要特征"""
        # 分离不同组件
        grid_end = self.grid_size
        neighbor_end = grid_end + self.neighbor_size
        
        user_density = state[:grid_end]
        neighbor_info = state[grid_end:neighbor_end]
        surface_info = state[neighbor_end:]
        
        # 增强用户密度信息：对非零值进行缩放
        enhanced_density = np.where(user_density > 0, 
                                  np.sqrt(user_density) + 0.1,  # 增强非零值
                                  user_density)
        
        # 邻居信息标准化
        enhanced_neighbor = neighbor_info / (np.linalg.norm(neighbor_info) + 1e-8)
        
        # 表面信息分组处理
        surface_reshaped = surface_info.reshape(-1, 6)  # 每个表面6个特征
        enhanced_surface = []
        for surface_state in surface_reshaped:
            # 位置信息（前3维）和角度信息（后3维）分别处理
            pos_info = surface_state[:3]
            angle_info = surface_state[3:]
            
            # 位置标准化
            pos_norm = pos_info / (np.linalg.norm(pos_info) + 1e-8)
            # 角度信息保持原值但增强范围
            angle_enhanced = np.tanh(angle_info * 2.0)
            
            enhanced_surface.extend(pos_norm)
            enhanced_surface.extend(angle_enhanced)
        
        enhanced_surface = np.array(enhanced_surface)
        
        # 重新组合
        processed_state = np.concatenate([
            enhanced_density,
            enhanced_neighbor, 
            enhanced_surface
        ])
        
        return processed_state
    
    def _get_agent_centric_surface_state(self, agent_id: int, surfaces: List[Surface]) -> np.ndarray:
        """获取智能体为中心重排序的表面状态：当前智能体的表面放在第一位"""
        surface_states = []
        
        # 首先添加当前智能体的表面状态（标记为自己）
        if agent_id < len(surfaces):
            current_surface = surfaces[agent_id]
            pos_normalized = current_surface.center / np.array(self.params.environment_size)
            azimuth_norm = (current_surface.azimuth + 180) / 360
            elevation_norm = (current_surface.elevation + 90) / 180
            
            # 最后一个特征设为1.0，标记这是当前智能体
            current_state = np.array([
                pos_normalized[0], pos_normalized[1], pos_normalized[2],
                azimuth_norm, elevation_norm, 1.0
            ])
            surface_states.append(current_state)
        
        # 然后按顺序添加其他智能体的表面状态（标记为其他智能体）
        for i, surface in enumerate(surfaces):
            if i != agent_id:
                pos_normalized = surface.center / np.array(self.params.environment_size)
                azimuth_norm = (surface.azimuth + 180) / 360
                elevation_norm = (surface.elevation + 90) / 180
                
                # 最后一个特征设为0.0，标记这是其他智能体
                other_state = np.array([
                    pos_normalized[0], pos_normalized[1], pos_normalized[2],
                    azimuth_norm, elevation_norm, 0.0
                ])
                surface_states.append(other_state)
        
        # 填充到固定长度（如果表面不足）
        while len(surface_states) < self.params.num_surfaces:
            surface_states.append(np.zeros(6))
        
        return np.concatenate(surface_states)
    
    def _get_neighbor_occupancy_fast(self, agent_id: int, occupied_positions: set,
                                   current_position_indices: List[int],
                                   action_space_manager: ActionSpace) -> np.ndarray:
        """快速计算邻居占用状态"""
        if agent_id >= len(current_position_indices):
            return np.zeros(self.neighbor_size)
        
        current_pos_idx = current_position_indices[agent_id]
        neighbors = action_space_manager.neighbors.get(current_pos_idx, {})
        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        occupancy = np.zeros(8)
        for i, direction in enumerate(direction_names):
            neighbor_pos_idx = neighbors.get(direction, None)
            if neighbor_pos_idx is not None and neighbor_pos_idx in occupied_positions:
                occupancy[i] = 1.0
        
        return occupancy
    
    def _get_all_surface_state_vectorized(self, surfaces: List[Surface]) -> np.ndarray:
        """向量化计算所有表面状态"""
        surface_states = []
        
        for surface in surfaces:
            # 位置归一化到[0,1]
            pos_normalized = surface.center / np.array(self.params.environment_size)
            
            # 角度归一化
            azimuth_norm = (surface.azimuth + 180) / 360
            elevation_norm = (surface.elevation + 90) / 180
            
            surface_state = np.array([
                pos_normalized[0], pos_normalized[1], pos_normalized[2],
                azimuth_norm, elevation_norm, 1.0
            ])
            surface_states.append(surface_state)
        
        # 如果表面不足，填充零
        while len(surface_states) < self.params.num_surfaces:
            surface_states.append(np.zeros(6))
        
        return np.concatenate(surface_states)


class OptimizedSixDMAEnvironment(gym.Env):
    """优化的6DMA多智能体强化学习环境"""
    
    def __init__(self, params: SystemParams, enable_cache: bool = True, enable_parallel: bool = True):
        super().__init__()
        self.params = params
        self.action_space_manager = ActionSpace(params)
        self.max_episode_steps = 50
        
        # 优化组件
        self.channel_cache = OptimizedChannelCache(enable_parallel=enable_parallel) if enable_cache else None
        self.state_manager = VectorizedStateManager(params)
        
        # 初始化用户和表面
        self.users = UserMobility.generate_user_positions(params)
        self.surfaces = []
        self.antennas = []
        
        # Episode级别的信道矩阵缓存
        self.episode_channel_matrix = None
        self.episode_channel_valid = False
        self.users_positions_cache = None
        
        # Episode统计
        self.episode_count = 0
        self.episode_rewards_history = []
        self.episode_capacities_history = []
        self.episode_losses_history = []
        self.full_reset_count = 0  # 完全重置次数统计
        
        # 增强奖励系统
        # self.enhanced_reward_calculator = EnhancedRewardCalculator(params)
        # self.training_diagnostics = TrainingDiagnostics()
        
        # 状态和动作空间
        self.state_size = self.state_manager.total_state_size
        self.local_action_size = 9 * 9
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.local_action_size,), dtype=np.float32
        )
        
        # 当前状态
        self.current_surface_position_indices = []
        self.occupied_position_indices = set()
        
        # 性能统计
        self.performance_stats = {
            'step_times': [],
            'channel_compute_times': [],
            'state_compute_times': [],
            'action_execute_times': []
        }
        
        print(f"优化6DMA环境初始化完成:")
        print(f"  缓存启用: {enable_cache}")
        print(f"  并行计算: {enable_parallel}")
        print(f"  Episode级信道缓存: 启用")
        print(f"  状态空间大小: {self.state_size}")
        print(f"  局部动作空间大小: {self.local_action_size}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境 - Episode间更新用户位置，每100个episode完全重置"""
        # 取消固定随机种子，使用更真实的随机性
        if seed is not None:
            np.random.seed(seed)
        
        # Episode计数
        self.episode_count += 1
        
        # 检查是否需要完全重置（每100个episode）
        is_full_reset = (self.episode_count % 100 == 1) or (self.episode_count == 1)
        
        if is_full_reset:
            # 完全重置：重新生成用户位置和天线位置
            self.full_reset_count += 1
            print(f"Episode {self.episode_count}: 完全重置环境 (第{self.full_reset_count}次完全重置)")
            # 使用当前时间作为随机种子，确保每次重置都产生不同的用户分布
            random_seed = int(time.time() * 1000) % 10000
            self.users = UserMobility.generate_user_positions(self.params, seed=random_seed)
            self._initialize_surfaces_optimized()
            print(f"  - 重新生成用户分布 (随机种子: {random_seed})")
            print(f"  - 重新初始化天线位置")
        else:
            # 渐进式更新：保持天线位置，更新用户位置
            self._update_users_between_episodes()
            print(f"Episode {self.episode_count}: 渐进式更新用户位置，保持天线位置")
        
        # 计算真实的episode初始容量（基于当前车辆位置和天线位置）
        true_initial_capacity = self._calculate_system_capacity_optimized()
        print(f"Episode {self.episode_count}: 真实初始容量 = {true_initial_capacity:.1f}")
        
        # 重置增强奖励系统的episode统计
        if hasattr(self, 'enhanced_reward_calculator'):
            # 如果不是第一次重置，先完成上一个episode（更新动态基准）
            if self.episode_count > 0:
                self.enhanced_reward_calculator.complete_episode()
            
            # 重置当前episode
            self.enhanced_reward_calculator.reset_episode()
        
        # 预计算Episode级别的信道矩阵
        self._precompute_episode_channel_matrix()
        
        # 重置episode统计
        self.episode_step = 0
        self.current_episode_rewards = []
        self.current_episode_capacities = []
        
        # 计算初始状态
        start_time = time.time()
        states = self.state_manager.compute_all_states_vectorized(
            self.users, self.surfaces, self.occupied_position_indices,
            self.current_surface_position_indices, self.action_space_manager
        )
        state_time = time.time() - start_time
        
        # 计算初始容量（使用缓存的信道矩阵）
        start_time = time.time()
        total_capacity = self._calculate_system_capacity_with_cached_matrix()
        capacity_time = time.time() - start_time
        
        info = self._get_info_optimized(total_capacity, state_time, capacity_time)
        info['episode_count'] = self.episode_count
        info['channel_matrix_cached'] = self.episode_channel_valid
        
        return states, info
    
    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[bool], List[bool], Dict]:
        """优化的环境步进 - Episode内用户位置固定"""
        step_start_time = time.time()
        self.episode_step += 1
        
        # 1. 执行动作（用户位置在episode内保持不变）
        action_start = time.time()
        valid_actions = self._execute_actions_optimized(actions)
        action_time = time.time() - action_start
        
        # 2. 计算奖励（使用缓存的信道矩阵）
        reward_start = time.time()
        rewards, total_rate = self._calculate_rewards_optimized(valid_actions)
        reward_time = time.time() - reward_start
        
        # 3. 收集episode统计
        self.current_episode_rewards.extend(rewards)
        self.current_episode_capacities.append(total_rate)
        
        # 4. 检查终止条件
        terminated = [False] * self.params.num_surfaces
        truncated = [self.episode_step >= self.max_episode_steps] * self.params.num_surfaces
        
        # 5. 获取新状态
        state_start = time.time()
        next_states = self.state_manager.compute_all_states_vectorized(
            self.users, self.surfaces, self.occupied_position_indices,
            self.current_surface_position_indices, self.action_space_manager
        )
        state_time = time.time() - state_start
        
        # 6. Episode结束时的统计输出
        if any(terminated) or any(truncated):
            self._log_episode_summary()
        
        # 7. 生成信息
        total_step_time = time.time() - step_start_time
        info = self._get_info_optimized(total_rate, state_time, reward_time)
        info['users_positions_fixed'] = True
        info['channel_matrix_cached'] = self.episode_channel_valid
        
        # 8. 更新性能统计
        self.performance_stats['step_times'].append(total_step_time)
        self.performance_stats['channel_compute_times'].append(reward_time)
        self.performance_stats['state_compute_times'].append(state_time)
        self.performance_stats['action_execute_times'].append(action_time)
        
        return next_states, rewards, terminated, truncated, info
    
    def _initialize_surfaces_optimized(self):
        """优化的表面初始化"""
        self.surfaces = []
        self.antennas = []
        self.occupied_position_indices = set()
        
        # 批量选择位置
        available_positions = list(range(len(self.action_space_manager.all_positions)))
        selected_positions = np.random.choice(
            available_positions, size=self.params.num_surfaces, replace=False
        )
        
        self.current_surface_position_indices = selected_positions.tolist()
        
        # 批量创建表面和天线
        center = np.array(self.params.base_station_pos)
        
        for s, pos_idx in enumerate(selected_positions):
            position = self.action_space_manager.all_positions[pos_idx]
            radial_normal = (position - center) / np.linalg.norm(position - center)
            
            surface = Surface(
                id=s, center=position.copy(), normal=radial_normal.copy(),
                azimuth=np.degrees(np.arctan2(radial_normal[1], radial_normal[0])),
                elevation=np.degrees(np.arcsin(radial_normal[2]))
            )
            self.surfaces.append(surface)
            self.occupied_position_indices.add(pos_idx)
            
            # 生成天线阵列
            antenna_positions = self._generate_surface_antenna_array_vectorized(surface)
            
            for a in range(self.params.antennas_per_surface):
                antenna = Antenna(
                    surface_id=s,
                    global_id=s * self.params.antennas_per_surface + a,
                    local_id=a,
                    position=antenna_positions[a].copy(),
                    normal=surface.normal.copy(),
                    surface_center=surface.center.copy()
                )
                self.antennas.append(antenna)
                surface.antennas.append(antenna)
    
    def _generate_surface_antenna_array_vectorized(self, surface: Surface) -> np.ndarray:
        """向量化生成表面天线阵列"""
        center = surface.center
        normal = surface.normal
        spacing = self.params.antenna_spacing
        
        # 构建局部坐标系
        if abs(normal[2]) < 0.9:
            ref_vec = np.array([0, 0, 1])
        else:
            ref_vec = np.array([1, 0, 0])
        
        u = np.cross(normal, ref_vec)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # 2x2阵列本地位置（向量化）
        local_positions = np.array([
            [-spacing/2, -spacing/2, 0],
            [spacing/2, -spacing/2, 0],
            [-spacing/2, spacing/2, 0],
            [spacing/2, spacing/2, 0]
        ])
        
        # 向量化转换到全局坐标
        local_offsets = local_positions[:, 0:1] * u + local_positions[:, 1:2] * v
        antenna_positions = center + local_offsets
        
        return antenna_positions
    
    def _execute_actions_optimized(self, actions: List[np.ndarray]) -> List[bool]:
        """优化的动作执行"""
        valid_actions = []
        new_position_indices = set()
        
        # 批量处理动作
        for agent_id, action_probs in enumerate(actions):
            if agent_id >= len(self.current_surface_position_indices):
                valid_actions.append(False)
                continue
            
            current_pos_idx = self.current_surface_position_indices[agent_id]
            local_action_indices, action_matrix = self.action_space_manager.get_local_action_space(current_pos_idx)
            
            if len(local_action_indices) == 0 or action_matrix.size == 0:
                valid_actions.append(False)
                continue
            
            # 快速动作选择
            success, target_pos_idx, selected_action = self._select_best_action_fast(
                action_probs, action_matrix, current_pos_idx
            )
            
            if success:
                self._move_surface_to_position_fast(agent_id, target_pos_idx, selected_action)
                valid_actions.append(True)
                new_position_indices.add(target_pos_idx)
            else:
                valid_actions.append(False)
                new_position_indices.add(current_pos_idx)
        
        # 更新占用位置
        self.occupied_position_indices = new_position_indices
        return valid_actions
    
    def _select_best_action_fast(self, action_probs: np.ndarray, action_matrix: np.ndarray, 
                               current_pos_idx: int) -> Tuple[bool, int, Dict]:
        """快速动作选择"""
        # 确保动作概率维度正确
        if len(action_probs) != self.local_action_size:
            action_probs = np.resize(action_probs, self.local_action_size)
        
        # 重塑为9×9矩阵
        action_matrix_probs = action_probs.reshape(9, 9)
        
        # 找到最优可用动作
        best_prob = -1
        best_position = None
        best_rotation = None
        
        for pos_i in range(min(9, action_matrix.shape[0])):
            if action_matrix[pos_i, 0] != -1:
                target_pos_idx = self.action_space_manager.position_rotation_pairs[
                    action_matrix[pos_i, 0]]['position_idx']
                
                if (target_pos_idx not in self.occupied_position_indices or 
                    target_pos_idx == current_pos_idx):
                    
                    # 使用softmax选择旋转
                    pos_probs = action_matrix_probs[pos_i, :]
                    softmax_probs = np.exp(pos_probs - np.max(pos_probs))
                    softmax_probs /= np.sum(softmax_probs)
                    
                    selected_rot = np.random.choice(9, p=softmax_probs)
                    prob = softmax_probs[selected_rot]
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_position = pos_i
                        best_rotation = selected_rot
        
        if best_position is not None and best_rotation < action_matrix.shape[1]:
            selected_action_idx = action_matrix[best_position, best_rotation]
            if selected_action_idx != -1:
                selected_action = self.action_space_manager.position_rotation_pairs[selected_action_idx]
                return True, selected_action['position_idx'], selected_action
        
        return False, current_pos_idx, {}
    
    def _move_surface_to_position_fast(self, surface_id: int, target_pos_idx: int, action: Dict):
        """快速移动表面到目标位置"""
        if surface_id >= len(self.surfaces):
            return
        
        surface = self.surfaces[surface_id]
        
        # 更新表面
        surface.center = action['position'].copy()
        surface.normal = action['normal'].copy()
        surface.azimuth = np.degrees(np.arctan2(surface.normal[1], surface.normal[0]))
        surface.elevation = np.degrees(np.arcsin(np.clip(surface.normal[2], -1, 1)))
        
        # 批量更新天线
        new_antenna_positions = self._generate_surface_antenna_array_vectorized(surface)
        for i, antenna in enumerate(surface.antennas):
            antenna.position = new_antenna_positions[i].copy()
            antenna.normal = surface.normal.copy()
            antenna.surface_center = surface.center.copy()
        
        # 更新位置索引
        self.current_surface_position_indices[surface_id] = target_pos_idx
    
    def _calculate_rewards_optimized(self, valid_actions: List[bool]) -> Tuple[List[float], float]:
        """优化的奖励计算 - 使用增强奖励系统"""
        # 使用缓存计算系统容量
        total_rate = self._calculate_system_capacity_optimized()
        
        # 使用增强奖励系统计算个体化奖励
        enhanced_rewards, reward_stats = self.enhanced_reward_calculator.calculate_enhanced_rewards(
            current_capacity=total_rate,
            antennas=self.antennas,
            users=self.users,
            valid_actions=valid_actions,
            current_positions=self.current_surface_position_indices,
            occupied_positions=self.occupied_position_indices,
            episode_step=self.episode_step
        )
        
        # 存储奖励统计信息
        if hasattr(self, 'current_reward_stats'):
            self.current_reward_stats = reward_stats
        else:
            self.current_reward_stats = reward_stats
        
        return enhanced_rewards, total_rate
    
    def _calculate_system_capacity_optimized(self) -> float:
        """优化的系统容量计算"""
        if self.channel_cache:
            # 使用缓存的批量信道矩阵计算
            H = self.channel_cache.compute_channel_matrix_batch(
                self.antennas, self.users, self.params
            )
        else:
            # 传统计算方法
            H = self._calculate_channel_matrix_traditional()
        
        # 向量化速率计算
        rates = self._calculate_theoretical_rates_vectorized(H)
        return np.sum(rates)
    
    def _calculate_channel_matrix_traditional(self) -> np.ndarray:
        """传统信道矩阵计算（无缓存）"""
        num_antennas = len(self.antennas)
        num_users = len(self.users)
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        for u, user in enumerate(self.users):
            for a, antenna in enumerate(self.antennas):
                distance = np.linalg.norm(user.position - antenna.position)
                antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
                    antenna, user, self.params)
                
                if user.type == 'vehicle':
                    H[a, u] = ChannelModel.vehicle_channel_model_simplified(
                        distance, antenna_gain_linear, antenna, user, self.params)
                else:
                    H[a, u] = ChannelModel.uav_channel_model_v2(
                        distance, antenna_gain_linear, user, self.params)
        
        return H
    
    def _calculate_theoretical_rates_vectorized(self, H: np.ndarray, transmit_power_dBm: float = 23.0) -> np.ndarray:
        """向量化计算理论速率
        
        Args:
            H: 信道矩阵 [num_antennas, num_users]
            transmit_power_dBm: 发射功率 (dBm)，默认23dBm (3GPP标准)
        """
        # 系统参数
        noise_power_dBm = -174
        bandwidth_MHz = 20
        noise_figure_dB = 7
        
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
    
    def _get_info_optimized(self, total_capacity: float, state_time: float, capacity_time: float) -> Dict:
        """获取优化的环境信息"""
        info = {
            'total_capacity': total_capacity,
            'episode_step': self.episode_step,
            'num_users': len(self.users),
            'occupied_positions': len(self.occupied_position_indices),
            'state_compute_time': state_time,
            'capacity_compute_time': capacity_time
        }
        
        # 添加缓存统计
        if self.channel_cache:
            cache_stats = self.channel_cache.get_cache_stats()
            info.update({f'cache_{k}': v for k, v in cache_stats.items()})
        
        return info
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = {}
        for key, times in self.performance_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times)
                }
        
        return stats
    
    def clear_performance_stats(self):
        """清理性能统计"""
        for key in self.performance_stats:
            self.performance_stats[key].clear()
    
    def _update_users_between_episodes(self):
        """Episode间更新用户位置 - 渐进式移动"""
        for user in self.users:
            if user.type == 'vehicle':
                # 车辆：沿道路随机移动一段距离
                move_distance = np.random.uniform(10, 30)  # 10-30米的移动
                displacement = user.direction * move_distance
                new_position = user.position + displacement
                
                # 边界处理 - 保持在道路范围内
                if user.lane in ['north_bound', 'south_bound']:
                    if new_position[1] > 300:
                        new_position[1] = new_position[1] - 300
                    elif new_position[1] < 0:
                        new_position[1] = new_position[1] + 300
                else:
                    if new_position[0] > 300:
                        new_position[0] = new_position[0] - 300
                    elif new_position[0] < 0:
                        new_position[0] = new_position[0] + 300
                
                user.position = new_position
                
            elif user.type == 'UAV':
                # UAV：随机角度和半径调整
                center = np.array([150, 150, user.height])
                current_radius = np.linalg.norm(user.position[:2] - center[:2])
                
                # 随机调整半径和角度
                radius_change = np.random.uniform(-10, 10)
                new_radius = np.clip(current_radius + radius_change, 20, 80)
                
                angle_change = np.random.uniform(-np.pi/6, np.pi/6)  # ±30度
                current_angle = np.arctan2(user.position[1] - center[1], user.position[0] - center[0])
                new_angle = current_angle + angle_change
                
                # 高度微调
                height_change = np.random.uniform(-5, 5)
                new_height = np.clip(user.height + height_change, 
                                   self.params.air_height_range[0], 
                                   self.params.air_height_range[1])
                
                # 更新位置
                user.position = np.array([
                    center[0] + new_radius * np.cos(new_angle),
                    center[1] + new_radius * np.sin(new_angle),
                    new_height
                ])
                user.height = new_height
                
                # 更新移动方向
                user.direction = np.array([-np.sin(new_angle), np.cos(new_angle), 0])
        
        # 标记信道矩阵需要更新
        self.episode_channel_valid = False
        print(f"  已更新 {len(self.users)} 个用户位置")
    
    def _precompute_episode_channel_matrix(self):
        """预计算当前episode的信道矩阵"""
        if not self.antennas or not self.users:
            self.episode_channel_valid = False
            return
        
        print(f"  正在预计算Episode {self.episode_count}的信道矩阵...")
        start_time = time.time()
        
        if self.channel_cache:
            # 使用优化的缓存计算
            self.episode_channel_matrix = self.channel_cache.compute_channel_matrix_batch(
                self.antennas, self.users, self.params
            )
        else:
            # 传统计算方法
            self.episode_channel_matrix = self._calculate_channel_matrix_traditional()
        
        computation_time = time.time() - start_time
        self.episode_channel_valid = True
        
        print(f"  信道矩阵预计算完成: {computation_time:.4f}秒, 形状: {self.episode_channel_matrix.shape}")
        
        # 缓存用户位置以检测变化
        self.users_positions_cache = np.array([user.position.copy() for user in self.users])
    
    def _calculate_system_capacity_with_cached_matrix(self) -> float:
        """使用缓存的信道矩阵计算系统容量"""
        if not self.episode_channel_valid or self.episode_channel_matrix is None:
            # 回退到实时计算
            return self._calculate_system_capacity_optimized()
        
        # 使用缓存的信道矩阵计算速率
        rates = self._calculate_theoretical_rates_vectorized(self.episode_channel_matrix)
        return np.sum(rates)
    
    def _log_episode_summary(self):
        """记录并输出Episode总结"""
        if not self.current_episode_rewards or not self.current_episode_capacities:
            return
        
        # 计算episode统计
        episode_total_reward = sum(self.current_episode_rewards)
        episode_avg_reward = np.mean(self.current_episode_rewards)
        episode_max_capacity = max(self.current_episode_capacities)
        episode_avg_capacity = np.mean(self.current_episode_capacities)
        episode_final_capacity = self.current_episode_capacities[-1]
        
        # 存储历史记录
        self.episode_rewards_history.append(episode_total_reward)
        self.episode_capacities_history.append(episode_avg_capacity)
        
        # 计算移动平均（最近10个episode）
        recent_rewards = self.episode_rewards_history[-10:]
        recent_capacities = self.episode_capacities_history[-10:]
        
        # 输出精简的episode总结
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} 总结 ({self.episode_step}/{self.max_episode_steps}步)")
        print(f"{'='*60}")
        
        # 基础统计 - 奖励与容量合并
        print(f"Episode统计:")
        print(f"  总奖励: {episode_total_reward:.4f} | 平均奖励: {episode_avg_reward:.4f}")
        print(f"  最近10轮平均奖励: {np.mean(recent_rewards):.4f}")
        
        # 容量分析 - 基于动态基准的新指标
        if hasattr(self, 'current_reward_stats') and self.current_reward_stats:
            current_cap = self.current_reward_stats.get('current_capacity', 0)
            avg_cap = self.current_reward_stats.get('episode_avg_capacity', 0)
            max_cap = self.current_reward_stats.get('episode_max_capacity', 0)
            dynamic_baseline = self.current_reward_stats.get('dynamic_baseline', 0)
            baseline_episodes_count = self.current_reward_stats.get('baseline_episodes_count', 0)
            
            print(f"容量分析:")
            print(f"  当前: {current_cap:.1f} | 平均: {avg_cap:.1f} | 最大: {max_cap:.1f} Mbps")
            print(f"  最近10轮平均: {np.mean(recent_capacities):.2f} Mbps")
            print(f"  动态基准: {dynamic_baseline:.1f} Mbps (基于{baseline_episodes_count}个episode)")
            
            # 动态基准改进指标
            step_improvement = self.current_reward_stats.get('step_improvement', 0)
            baseline_improvement = self.current_reward_stats.get('dynamic_baseline_improvement', 0)
            print(f"  改进指标: 步进 {step_improvement:+.1f}% | 相对动态基准 {baseline_improvement:+.1f}%")
        else:
            print(f"容量分析:")
            print(f"  当前: {episode_final_capacity:.1f} | 平均: {episode_avg_capacity:.1f} | 最大: {episode_max_capacity:.1f} Mbps")
            print(f"  最近10轮平均: {np.mean(recent_capacities):.2f} Mbps")
        
        # 奖励组件分析 - 基于动态基准的新组件
        if hasattr(self, 'current_reward_stats') and self.current_reward_stats:
            print(f"奖励组件 (动态基准版本):")
            abs_capacity = self.current_reward_stats.get('current_capacity', 0)/2000.0
            step_contrib = self.current_reward_stats.get('step_reward_contribution', 0)
            baseline_contrib = self.current_reward_stats.get('dynamic_baseline_reward_contribution', 0)
            trend_contrib = self.current_reward_stats.get('trend_reward_contribution', 0)
            
            print(f"  绝对容量: {abs_capacity:+.3f} | 步进改进: {step_contrib:+.3f}(40%) | 动态基准: {baseline_contrib:+.3f}(50%) | 趋势: {trend_contrib:+.3f}(10%)")
            print(f"  总改进奖励: {self.current_reward_stats.get('total_improvement_reward', 0):+.4f} | 马尔可夫性质: ✅")
            
            # 动态基准奖励分析
            dynamic_baseline = self.current_reward_stats.get('dynamic_baseline', 0)
            current_capacity = self.current_reward_stats.get('current_capacity', 0)
            if dynamic_baseline > 0:
                baseline_ratio = (current_capacity - dynamic_baseline) / dynamic_baseline
                if baseline_ratio > 0:
                    print(f"  🎯 超越基准 {baseline_ratio*100:+.1f}% → 非线性奖励: {baseline_contrib:+.3f}")
                else:
                    print(f"  📉 低于基准 {baseline_ratio*100:+.1f}% → 线性惩罚: {baseline_contrib:+.3f}")
            
            # 显示动态基准的历史信息（如果可用）
            if hasattr(self, 'enhanced_reward_calculator'):
                baseline_info = self.enhanced_reward_calculator.get_dynamic_baseline_info()
                recent_episodes = baseline_info.get('recent_episodes_avg', [])
                if len(recent_episodes) > 1:
                    recent_trend = "📈" if recent_episodes[-1] > recent_episodes[0] else "📉" if recent_episodes[-1] < recent_episodes[0] else "➡️"
                    print(f"  基准历史 ({len(recent_episodes)}个): {recent_episodes[-1]:.1f} {recent_trend} (最近vs最早: {recent_episodes[-1] - recent_episodes[0]:+.1f})")
        
        # 系统状态 - 性能+缓存+重置合并
        perf_info = []
        if self.performance_stats['step_times']:
            avg_step_time = np.mean(self.performance_stats['step_times'][-self.episode_step:])
            perf_info.append(f"步进时间: {avg_step_time:.3f}s")
        
        perf_info.append(f"信道缓存: {'是' if self.episode_channel_valid else '否'}")
        
        if self.channel_cache:
            cache_stats = self.channel_cache.get_cache_stats()
            if cache_stats['total_accesses'] > 0:
                perf_info.append(f"缓存命中率: {cache_stats['hit_rate']:.1%}")
        
        perf_info.append(f"完全重置: {self.full_reset_count}次")
        
        next_reset = 100 - (self.episode_count % 100)
        if next_reset == 100:
            next_reset = 0
        perf_info.append(f"距下次重置: {next_reset}轮")
        
        print(f"系统状态: {' | '.join(perf_info)}")
        
        print(f"{'='*60}\n")
    
    def get_episode_statistics(self) -> Dict:
        """获取episode统计信息"""
        return {
            'episode_count': self.episode_count,
            'rewards_history': self.episode_rewards_history.copy(),
            'capacities_history': self.episode_capacities_history.copy(),
            'avg_reward_last_10': np.mean(self.episode_rewards_history[-10:]) if len(self.episode_rewards_history) >= 10 else 0,
            'avg_capacity_last_10': np.mean(self.episode_capacities_history[-10:]) if len(self.episode_capacities_history) >= 10 else 0,
            'channel_matrix_cached': self.episode_channel_valid,
            'users_count': len(self.users),
            'antennas_count': len(self.antennas)
        }
