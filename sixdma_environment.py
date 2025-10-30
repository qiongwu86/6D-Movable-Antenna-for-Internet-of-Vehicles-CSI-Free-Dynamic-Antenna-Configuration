import time

import numpy as np
from typing import List, Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import copy

from sixDMA_Environment_core_class import SystemParams, ActionSpace, UserMobility, Surface, Antenna, ChannelModel


class SixDMAEnvironment(gym.Env):
    """6DMA多智能体强化学习环境"""

    def __init__(self, params: SystemParams):
        super().__init__()
        self.params = params
        self.action_space_manager = ActionSpace(params)
        self.max_episode_steps = 50

        # 初始化用户和表面
        self.users = UserMobility.generate_user_positions(params)
        self.surfaces = []
        self.antennas = []

        # 状态空间维度
        self.grid_size = params.grid_x * params.grid_y * params.grid_z  # 用户密度网格
        self.neighbor_size = 8  # 邻近位置占用状态 (8个位置)
        self.surface_state_size = params.num_surfaces * 6  # 所有表面状态(位置+角度)
        self.state_size = self.grid_size + self.neighbor_size + self.surface_state_size

        # 每个智能体的动作空间大小（固定9×9=81）
        self.local_action_size = 9 * 9

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.local_action_size,),
            dtype=np.float32
        )

        # 当前状态 - 存储位置索引而非动作索引
        self.current_surface_position_indices = []
        self.occupied_position_indices = set()

        print(f"6DMA环境初始化完成:")
        print(f"  全局动作空间大小: {len(self.action_space_manager.position_rotation_pairs)}")
        print(f"  局部动作空间大小: {self.local_action_size}")
        print(f"  每位置旋转数: {self.action_space_manager.rotations_per_position}")
        print(f"  状态空间大小: {self.state_size}")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        # 重新生成用户
        self.users = UserMobility.generate_user_positions(self.params)

        # 随机初始化表面位置
        self._initialize_surfaces()

        # 重置统计
        self.episode_step = 0

        # 计算初始状态
        states = self._get_all_states()
        total_capacity = self._calculate_system_capacity()
        info = self._get_info(total_capacity)

        return states, info

    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[bool], List[bool], Dict]:
        """环境步进"""
        # 记录 step 方法开始时间
        step_start_time = time.time()

        self.episode_step += 1

        # 更新用户位置计时
        update_users_start = time.time()
        self.users = UserMobility.update_user_positions(self.users)
        update_users_time = time.time() - update_users_start

        # 执行动作计时
        execute_actions_start = time.time()
        valid_actions = self._execute_actions(actions)
        execute_actions_time = time.time() - execute_actions_start

        # 计算奖励计时
        calculate_rewards_start = time.time()
        rewards, total_rate = self._calculate_rewards(valid_actions)
        calculate_rewards_time = time.time() - calculate_rewards_start

        # 检查终止条件计时
        check_termination_start = time.time()
        terminated = [False] * self.params.num_surfaces
        truncated = [self.episode_step >= self.max_episode_steps] * self.params.num_surfaces
        check_termination_time = time.time() - check_termination_start

        # 获取新状态计时
        get_states_start = time.time()
        next_states = self._get_all_states()
        get_states_time = time.time() - get_states_start

        # 获取信息计时
        get_info_start = time.time()
        info = self._get_info(total_rate)
        get_info_time = time.time() - get_info_start

        # 打印各部分耗时，确保时间值是数值类型
        # try:
        #     print(f"Environment Step {self.episode_step}:")
        #     print(f"  Update user positions time: {float(update_users_time):.4f}s")
        #     print(f"  Execute actions time: {float(execute_actions_time):.4f}s")
        #     print(f"  Calculate rewards time: {float(calculate_rewards_time):.4f}s")
        #     print(f"  Check termination time: {float(check_termination_time):.4f}s")
        #     print(f"  Get states time: {float(get_states_time):.4f}s")
        #     print(f"  Get info time: {float(get_info_time):.4f}s")
        #     print(f"  Total step time: {float(time.time() - step_start_time):.4f}s")
        # except (TypeError, ValueError) as e:
        #     print(f"Error formatting time values: {e}")
        #     print(f"update_users_time: {update_users_time}, type: {type(update_users_time)}")
            # Add similar debug prints for other time variables if needed

        return next_states, rewards, terminated, truncated, info

    def _initialize_surfaces(self):
        """初始化表面位置 - 只选择位置，不选择旋转"""
        self.surfaces = []
        self.antennas = []
        self.occupied_position_indices = set()

        # 随机选择不重复的位置索引
        available_position_indices = list(range(len(self.action_space_manager.all_positions)))
        selected_position_indices = np.random.choice(
            available_position_indices,
            size=self.params.num_surfaces,
            replace=False
        )

        self.current_surface_position_indices = selected_position_indices.tolist()

        for s, pos_idx in enumerate(selected_position_indices):
            position = self.action_space_manager.all_positions[pos_idx]

            # 默认使用径向方向（rotation_idx=0）
            center = np.array(self.params.base_station_pos)
            radial_normal = (position - center) / np.linalg.norm(position - center)

            surface = Surface(
                id=s,
                center=position.copy(),
                normal=radial_normal.copy(),
                azimuth=0.0,
                elevation=0.0
            )

            # 计算方位角和俯仰角
            normal = surface.normal
            surface.azimuth = np.degrees(np.arctan2(normal[1], normal[0]))
            surface.elevation = np.degrees(np.arcsin(normal[2]))

            self.surfaces.append(surface)
            self.occupied_position_indices.add(pos_idx)

            # 生成表面上的天线
            antenna_positions = self._generate_surface_antenna_array(surface)

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

    def _generate_surface_antenna_array(self, surface: Surface) -> np.ndarray:
        """在表面上生成2x2天线阵列"""
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
        v = v / np.linalg.norm(v)

        # 2x2阵列本地位置
        local_positions = np.array([
            [-spacing / 2, -spacing / 2, 0],
            [spacing / 2, -spacing / 2, 0],
            [-spacing / 2, spacing / 2, 0],
            [spacing / 2, spacing / 2, 0]
        ])

        # 转换到全局坐标
        antenna_positions = np.zeros((4, 3))
        for i in range(4):
            local_offset = local_positions[i, 0] * u + local_positions[i, 1] * v
            antenna_positions[i] = center + local_offset

        return antenna_positions

    def _get_all_states(self) -> np.ndarray:
        """获取所有智能体的状态"""
        states = []

        for surface_id in range(self.params.num_surfaces):
            state = self._get_agent_state(surface_id)
            states.append(state)

        return np.array(states, dtype=np.float32)

    def _get_agent_state(self, surface_id: int) -> np.ndarray:
        """获取单个智能体的状态"""
        state_components = []

        # 1. 用户密度网格
        user_density_grid = self._calculate_user_density_grid()
        state_components.append(user_density_grid.flatten())

        # 2. 邻近位置占用状态
        neighbor_occupancy = self._get_neighbor_occupancy(surface_id)
        state_components.append(neighbor_occupancy)

        # 3. 所有表面状态
        all_surface_state = self._get_all_surface_state()
        state_components.append(all_surface_state)

        # 拼接状态
        full_state = np.concatenate(state_components)

        # 确保状态维度正确
        if len(full_state) < self.state_size:
            padding = np.zeros(self.state_size - len(full_state))
            full_state = np.concatenate([full_state, padding])
        elif len(full_state) > self.state_size:
            full_state = full_state[:self.state_size]

        return full_state.astype(np.float32)

    def _calculate_user_density_grid(self) -> np.ndarray:
        """计算用户密度网格"""
        grid = np.zeros((self.params.grid_x, self.params.grid_y, self.params.grid_z))

        # 网格尺寸
        x_step = self.params.environment_size[0] / self.params.grid_x
        y_step = self.params.environment_size[1] / self.params.grid_y
        z_step = self.params.environment_size[2] / self.params.grid_z

        for user in self.users:
            x, y, z = user.position

            # 计算网格索引
            x_idx = min(int(x / x_step), self.params.grid_x - 1)
            y_idx = min(int(y / y_step), self.params.grid_y - 1)
            z_idx = min(int(z / z_step), self.params.grid_z - 1)

            # 确保索引在有效范围内
            x_idx = max(0, x_idx)
            y_idx = max(0, y_idx)
            z_idx = max(0, z_idx)

            grid[x_idx, y_idx, z_idx] += 1

        # 归一化（每个网格最多用户数假设为10）
        max_users_per_grid = 5
        grid = np.clip(grid / max_users_per_grid, 0, 1)

        return grid

    def _get_neighbor_occupancy(self, surface_id: int) -> np.ndarray:
        """获取相邻8个位置的占用状态"""
        if surface_id >= len(self.current_surface_position_indices):
            return np.zeros(self.neighbor_size)

        current_pos_idx = self.current_surface_position_indices[surface_id]

        # 获取邻居关系
        neighbors = self.action_space_manager.neighbors.get(current_pos_idx, {})
        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        occupancy = np.zeros(8)

        for i, direction in enumerate(direction_names):
            neighbor_pos_idx = neighbors.get(direction, None)
            if neighbor_pos_idx is not None and neighbor_pos_idx in self.occupied_position_indices:
                occupancy[i] = 1.0

        return occupancy

    def _get_all_surface_state(self) -> np.ndarray:
        """获取所有表面状态"""
        surface_states = []

        for surface in self.surfaces:
            # 位置归一化到[0,1]
            pos_normalized = surface.center / np.array(self.params.environment_size)

            # 角度归一化
            azimuth_norm = (surface.azimuth + 180) / 360  # [-180,180] -> [0,1]
            elevation_norm = (surface.elevation + 90) / 180  # [-90,90] -> [0,1]

            surface_state = np.concatenate([
                pos_normalized,
                [azimuth_norm, elevation_norm],
                [1.0]  # 激活状态
            ])

            surface_states.append(surface_state)

        # 如果表面不足，填充零
        while len(surface_states) < self.params.num_surfaces:
            surface_states.append(np.zeros(6))

        return np.concatenate(surface_states)

    def _execute_actions(self, actions: List[np.ndarray]) -> List[bool]:
        """执行动作，使用9×9局部动作空间"""
        valid_actions = []
        new_position_indices = set()

        for agent_id, action_probs in enumerate(actions):
            if agent_id >= len(self.current_surface_position_indices):
                valid_actions.append(False)
                continue

            current_pos_idx = self.current_surface_position_indices[agent_id]

            # 获取局部动作空间
            local_action_indices, action_matrix = self.action_space_manager.get_local_action_space(current_pos_idx)

            if len(local_action_indices) == 0 or action_matrix.size == 0:
                valid_actions.append(False)
                continue

            # 确保动作概率维度正确
            if len(action_probs) != self.local_action_size:
                action_probs = action_probs[:self.local_action_size]
                if len(action_probs) < self.local_action_size:
                    padding = np.zeros(self.local_action_size - len(action_probs))
                    action_probs = np.concatenate([action_probs, padding])

            # 重塑为9×9矩阵
            action_matrix_probs = action_probs.reshape(9, 9)

            # 寻找最优动作
            selected_position = None
            selected_rotation = None
            max_prob = -1

            for pos_i in range(9):
                if pos_i < action_matrix.shape[0]:
                    # 检查该位置是否可用
                    if action_matrix[pos_i, 0] != -1:
                        target_pos_idx = self.action_space_manager.position_rotation_pairs[action_matrix[pos_i, 0]][
                            'position_idx']

                        if target_pos_idx not in self.occupied_position_indices or target_pos_idx == current_pos_idx:
                            # 位置可用，在该位置的旋转中选择
                            pos_probs = action_matrix_probs[pos_i, :]
                            exp_probs = np.exp(pos_probs - np.max(pos_probs))
                            softmax_probs = exp_probs / np.sum(exp_probs)

                            selected_rot = np.random.choice(9, p=softmax_probs)
                            prob = softmax_probs[selected_rot]

                            if prob > max_prob:
                                max_prob = prob
                                selected_position = pos_i
                                selected_rotation = selected_rot

            if selected_position is not None and selected_rotation < action_matrix.shape[1]:
                # 执行选中的动作
                selected_action_idx = action_matrix[selected_position, selected_rotation]
                if selected_action_idx != -1:
                    selected_action = self.action_space_manager.position_rotation_pairs[selected_action_idx]
                    target_pos_idx = selected_action['position_idx']

                    self._move_surface_to_position(agent_id, target_pos_idx, selected_action)
                    valid_actions.append(True)
                    new_position_indices.add(target_pos_idx)
                else:
                    # 无效动作索引
                    valid_actions.append(False)
                    new_position_indices.add(current_pos_idx)
            else:
                # 没有可用位置，保持原位
                valid_actions.append(False)
                new_position_indices.add(current_pos_idx)

        # 更新占用位置
        self.occupied_position_indices = new_position_indices

        return valid_actions

    def _move_surface_to_position(self, surface_id: int, target_pos_idx: int, action: Dict):
        """将表面移动到目标位置"""
        if surface_id >= len(self.surfaces):
            return

        surface = self.surfaces[surface_id]

        # 更新表面位置和方向
        surface.center = action['position'].copy()
        surface.normal = action['normal'].copy()

        # 重新计算角度
        normal = surface.normal
        surface.azimuth = np.degrees(np.arctan2(normal[1], normal[0]))
        surface.elevation = np.degrees(np.arcsin(np.clip(normal[2], -1, 1)))

        # 重新生成天线位置
        new_antenna_positions = self._generate_surface_antenna_array(surface)

        # 更新天线
        for i, antenna in enumerate(surface.antennas):
            antenna.position = new_antenna_positions[i].copy()
            antenna.normal = surface.normal.copy()
            antenna.surface_center = surface.center.copy()

        # 更新位置索引
        self.current_surface_position_indices[surface_id] = target_pos_idx

    def _calculate_rewards(self, valid_actions: List[bool]) -> List[float]:
        """计算奖励"""
        rewards = []

        # 计算系统总容量
        total_rate = self._calculate_system_capacity()

        # 基础奖励：系统容量
        base_reward = total_rate / 1000.0  # 归一化到合理范围

        for agent_id, is_valid in enumerate(valid_actions):
            reward = base_reward

            # 无效动作惩罚
            if not is_valid:
                reward -= 10.0

            rewards.append(reward)

        return rewards, total_rate

    def _calculate_system_capacity(self) -> float:
        """计算系统容量"""
        # 构建信道矩阵
        H = self._calculate_channel_matrix()

        # 计算理论速率
        rates = self._calculate_theoretical_rates(H)

        return np.sum(rates)

    def _calculate_channel_matrix(self) -> np.ndarray:
        """计算信道矩阵"""
        num_antennas = len(self.antennas)
        num_users = len(self.users)

        H = np.zeros((num_antennas, num_users), dtype=complex)

        for u, user in enumerate(self.users):
            for a, antenna in enumerate(self.antennas):
                H[a, u] = self._calculate_channel_coefficient(antenna, user)

        return H

    def _calculate_channel_coefficient(self, antenna: Antenna, user) -> complex:
        """计算信道系数"""
        distance = np.linalg.norm(user.position - antenna.position)

        # 计算天线增益
        antenna_gain_linear = ChannelModel.calculate_3gpp_antenna_gain(
            antenna, user, self.params)

        # 根据用户类型选择信道模型
        if user.type == 'vehicle':
            return ChannelModel.vehicle_channel_model_simplified(
                distance, antenna_gain_linear, antenna, user, self.params)
        else:
            return ChannelModel.uav_channel_model_v2(
                distance, antenna_gain_linear, user, self.params)

    def _calculate_theoretical_rates(self, H: np.ndarray) -> np.ndarray:
        """计算理论速率"""
        # 系统参数
        noise_power_dBm = -174
        bandwidth_MHz = 20
        noise_figure_dB = 7

        # 噪声功率
        total_noise_dBm = noise_power_dBm + 10 * np.log10(bandwidth_MHz * 1e6) + noise_figure_dB
        noise_power_W = 10 ** ((total_noise_dBm - 30) / 10)

        # 发射功率
        transmit_power_dBm = 23  # 3GPP标准: 车辆和UAV上行链路功率
        transmit_power_W = 10 ** ((transmit_power_dBm - 30) / 10)

        num_users = H.shape[1]
        rates = np.zeros(num_users)

        for k in range(num_users):
            h_k = H[:, k]

            # 信号功率
            signal_power = transmit_power_W * np.abs(np.vdot(h_k, h_k))

            # 简化干扰计算
            interference_power = 0
            for j in range(num_users):
                if j != k:
                    h_j = H[:, j]
                    if np.vdot(h_j, h_j) != 0:
                        interference_power += (transmit_power_W *
                                               np.abs(np.vdot(h_k, h_j)) ** 2 /
                                               (np.vdot(h_j, h_j) + 1e-10))

            # SINR
            sinr = np.abs(signal_power) / (interference_power + noise_power_W)
            sinr = np.clip(sinr, 1e-6, 1e6)

            # 香农容量
            capacity_bps_hz = np.log2(1 + sinr)
            rates[k] = capacity_bps_hz * bandwidth_MHz

        return rates

    def _get_info(self, total_capacity) -> Dict:
        """获取环境信息"""

        return {
            'total_capacity': total_capacity,
            'episode_step': self.episode_step,
            'num_users': len(self.users),
            'occupied_positions': len(self.occupied_position_indices)
        }