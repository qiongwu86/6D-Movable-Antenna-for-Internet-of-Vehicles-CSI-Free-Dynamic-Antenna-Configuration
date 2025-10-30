import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull, KDTree
import random
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import math


@dataclass
class SystemParams:
    """系统参数配置"""
    environment_size: Tuple[int, int, int] = (300, 300, 100)
    base_station_pos: Tuple[float, float, float] = (150.0, 150.0, 20.0)
    antenna_sphere_radius: float = 1.0
    all_positions: int = 128
    num_surfaces: int = 32
    antennas_per_surface: int = 4
    antenna_spacing: float = 0.05
    num_ground_users: int = 20
    num_air_users: int = 5
    air_height_range: Tuple[float, float] = (50.0, 100.0)
    fc: float = 3.5e9
    c: float = 3e8

    # 网格化参数
    grid_x: int = 10
    grid_y: int = 10
    grid_z: int = 2

    # 3GPP标准天线参数
    antenna_phi_3dB: float = 65.0
    antenna_theta_3dB: float = 25.0
    antenna_G_max: float = 8.0
    antenna_G_s: float = 30.0
    antenna_G_v: float = 30.0

    def __post_init__(self):
        self.lambda_wave = self.c / self.fc

        # 建筑物位置
        self.buildings = np.array([
            [130, 130, 20],
            [170, 130, 25],
            [130, 170, 22],
            [170, 170, 18]
        ])


@dataclass
class User:
    """用户类"""
    id: int
    type: str  # 'vehicle' or 'UAV'
    position: np.ndarray
    height: float
    velocity: float = 0.0
    direction: np.ndarray = None
    lane: str = None
    # UAV轨迹参数（极坐标）
    orbit_center: np.ndarray = None  # 轨道中心
    orbit_radius: float = 0.0        # 轨道半径
    orbit_angle: float = 0.0         # 当前角度
    # UAV高度变化参数（独立于水平运动）
    target_height: float = 0.0       # 目标高度
    vertical_velocity: float = 0.0   # 垂直速度 (m/s，正值向上，负值向下)

    def __post_init__(self):
        if self.direction is None:
            self.direction = np.array([0.0, 0.0, 0.0])
        if self.orbit_center is None:
            self.orbit_center = np.array([0.0, 0.0, 0.0])


@dataclass
class Surface:
    """天线表面类"""
    id: int
    center: np.ndarray
    normal: np.ndarray
    azimuth: float
    elevation: float
    antennas: List = None

    def __post_init__(self):
        if self.antennas is None:
            self.antennas = []


@dataclass
class Antenna:
    """天线类"""
    surface_id: int
    global_id: int
    local_id: int
    position: np.ndarray
    normal: np.ndarray
    surface_center: np.ndarray


class ActionSpace:
    """基于经纬线网格的动作空间类 - 修复极点第一层与中间区域重叠问题"""

    def __init__(self, params: SystemParams):
        self.params = params
        self.antenna_spacing = params.antenna_spacing
        self.d_min = 2 * self.antenna_spacing  # 最小距离约束

        # 生成经纬线网格位置
        self.grid_positions, self.grid_info = self._generate_lat_lon_grid()
        self.all_positions = self.grid_positions

        # 每个位置9个旋转
        self.rotations_per_position = 9

        # 构建邻居关系
        self._build_neighbor_relationships()

        # 生成动作空间
        self.position_rotation_pairs = self._generate_action_space()

        print(f"经纬线网格动作空间:")
        print(f"  网格位置数: {len(self.all_positions)}")
        print(f"  纬线圈数: {self.grid_info['num_lat_circles']}")
        print(f"  每个纬线圈经线数: {self.grid_info['meridians_per_circle']}")
        print(f"  最小距离约束 d_min: {self.d_min:.4f}")
        print(f"  每位置旋转数: {self.rotations_per_position}")
        print(f"  总动作数: {len(self.position_rotation_pairs)}")

    def _generate_lat_lon_grid(self) -> Tuple[np.ndarray, Dict]:
        """生成经纬线网格位置 - 新设计：固定12条经线，基于正12边形的纬线圈"""
        center = np.array(self.params.base_station_pos)
        radius = self.params.antenna_sphere_radius
        
        # 固定12条经线，角度间距为30°
        NUM_MERIDIANS = 12
        meridian_angles = [i * 2 * np.pi / NUM_MERIDIANS for i in range(NUM_MERIDIANS)]
        
        print(f"使用固定的12条经线，角度间距: {360/NUM_MERIDIANS}°")
        
        positions = []
        grid_info = {
            'num_lat_circles': 0,
            'meridians_per_circle': [],
            'pole_indices': {'north': None, 'south': None},
            'position_to_grid': {},  # position_idx -> (lat_idx, lon_idx)
            'grid_to_position': {}   # (lat_idx, lon_idx) -> position_idx
        }
        
        position_idx = 0
        lat_circles = []
        
        # 1. 北极点
        north_pole = center + np.array([0, 0, radius])
        positions.append(north_pole)
        grid_info['position_to_grid'][position_idx] = (0, 0)
        grid_info['grid_to_position'][(0, 0)] = position_idx
        grid_info['pole_indices']['north'] = position_idx
        grid_info['meridians_per_circle'].append(1)
        lat_circles.append(np.pi/2)  # 北极纬度
        position_idx += 1
        
        # 2. 确定第一条纬线圈（北）
        # 正12边形外接圆半径 = 边长 / (2 * sin(π/12))
        # 边长 = d_min，所以第一纬线圈半径 = d_min / (2 * sin(π/12))
        first_circle_radius = self.d_min / (2 * np.sin(np.pi / NUM_MERIDIANS))
        
        # 对应的纬度角
        if first_circle_radius <= radius:
            first_lat_angle = np.arccos(first_circle_radius / radius)
            print(f"第一纬线圈: 半径={first_circle_radius:.4f}, 纬度角={np.degrees(first_lat_angle):.1f}°")
            
            # 生成第一纬线圈的12个点
            for i, lon_angle in enumerate(meridian_angles):
                x = radius * np.cos(first_lat_angle) * np.cos(lon_angle)
                y = radius * np.cos(first_lat_angle) * np.sin(lon_angle)
                z = radius * np.sin(first_lat_angle)
                
                position = center + np.array([x, y, z])
                positions.append(position)
                grid_info['position_to_grid'][position_idx] = (1, i)
                grid_info['grid_to_position'][(1, i)] = position_idx
                position_idx += 1
            
            grid_info['meridians_per_circle'].append(NUM_MERIDIANS)
            lat_circles.append(first_lat_angle)
        
        # 3. 生成中间纬线圈
        # 从第一纬线圈开始，每次向南移动固定角度距离
        current_lat_angle = first_lat_angle
        lat_idx = 2
        
        angular_step = self.d_min / radius  # 角度步长
        
        while current_lat_angle > angular_step:
            current_lat_angle -= angular_step
            current_circle_radius = radius * np.cos(current_lat_angle)
            
            # 检查当前纬线圈能否容纳12个满足距离约束的点
            if current_circle_radius > 1e-6:
                # 12个点之间的最小角距离
                min_angular_distance = 2 * np.pi / NUM_MERIDIANS
                # 对应的最小弧长距离
                min_arc_distance = min_angular_distance * current_circle_radius
                
                if min_arc_distance >= self.d_min * 0.95:  # 允许5%的误差
                    # 可以放置12个点
                    for i, lon_angle in enumerate(meridian_angles):
                        x = radius * np.cos(current_lat_angle) * np.cos(lon_angle)
                        y = radius * np.cos(current_lat_angle) * np.sin(lon_angle)
                        z = radius * np.sin(current_lat_angle)
                        
                        position = center + np.array([x, y, z])
                        positions.append(position)
                        grid_info['position_to_grid'][position_idx] = (lat_idx, i)
                        grid_info['grid_to_position'][(lat_idx, i)] = position_idx
                        position_idx += 1
                    
                    grid_info['meridians_per_circle'].append(NUM_MERIDIANS)
                    lat_circles.append(current_lat_angle)
                    lat_idx += 1
                else:
                    print(f"纬度{np.degrees(current_lat_angle):.1f}°处圆周太小，跳过")
                    break
            else:
                break
        
        # 4. 赤道（如果还没有的话）
        if current_lat_angle > 0.1:  # 如果还没到赤道附近
            equator_circle_radius = radius
            min_arc_distance = (2 * np.pi / NUM_MERIDIANS) * equator_circle_radius
            
            if min_arc_distance >= self.d_min * 0.95:
                for i, lon_angle in enumerate(meridian_angles):
                    x = radius * np.cos(lon_angle)
                    y = radius * np.sin(lon_angle)
                    z = 0
                    
                    position = center + np.array([x, y, z])
                    positions.append(position)
                    grid_info['position_to_grid'][position_idx] = (lat_idx, i)
                    grid_info['grid_to_position'][(lat_idx, i)] = position_idx
                    position_idx += 1
                
                grid_info['meridians_per_circle'].append(NUM_MERIDIANS)
                lat_circles.append(0.0)
                lat_idx += 1
        
        # 5. 南半球纬线圈（对称生成）
        north_circles = [angle for angle in lat_circles[1:] if angle > 0]  # 排除极点和赤道
        for north_angle in reversed(north_circles):
            south_angle = -north_angle
            
            for i, lon_angle in enumerate(meridian_angles):
                x = radius * np.cos(south_angle) * np.cos(lon_angle)
                y = radius * np.cos(south_angle) * np.sin(lon_angle)
                z = radius * np.sin(south_angle)
                
                position = center + np.array([x, y, z])
                positions.append(position)
                grid_info['position_to_grid'][position_idx] = (lat_idx, i)
                grid_info['grid_to_position'][(lat_idx, i)] = position_idx
                position_idx += 1
            
            grid_info['meridians_per_circle'].append(NUM_MERIDIANS)
            lat_circles.append(south_angle)
            lat_idx += 1
        
        # 6. 南极点
        south_pole = center + np.array([0, 0, -radius])
        positions.append(south_pole)
        grid_info['position_to_grid'][position_idx] = (lat_idx, 0)
        grid_info['grid_to_position'][(lat_idx, 0)] = position_idx
        grid_info['pole_indices']['south'] = position_idx
        grid_info['meridians_per_circle'].append(1)
        lat_circles.append(-np.pi/2)
        
        grid_info['num_lat_circles'] = lat_idx + 1
        
        print(f"生成完成，总共 {grid_info['num_lat_circles']} 个纬线圈，{len(positions)} 个网格点")
        print(f"纬线圈经线数分布: {grid_info['meridians_per_circle']}")

        return np.array(positions), grid_info
    def _build_neighbor_relationships(self):
        """构建邻居关系：米字形8个方向，极点特殊处理"""
        self.neighbors = {}  # position_idx -> {'direction_name': neighbor_position_idx}

        # 8个方向的定义
        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        for pos_idx in range(len(self.all_positions)):
            lat_idx, lon_idx = self.grid_info['position_to_grid'][pos_idx]
            neighbors = {}

            # 检查是否为极点
            if pos_idx == self.grid_info['pole_indices']['north']:
                # 北极点：邻居为北第一层圆环上的8个点
                neighbors = self._get_pole_neighbors(pos_idx, 'north')
            elif pos_idx == self.grid_info['pole_indices']['south']:
                # 南极点：邻居为南第一层圆环上的8个点
                neighbors = self._get_pole_neighbors(pos_idx, 'south')
            else:
                # 非极点：正常的8邻居查找
                neighbor_coords = self._get_8_neighbors(lat_idx, lon_idx)

                for i, (neighbor_lat, neighbor_lon) in enumerate(neighbor_coords):
                    if (neighbor_lat, neighbor_lon) in self.grid_info['grid_to_position']:
                        neighbor_pos_idx = self.grid_info['grid_to_position'][(neighbor_lat, neighbor_lon)]
                        neighbors[direction_names[i]] = neighbor_pos_idx
                    else:
                        neighbors[direction_names[i]] = None  # 边界外

            self.neighbors[pos_idx] = neighbors

    def _get_pole_neighbors(self, pole_pos_idx: int, pole_type: str) -> Dict[str, int]:
        """获取极点的8个邻居 - 适配新的12经线网格"""
        neighbors = {}
        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        if pole_type == 'north':
            # 北极点的邻居在第一纬线圈（lat_idx=1）
            neighbor_lat_idx = 1
        else:
            # 南极点的邻居在倒数第二纬线圈
            neighbor_lat_idx = self.grid_info['num_lat_circles'] - 2

        # 获取该圆环上的所有点（应该有12个点）
        neighbor_circle_points = []
        if (neighbor_lat_idx >= 0 and 
            neighbor_lat_idx < self.grid_info['num_lat_circles'] and 
            self.grid_info['meridians_per_circle'][neighbor_lat_idx] == 12):
            
            for lon_idx in range(12):
                if (neighbor_lat_idx, lon_idx) in self.grid_info['grid_to_position']:
                    point_idx = self.grid_info['grid_to_position'][(neighbor_lat_idx, lon_idx)]
                    neighbor_circle_points.append((point_idx, lon_idx))

        # 从12个点中选择8个作为8个方向的邻居
        if len(neighbor_circle_points) >= 8:
            # 选择对应8个方向的邻居点（每30°选一个，跳过4个）
            # N(0°), NE(45°), E(90°), SE(135°), S(180°), SW(225°), W(270°), NW(315°)
            # 对应经线索引: 0, 1.5→2, 3, 4.5→5, 6, 7.5→8, 9, 10.5→11
            meridian_indices = [0, 2, 3, 5, 6, 8, 9, 11]  # 8个均匀分布的经线索引
            
            for i, direction in enumerate(direction_names):
                if i < len(meridian_indices):
                    meridian_idx = meridian_indices[i]
                    if meridian_idx < len(neighbor_circle_points):
                        neighbors[direction] = neighbor_circle_points[meridian_idx][0]
                    else:
                        neighbors[direction] = None
                else:
                    neighbors[direction] = None
        else:
            # 如果点数不足，按顺序分配
            for i, direction in enumerate(direction_names):
                if i < len(neighbor_circle_points):
                    neighbors[direction] = neighbor_circle_points[i][0]
                else:
                    neighbors[direction] = None

        return neighbors

    def _get_8_neighbors(self, lat_idx: int, lon_idx: int) -> List[Tuple[int, int]]:
        """获取8个邻居的网格坐标（非极点情况）"""
        num_lat_circles = self.grid_info['num_lat_circles']
        num_meridians_current = self.grid_info['meridians_per_circle'][lat_idx]

        neighbors = []

        # 北方邻居 (N)
        if lat_idx > 0:
            neighbor_lat = lat_idx - 1
            num_meridians_north = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_north == 1:
                neighbor_lon = 0  # 极点
            else:
                neighbor_lon = int(lon_idx * num_meridians_north / num_meridians_current) % num_meridians_north
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))  # 无效邻居

        # 东北方邻居 (NE)
        if lat_idx > 0:
            neighbor_lat = lat_idx - 1
            num_meridians_north = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_north == 1:
                neighbor_lon = 0
            else:
                neighbor_lon = int((lon_idx + 0.5) * num_meridians_north / num_meridians_current) % num_meridians_north
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 东方邻居 (E)
        if num_meridians_current > 1:
            neighbor_lat = lat_idx
            neighbor_lon = (lon_idx + 1) % num_meridians_current
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 东南方邻居 (SE)
        if lat_idx < num_lat_circles - 1:
            neighbor_lat = lat_idx + 1
            num_meridians_south = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_south == 1:
                neighbor_lon = 0
            else:
                neighbor_lon = int((lon_idx + 0.5) * num_meridians_south / num_meridians_current) % num_meridians_south
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 南方邻居 (S)
        if lat_idx < num_lat_circles - 1:
            neighbor_lat = lat_idx + 1
            num_meridians_south = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_south == 1:
                neighbor_lon = 0
            else:
                neighbor_lon = int(lon_idx * num_meridians_south / num_meridians_current) % num_meridians_south
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 西南方邻居 (SW)
        if lat_idx < num_lat_circles - 1:
            neighbor_lat = lat_idx + 1
            num_meridians_south = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_south == 1:
                neighbor_lon = 0
            else:
                neighbor_lon = int((lon_idx - 0.5) * num_meridians_south / num_meridians_current) % num_meridians_south
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 西方邻居 (W)
        if num_meridians_current > 1:
            neighbor_lat = lat_idx
            neighbor_lon = (lon_idx - 1) % num_meridians_current
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        # 西北方邻居 (NW)
        if lat_idx > 0:
            neighbor_lat = lat_idx - 1
            num_meridians_north = self.grid_info['meridians_per_circle'][neighbor_lat]
            if num_meridians_north == 1:
                neighbor_lon = 0
            else:
                neighbor_lon = int((lon_idx - 0.5) * num_meridians_north / num_meridians_current) % num_meridians_north
            neighbors.append((neighbor_lat, neighbor_lon))
        else:
            neighbors.append((-1, -1))

        return neighbors

    def _generate_action_space(self) -> List[Dict]:
        """生成动作空间：每个位置9个旋转"""
        action_pairs = []

        for pos_idx, position in enumerate(self.all_positions):
            # 生成9个旋转：8个三角面法向量 + 1个径向
            rotations = self._generate_9_rotations(pos_idx, position)

            for rot_idx, (rotation_matrix, normal, rotation_type) in enumerate(rotations):
                action_pairs.append({
                    'position_idx': pos_idx,
                    'position': position.copy(),
                    'normal': normal.copy(),
                    'type': rotation_type,
                    'rotation_idx': rot_idx,
                    'rotation_matrix': rotation_matrix.copy()
                })

        return action_pairs

    def _generate_9_rotations(self, pos_idx: int, position: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """为位置生成9个旋转：8个三角面法向量 + 1个径向"""
        rotations = []
        center = np.array(self.params.base_station_pos)

        # 1. 径向旋转（向外）
        radial_normal = (position - center) / np.linalg.norm(position - center)
        radial_matrix = self._create_rotation_matrix_from_normal(radial_normal)
        rotations.append((radial_matrix, radial_normal, 'radial'))

        # 2. 获取8个邻居位置
        neighbors = self.neighbors[pos_idx]
        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        neighbor_positions = []
        for direction in direction_names:
            if neighbors[direction] is not None:
                neighbor_pos_idx = neighbors[direction]
                neighbor_pos = self.all_positions[neighbor_pos_idx]
                neighbor_positions.append(neighbor_pos)
            else:
                # 边界处理：使用径向投影
                neighbor_positions.append(self._get_boundary_neighbor(position, direction))

        # 3. 计算8个三角面的法向量
        for i in range(8):
            # 当前三角面：中心点 + 第i个邻居 + 第(i+1)个邻居
            neighbor1 = neighbor_positions[i]
            neighbor2 = neighbor_positions[(i + 1) % 8]

            # 计算三角面法向量
            vec1 = neighbor1 - position
            vec2 = neighbor2 - position

            if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                cross_product = np.cross(vec1, vec2)
                if np.linalg.norm(cross_product) > 1e-6:
                    face_normal = cross_product / np.linalg.norm(cross_product)
                else:
                    face_normal = radial_normal  # 退化情况
            else:
                face_normal = radial_normal

            face_matrix = self._create_rotation_matrix_from_normal(face_normal)
            rotations.append((face_matrix, face_normal, f'face_{direction_names[i]}'))

        return rotations

    def _get_boundary_neighbor(self, position: np.ndarray, direction: str) -> np.ndarray:
        """边界处理：生成虚拟邻居位置"""
        center = np.array(self.params.base_station_pos)
        radius = self.params.antenna_sphere_radius

        # 构建局部坐标系
        radial_vec = (position - center) / np.linalg.norm(position - center)

        if abs(radial_vec[2]) < 0.9:
            tangent1 = np.cross(radial_vec, np.array([0, 0, 1]))
        else:
            tangent1 = np.cross(radial_vec, np.array([1, 0, 0]))
        tangent1 = tangent1 / np.linalg.norm(tangent1)

        tangent2 = np.cross(radial_vec, tangent1)

        # 根据方向生成虚拟邻居
        direction_vectors = {
            'N': tangent2,
            'NE': (tangent1 + tangent2) / np.sqrt(2),
            'E': tangent1,
            'SE': (tangent1 - tangent2) / np.sqrt(2),
            'S': -tangent2,
            'SW': (-tangent1 - tangent2) / np.sqrt(2),
            'W': -tangent1,
            'NW': (-tangent1 + tangent2) / np.sqrt(2)
        }

        direction_vec = direction_vectors.get(direction, tangent1)
        virtual_neighbor = position + self.d_min * direction_vec

        # 投影回球面
        virtual_neighbor_vec = virtual_neighbor - center
        virtual_neighbor_normalized = virtual_neighbor_vec / np.linalg.norm(virtual_neighbor_vec)
        return center + radius * virtual_neighbor_normalized

    def _create_rotation_matrix_from_normal(self, normal: np.ndarray) -> np.ndarray:
        """从法向量创建旋转矩阵"""
        try:
            # 法向量作为x'轴
            x_axis = normal / np.linalg.norm(normal)

            # 选择一个参考向量来构建y'轴
            if abs(x_axis[2]) < 0.9:
                ref_vec = np.array([0, 0, 1])
            else:
                ref_vec = np.array([1, 0, 0])

            # y'轴
            y_axis = np.cross(x_axis, ref_vec)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # z'轴
            z_axis = np.cross(x_axis, y_axis)

            return np.column_stack([x_axis, y_axis, z_axis])
        except:
            return np.eye(3)

    def get_local_action_space(self, current_position_idx: int) -> Tuple[List[int], np.ndarray]:
        """获取智能体的局部动作空间(9×9=81维)"""
        if current_position_idx not in self.neighbors:
            return [], np.array([])

        # 获取当前位置和8个邻居位置（按固定方向顺序）
        local_position_indices = [current_position_idx]

        direction_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for direction in direction_names:
            neighbor_idx = self.neighbors[current_position_idx][direction]
            if neighbor_idx is not None:
                local_position_indices.append(neighbor_idx)
            else:
                # 边界位置用-1表示
                local_position_indices.append(-1)

        # 获取这些位置对应的所有动作索引
        local_action_indices = []
        for pos_idx in local_position_indices:
            if pos_idx != -1:
                # 找到该位置的所有旋转动作
                position_actions = []
                for action_idx, action in enumerate(self.position_rotation_pairs):
                    if action['position_idx'] == pos_idx:
                        position_actions.append((action_idx, action['rotation_idx']))

                # 按旋转索引排序
                position_actions.sort(key=lambda x: x[1])
                local_action_indices.extend([idx for idx, _ in position_actions])
            else:
                # 边界位置用-1填充
                local_action_indices.extend([-1] * self.rotations_per_position)

        # 构建局部动作映射矩阵 (9×9)
        try:
            action_matrix = np.array(local_action_indices).reshape(9, self.rotations_per_position)
        except:
            # 如果reshape失败，返回空
            action_matrix = np.array([]).reshape(0, 0)

        return local_action_indices, action_matrix

    def get_movement_direction_name(self, current_pos_idx: int, target_pos_idx: int) -> str:
        """获取移动方向名称"""
        if current_pos_idx == target_pos_idx:
            return "STAY"

        if current_pos_idx not in self.neighbors:
            return "UNKNOWN"

        neighbors = self.neighbors[current_pos_idx]
        for direction, neighbor_idx in neighbors.items():
            if neighbor_idx == target_pos_idx:
                return direction

        return "UNKNOWN"


class ChannelModel:
    """信道模型类"""

    @staticmethod
    def calculate_3gpp_antenna_gain(antenna: Antenna, user: User, params: SystemParams) -> float:
        """计算3GPP标准天线增益"""
        delta_pos = user.position - antenna.position
        distance = np.linalg.norm(delta_pos)

        if distance < 1e-6:
            return 1.0

        unit_vec = delta_pos / distance

        # 直接使用天线的法向量，不进行智能调整
        antenna_normal = antenna.normal

        # 计算角度
        cos_angle = np.clip(np.dot(unit_vec, antenna_normal), -1, 1)
        theta_b = np.arccos(cos_angle)
        theta_b_deg = np.degrees(theta_b)

        # 计算方位角
        if abs(antenna_normal[2]) < 0.9:
            ref_vec = np.array([0, 0, 1])
        else:
            ref_vec = np.array([1, 0, 0])

        u_vec = np.cross(antenna_normal, ref_vec)
        if np.linalg.norm(u_vec) > 1e-6:
            u_vec = u_vec / np.linalg.norm(u_vec)

        v_vec = np.cross(antenna_normal, u_vec)
        if np.linalg.norm(v_vec) > 1e-6:
            v_vec = v_vec / np.linalg.norm(v_vec)

        proj_vec = unit_vec - np.dot(unit_vec, antenna_normal) * antenna_normal

        if np.linalg.norm(proj_vec) > 1e-6:
            proj_vec = proj_vec / np.linalg.norm(proj_vec)
            cos_phi = np.dot(proj_vec, u_vec)
            sin_phi = np.dot(proj_vec, v_vec)
            phi_b = np.arctan2(sin_phi, cos_phi)
            phi_b_deg = np.degrees(abs(phi_b))
        else:
            phi_b_deg = 0

        # 3GPP方向图计算
        A_H = -min(12 * (phi_b_deg / params.antenna_phi_3dB) ** 2, params.antenna_G_s)
        A_V = -min(12 * (theta_b_deg / params.antenna_theta_3dB) ** 2, params.antenna_G_v)
        A_total_dB = params.antenna_G_max - min(-(A_H + A_V), params.antenna_G_s)

        # 距离补偿
        distance_compensation = min(3, 20 * np.log10(50 / max(distance, 10)))
        adaptive_gain_dB = A_total_dB + distance_compensation

        return 10 ** (adaptive_gain_dB / 10)

    @staticmethod
    def vehicle_channel_model_simplified(distance: float, antenna_gain_linear: float,
                                         antenna: Antenna, user: User, params: SystemParams) -> complex:
        """车辆信道模型 (3GPP UMi标准)"""
        distance = max(distance, 1)
        fc_ghz = params.fc / 1e9

        # 计算LoS概率 (3GPP UMi标准)
        d_2d = distance  
        los_prob = min(18/d_2d, 1) * (1 - np.exp(-d_2d/36)) + np.exp(-d_2d/36)
        
        # 判断LoS/NLoS条件
        is_los = np.random.rand() < los_prob
        
        # 3GPP UMi路径损耗模型
        if is_los:
            # LoS条件: PL = 32.4 + 20*log10(d) + 20*log10(fc)
            PL_dB = 32.4 + 20 * np.log10(distance) + 20 * np.log10(fc_ghz)
        else:
            # NLoS条件: PL = 35.3 + 22.4*log10(d) + 21.3*log10(fc)
            PL_dB = 35.3 + 22.4 * np.log10(distance) + 21.3 * np.log10(fc_ghz)
            
            # 3GPP标准NLOS额外损耗: 均值6dB, 标准差2dB
            nlos_loss = max(np.random.normal(6, 2), 2)  # 最小2dB
            PL_dB += nlos_loss

        PL_linear = 10 ** (-PL_dB / 10)

        # 计算多径
        h_multipath = ChannelModel._calculate_simplified_multipath(antenna, user, params)

        return np.sqrt(PL_linear) * antenna_gain_linear * h_multipath

    @staticmethod
    def _calculate_simplified_multipath(antenna: Antenna, user: User, params: SystemParams) -> complex:
        """计算多径 (简化版本，LoS/NLoS已在路径损耗中处理)"""
        h_total = 0
        distance = np.linalg.norm(user.position - antenna.position)

        # 简化LoS路径 (主路径)
        h_los = 1.0 * np.exp(1j * 2 * np.pi * params.fc * distance / params.c)
        h_total += h_los

        # 建筑物反射
        for building in params.buildings:
            reflection_exists, path_length, reflection_coeff = \
                ChannelModel._calculate_building_reflection(
                    antenna.position, user.position, building)

            if reflection_exists:
                amplitude = reflection_coeff * (1 / np.sqrt(path_length ** 2)) * 0.3
                phase = (2 * np.pi * params.fc * path_length / params.c +
                         np.pi + 0.2 * np.pi * np.random.randn())
                h_reflection = amplitude * np.exp(1j * phase)
                h_total += h_reflection

        # 瑞利衰落
        h_rayleigh = 0.2 * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        h_total += h_rayleigh

        # 归一化
        if abs(h_total) > 0:
            return h_total / abs(h_total) * np.sqrt(abs(h_total) ** 2)
        else:
            return (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    @staticmethod
    def _calculate_building_reflection(tx_pos: np.ndarray, rx_pos: np.ndarray,
                                       building_pos: np.ndarray) -> Tuple[bool, float, float]:
        """计算建筑物反射"""
        reflection_point = building_pos
        d1 = np.linalg.norm(tx_pos - reflection_point)
        d2 = np.linalg.norm(reflection_point - rx_pos)
        path_length = d1 + d2

        direct_distance = np.linalg.norm(tx_pos - rx_pos)

        if direct_distance < path_length < direct_distance * 2.5:
            reflection_coeff = 0.2 + 0.2 * np.random.rand()

            incident_vec = reflection_point - tx_pos
            reflected_vec = rx_pos - reflection_point

            if np.linalg.norm(incident_vec) > 0 and np.linalg.norm(reflected_vec) > 0:
                incident_vec /= np.linalg.norm(incident_vec)
                reflected_vec /= np.linalg.norm(reflected_vec)
                cos_angle = abs(np.dot(incident_vec, reflected_vec))
                reflection_coeff *= cos_angle ** 0.5

            return True, path_length, reflection_coeff
        else:
            return False, 0.0, 0.0

    @staticmethod
    def uav_channel_model_v2(distance: float, antenna_gain_linear: float,
                             user: User, params: SystemParams) -> complex:
        """UAV信道模型 (3GPP A2G标准)"""
        distance = max(distance, 1)
        fc_ghz = params.fc / 1e9
        height = user.height

        # 3GPP A2G路径损耗模型: PL = 28 + 22*log10(d) + 20*log10(fc)
        PL_dB = 28 + 22 * np.log10(distance) + 20 * np.log10(fc_ghz)
        
        # 高度补偿 (3GPP标准)
        if height > 80:
            PL_dB -= 2  # 高空减少损耗
        
        # 空中环境NLOS损耗较小: 均值3dB, 标准差1dB
        if np.random.rand() < 0.3:  # 30%概率有遮挡
            nlos_loss = max(np.random.normal(3, 1), 1)  # 最小1dB
            PL_dB += nlos_loss

        PL_linear = 10 ** (-PL_dB / 10)

        # 莱斯K因子
        if height > 80:
            K_dB = 15
        elif height > 50:
            K_dB = 10
        else:
            K_dB = 6

        K_linear = 10 ** (K_dB / 10)
        h_los = np.sqrt(K_linear / (K_linear + 1))
        h_nlos = (np.sqrt(1 / (K_linear + 1)) *
                  (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2))

        h_rice = h_los + h_nlos
        return np.sqrt(PL_linear) * antenna_gain_linear * h_rice


class UserMobility:
    """用户移动性模型"""

    @staticmethod
    def generate_user_positions(params: SystemParams, seed: int = 42) -> List[User]:
        """生成用户位置 - 十字路口场景"""
        np.random.seed(seed)
        users = []

        intersection_center = np.array([150, 150, 0])
        lane_width = 3.5
        road_length = 300

        # 道路边界
        road_bounds = {
            'ns_x': [146.5, 153.5],
            'ns_y': [0, 300],
            'ew_x': [0, 300],
            'ew_y': [146.5, 153.5]
        }

        vehicle_id = 1

        # 生成地面车辆
        for v in range(params.num_ground_users):
            user = User(id=vehicle_id, type='vehicle', position=np.zeros(3), height=1.5)
            user.velocity = 15 + 5 * np.random.rand()

            # 随机选择道路
            if np.random.rand() < 0.5:
                # 南北向道路
                x_pos = (road_bounds['ns_x'][0] +
                         (road_bounds['ns_x'][1] - road_bounds['ns_x'][0]) * np.random.rand())
                y_pos = (road_bounds['ns_y'][0] +
                         (road_bounds['ns_y'][1] - road_bounds['ns_y'][0]) * np.random.rand())

                if x_pos < 150:
                    user.lane = 'north_bound'
                    user.direction = np.array([0, 1, 0])
                else:
                    user.lane = 'south_bound'
                    user.direction = np.array([0, -1, 0])
            else:
                # 东西向道路
                x_pos = (road_bounds['ew_x'][0] +
                         (road_bounds['ew_x'][1] - road_bounds['ew_x'][0]) * np.random.rand())
                y_pos = (road_bounds['ew_y'][0] +
                         (road_bounds['ew_y'][1] - road_bounds['ew_y'][0]) * np.random.rand())

                if y_pos < 150:
                    user.lane = 'east_bound'
                    user.direction = np.array([1, 0, 0])
                else:
                    user.lane = 'west_bound'
                    user.direction = np.array([-1, 0, 0])

            user.position = np.array([x_pos, y_pos, user.height])
            users.append(user)
            vehicle_id += 1

        # 生成空中用户
        for u in range(params.num_air_users):
            height = (params.air_height_range[0] +
                      (params.air_height_range[1] - params.air_height_range[0]) * np.random.rand())

            angle = 2 * np.pi * np.random.rand()
            radius = 30 + 20 * np.random.rand()

            position = np.array([
                intersection_center[0] + radius * np.cos(angle),
                intersection_center[1] + radius * np.sin(angle),
                height
            ])

            user = User(id=vehicle_id, type='UAV', position=position, height=height)
            user.velocity = 10 + 5 * np.random.rand()
            user.direction = np.array([-np.sin(angle), np.cos(angle), 0])
            
            # 初始化UAV轨道参数
            user.orbit_center = np.array([intersection_center[0], intersection_center[1], height])
            user.orbit_radius = radius
            user.orbit_angle = angle
            
            # 初始化高度变化参数
            user.target_height = height  # 初始目标高度等于当前高度
            user.vertical_velocity = 0.0  # 初始垂直速度为0

            users.append(user)
            vehicle_id += 1

        print(f"生成用户: {params.num_ground_users}个地面车辆, {params.num_air_users}个空中用户")
        return users

    @staticmethod
    def update_user_positions(users: List[User], dt: float = 0.1, random_seed: int = None) -> List[User]:
        """更新用户位置（简化移动模型）"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        for user in users:
            if user.type == 'vehicle':
                # 车辆沿道路移动
                displacement = user.velocity * dt * user.direction
                user.position += displacement

                # 简单的边界处理
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

            elif user.type == 'UAV':
                # 无人机环绕移动 - 使用极坐标方法确保稳定轨道
                
                # 检查是否已初始化轨道参数
                if user.orbit_radius == 0.0:
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
                
                # ===== 垂直运动处理（独立于水平运动）=====
                if hasattr(user, 'target_height') and hasattr(user, 'vertical_velocity'):
                    # 检查是否需要设置新的目标高度
                    if not hasattr(user, '_height_update_counter'):
                        user._height_update_counter = 0
                    
                    user._height_update_counter += 1
                    
                    # 根据时间间隔调整更新频率：每5秒更新一次目标高度
                    update_frequency = max(5, int(5.0 / dt))  # dt=1.0时每5步，dt=0.1时每50步
                    
                    # 定期随机设置新的目标高度（每5秒一次）
                    if user._height_update_counter % update_frequency == 0:
                        # 在50-100米范围内随机选择新的目标高度
                        user.target_height = 50.0 + 50.0 * np.random.rand()
                        
                        # 计算到达目标高度所需的垂直速度（1-3 m/s）
                        height_diff = user.target_height - user.height
                        if abs(height_diff) > 0.5:  # 如果高度差大于0.5米
                            # 随机选择垂直速度（1-3 m/s），方向根据高度差确定
                            speed = 1.0 + 2.0 * np.random.rand()  # 1-3 m/s
                            user.vertical_velocity = np.sign(height_diff) * speed
                        else:
                            user.vertical_velocity = 0.0
                    
                    # 应用垂直运动
                    if abs(user.vertical_velocity) > 0.01:  # 有垂直速度时
                        # 计算新高度
                        new_height = user.height + user.vertical_velocity * dt
                        
                        # 检查是否到达或超过目标高度
                        if ((user.vertical_velocity > 0 and new_height >= user.target_height) or
                            (user.vertical_velocity < 0 and new_height <= user.target_height)):
                            # 到达目标高度，停止垂直运动
                            user.height = user.target_height
                            user.vertical_velocity = 0.0
                        else:
                            user.height = new_height
                        
                        # 严格限制高度范围在50-100米
                        user.height = np.clip(user.height, 50.0, 100.0)
                        
                        # 如果达到边界，停止垂直运动并调整目标
                        if user.height <= 50.0:
                            user.vertical_velocity = 0.0
                            user.target_height = max(user.target_height, 55.0)
                        elif user.height >= 100.0:
                            user.vertical_velocity = 0.0
                            user.target_height = min(user.target_height, 95.0)
                        
                        # 更新轨道中心的高度
                        user.orbit_center[2] = user.height
                
                # ===== 水平运动处理（极坐标轨道）=====
                # 根据极坐标计算新的水平位置
                new_x = user.orbit_center[0] + user.orbit_radius * np.cos(user.orbit_angle)
                new_y = user.orbit_center[1] + user.orbit_radius * np.sin(user.orbit_angle)
                new_z = user.height
                
                # 环境边界约束
                new_x = np.clip(new_x, 0, 300)
                new_y = np.clip(new_y, 0, 300)
                new_z = np.clip(new_z, 50, 100)
                
                # 更新位置
                user.position = np.array([new_x, new_y, new_z])
                
                # 更新切向方向向量（用于其他计算）
                user.direction = np.array([-np.sin(user.orbit_angle), np.cos(user.orbit_angle), 0])
                
                # 轨道半径自适应调整（防止边界碰撞时轨道过大）
                center_2d = np.array([150, 150])
                if (new_x <= 5 or new_x >= 295 or new_y <= 5 or new_y >= 295):
                    # 接近边界时缩小轨道半径
                    max_safe_radius = min(
                        abs(150 - 5), abs(295 - 150),  # X方向安全半径
                        abs(150 - 5), abs(295 - 150)   # Y方向安全半径
                    ) - 10  # 留10米安全边距
                    if user.orbit_radius > max_safe_radius:
                        user.orbit_radius = max_safe_radius
                        print(f"UAV {user.id}: 轨道半径调整为 {user.orbit_radius:.1f}m（边界保护）")
                
                # 定期轨道稳定性检查（每100步一次）
                if hasattr(user, '_stability_check_counter'):
                    user._stability_check_counter += 1
                else:
                    user._stability_check_counter = 0
                
                if user._stability_check_counter % 100 == 0:
                    # 检查实际位置与理论轨道的偏差
                    theoretical_pos = np.array([
                        user.orbit_center[0] + user.orbit_radius * np.cos(user.orbit_angle),
                        user.orbit_center[1] + user.orbit_radius * np.sin(user.orbit_angle)
                    ])
                    actual_pos = user.position[:2]
                    deviation = np.linalg.norm(actual_pos - theoretical_pos)
                    
                    if deviation > 5.0:  # 偏差超过5米时重新校准
                        user.orbit_angle = np.arctan2(actual_pos[1] - user.orbit_center[1],
                                                    actual_pos[0] - user.orbit_center[0])
                        user.orbit_radius = np.linalg.norm(actual_pos - user.orbit_center[:2])
                        print(f"UAV {user.id}: 轨道校准 - 半径:{user.orbit_radius:.1f}m, 角度:{np.degrees(user.orbit_angle):.1f}°")

        return users