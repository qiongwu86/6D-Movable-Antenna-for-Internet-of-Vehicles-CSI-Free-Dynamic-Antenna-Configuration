import numpy as np
import time
import json
import pickle
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt

from sixDMA_Environment_core_class import SystemParams, User, UserMobility, Antenna, ChannelModel
from grid_based_antenna_optimizer import GridBasedAntennaOptimizer, GridCell
from sixdma_environment_optimized import OptimizedSixDMAEnvironment


@dataclass
class GridUserInfo:
    """网格用户信息"""
    grid_id: int
    grid_type: str  # 'ground' or 'air'
    center_position: np.ndarray
    user_count: int
    user_ids: List[int]
    user_positions: List[np.ndarray]
    last_update_time: float


@dataclass
class AntennaAllocation:
    """天线分配信息"""
    surface_id: int
    antenna_position_idx: int
    antenna_position: np.ndarray
    antenna_normal: np.ndarray
    rotation_type: str
    rotation_idx: int  # 添加旋转索引
    covered_grids: Set[int]
    total_users_covered: int
    expected_average_rate: float
    allocation_score: float


class DynamicScenarioManager:
    """动态场景管理器"""
    
    def __init__(self, params: SystemParams, optimization_results_path: str = None, 
                 enable_adaptive_mapping: bool = False, stack_size: int = 5, random_seed: int = 42,
                 transmit_power_dbm: float = 23.0):
        self.params = params
        self.optimization_results_path = optimization_results_path
        self.enable_adaptive_mapping = enable_adaptive_mapping
        self.random_seed = random_seed
        self.transmit_power_dbm = transmit_power_dbm
        
        # 初始化网格系统（与优化器保持一致）
        self.grid_config = {
            'total_grids': 800,
            'ground_grids': 400,
            'air_grids': 400,
            'ground_grid_size': (20, 20),
            'air_grid_size': (20, 20),
            'ground_height': 1.5,
            'air_height_range': params.air_height_range
        }
        
        # 天线表面配置
        self.num_surfaces = params.num_surfaces  # 使用参数中的天线表面数
        
        # 存储结构
        self.grid_cells: List[GridCell] = []
        self.grid_user_info: Dict[int, GridUserInfo] = {}
        self.current_users: List[User] = []
        self.optimization_data: Dict = {}
        self.antenna_allocations: List[AntennaAllocation] = []
        
        # 缓存标志
        self.grid_space_initialized = False
        self.optimization_loaded = False
        
        # 强制重新生成网格（因为修改了空中网格逻辑）
        self.force_regenerate_grids = True
        
        # 动态更新配置
        self.update_interval = 1.0  # 用户位置更新间隔(秒)
        self.scenario_duration = 300.0  # 场景持续时间(秒)
        
        # 性能统计
        self.stats = {
            'total_updates': 0,
            'avg_occupied_grids': 0,
            'avg_users_per_occupied_grid': 0,
            'coverage_efficiency': 0,
            'allocation_history': [],
            'update_rates': [],  # 每次更新的总用户速率
            'avg_update_rate': 0  # 平均每次更新的速率
        }
        
        # 自适应网格-天线配对系统（仅在启用时初始化）
        if self.enable_adaptive_mapping:
            self.adaptive_grid_antenna_mapping = {}  # grid_id -> List[antenna_config] (堆栈式存储)
            self.grid_antenna_stack_size = stack_size  # 每个网格保存的优势天线数量（可配置）
            self.rate_history = []  # 历史速率记录
            self.antenna_config_history = []  # 历史天线配置记录
            self.max_history_length = 10  # 最大历史记录长度
            self.grid_user_rate_history = {}  # 网格用户速率历史 grid_id -> List[rates]
        else:
            self.adaptive_grid_antenna_mapping = {}
            self.rate_history = []
            self.antenna_config_history = []
            self.grid_user_rate_history = {}
        
        print(f"🚀 动态场景管理器初始化完成")
        print(f"  网格配置: {self.grid_config['total_grids']}个网格")
        print(f"  天线表面: {self.num_surfaces}个")
        print(f"  更新间隔: {self.update_interval}秒")
        print(f"  自适应映射: {'✅ 启用' if self.enable_adaptive_mapping else '❌ 禁用'}")
        if self.enable_adaptive_mapping:
            print(f"  堆栈大小: {self.grid_antenna_stack_size}个天线配置/网格")
    
    def initialize_scenario(self):
        """初始化动态场景"""
        print(f"\n📋 初始化动态场景...")
        
        # 1. 生成网格空间
        self._generate_grid_space()
        
        # 2. 加载优化结果
        if self.optimization_results_path:
            self._load_optimization_results()
        
        # 3. 生成初始用户分布
        self._generate_initial_users()
        
        # 4. 初始网格用户映射
        self._update_grid_user_mapping()
        
        # 5. 初始化自适应网格-天线映射（仅在启用时）
        if self.enable_adaptive_mapping:
            self._initialize_adaptive_grid_antenna_mapping()
        
        # 6. 初始天线分配
        self._perform_initial_antenna_allocation()
        
        print(f"✅ 场景初始化完成")
        print(f"  初始用户数: {len(self.current_users)}")
        print(f"  占用网格数: {len([g for g in self.grid_user_info.values() if g.user_count > 0])}")
        if self.enable_adaptive_mapping:
            print(f"  自适应映射网格数: {len(self.adaptive_grid_antenna_mapping)}")
        else:
            print(f"  使用传统优化映射")
    
    def _generate_grid_space(self):
        """生成800个网格空间（复用优化器的网格生成功能）"""
        if self.grid_space_initialized and not self.force_regenerate_grids:
            print(f"  网格空间已初始化，跳过生成")
            return
            
        print(f"  生成{self.grid_config['total_grids']}个网格空间...")
        
        # 创建临时优化器来生成网格
        temp_optimizer = GridBasedAntennaOptimizer(self.params)
        temp_optimizer.grid_config = self.grid_config  # 使用我们的网格配置
        temp_optimizer._generate_grid_space()
        
        # 复用生成的网格
        self.grid_cells = temp_optimizer.grid_cells
        self.grid_space_initialized = True
        self.force_regenerate_grids = False  # 重置强制重新生成标志
        print(f"    地面网格: {self.grid_config['ground_grids']}个")
        print(f"    空中网格: {self.grid_config['air_grids']}个 (高度范围: {self.grid_config['air_height_range']}m)")
    
    def _load_optimization_results(self):
        """加载优化结果"""
        if self.optimization_loaded:
            print(f"  优化结果已加载，跳过加载")
            return
            
        try:
            print(f"  加载优化结果: {self.optimization_results_path}")
            
            # 加载pickle格式的完整数据
            with open(f"{self.optimization_results_path}/complete_optimization_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.optimization_data = data
            
            # 加载JSON格式的分析结果
            with open(f"{self.optimization_results_path}/optimization_analysis.json", 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
                self.optimization_data['analysis_results'] = analysis_results
            
            self.optimization_loaded = True
            print(f"    ✅ 优化结果加载成功")
            print(f"    网格分析数: {len(analysis_results.get('grid_analysis', {}))}")
            print(f"    天线位置数: {len(analysis_results.get('antenna_ranking', []))}")
            
        except Exception as e:
            print(f"    ⚠️  优化结果加载失败: {e}")
            print(f"    将使用默认分配策略")
            self.optimization_data = {}
    
    def _initialize_adaptive_grid_antenna_mapping(self):
        """从预存的优化结果中提取网格-优势天线位置映射"""
        if not self.enable_adaptive_mapping:
            return
            
        print(f"  初始化自适应网格-天线映射...")
        
        if not self.optimization_data or 'analysis_results' not in self.optimization_data:
            print(f"    ⚠️  没有优化数据，跳过自适应映射初始化")
            return
        
        analysis_results = self.optimization_data['analysis_results']
        grid_analysis = analysis_results.get('grid_analysis', {})
        
        mapping_count = 0
        for grid_id_str, grid_data in grid_analysis.items():
            grid_id = int(grid_id_str)
            top_configs = grid_data.get('top_10_configs', [])
            
            if top_configs:
                # 提取前N个优势天线配置作为初始堆栈
                antenna_stack = []
                for config in top_configs[:self.grid_antenna_stack_size]:
                    # 根据旋转类型推断旋转索引
                    rotation_idx = self._infer_rotation_idx_from_type(config['rotation_type'])
                    
                    antenna_config = {
                        'position_idx': config['position_idx'],
                        'position': np.array(config['position']),
                        'normal': np.array(config['normal']),
                        'rotation_type': config['rotation_type'],
                        'rotation_idx': rotation_idx,  # 添加旋转索引
                        'expected_rate': config['average_rate_mbps'],
                        'quality_score': config['average_rate_mbps'],  # 初始质量评分
                        'update_count': 0  # 更新次数
                    }
                    antenna_stack.append(antenna_config)
                
                self.adaptive_grid_antenna_mapping[grid_id] = antenna_stack
                mapping_count += 1
        
        print(f"    ✅ 初始化了 {mapping_count} 个网格的自适应天线映射")
        print(f"    每个网格保存 {self.grid_antenna_stack_size} 个优势天线配置")
    
    def _infer_rotation_idx_from_type(self, rotation_type: str) -> int:
        """根据旋转类型推断旋转索引"""
        if rotation_type == 'radial':
            return 0
        elif rotation_type.startswith('face_'):
            # 从face_N, face_NE等推断索引
            direction_map = {
                'face_N': 1, 'face_NE': 2, 'face_E': 3, 'face_SE': 4,
                'face_S': 5, 'face_SW': 6, 'face_W': 7, 'face_NW': 8
            }
            return direction_map.get(rotation_type, 0)
        else:
            return 0  # 默认返回径向旋转
    
    def _generate_initial_users(self):
        """生成初始用户分布"""
        print(f"  生成初始用户分布...")
        
        # 使用UserMobility生成用户
        users = UserMobility.generate_user_positions(self.params, seed=self.random_seed)
        self.current_users = users
        
        vehicle_count = sum(1 for u in users if u.type == 'vehicle')
        uav_count = sum(1 for u in users if u.type == 'UAV')
        
        print(f"    车辆用户: {vehicle_count}个")
        print(f"    无人机用户: {uav_count}个")
        print(f"    总用户数: {len(users)}个")
    
    def _update_grid_user_mapping(self):
        """更新网格-用户映射"""
        current_time = time.time()
        
        # 清空当前映射
        for grid_id in range(len(self.grid_cells)):
            self.grid_user_info[grid_id] = GridUserInfo(
                grid_id=grid_id,
                grid_type=self.grid_cells[grid_id].grid_type,
                center_position=self.grid_cells[grid_id].center_position,
                user_count=0,
                user_ids=[],
                user_positions=[],
                last_update_time=current_time
            )
        
        # 将用户分配到网格
        for user in self.current_users:
            grid_id = self._find_user_grid(user)
            if grid_id is not None:
                grid_info = self.grid_user_info[grid_id]
                grid_info.user_count += 1
                grid_info.user_ids.append(user.id)
                grid_info.user_positions.append(user.position)
        
        # 统计
        occupied_grids = [g for g in self.grid_user_info.values() if g.user_count > 0]
        total_users_in_grids = sum(g.user_count for g in occupied_grids)
        
        # 调试信息：检查用户类型和位置
        vehicle_users = [u for u in self.current_users if u.type == 'vehicle']
        uav_users = [u for u in self.current_users if u.type == 'UAV']
        users_without_grid = [u for u in self.current_users if self._find_user_grid(u) is None]
        
        print(f"    占用网格: {len(occupied_grids)}/{len(self.grid_cells)} ({len(occupied_grids)/len(self.grid_cells):.1%})")
        print(f"    网格内用户: {total_users_in_grids}/{len(self.current_users)} ({total_users_in_grids/len(self.current_users):.1%})")
        print(f"    车辆用户: {len(vehicle_users)}个, 无人机用户: {len(uav_users)}个")
        print(f"    未分配到网格的用户: {len(users_without_grid)}个")
        
        if users_without_grid:
            print(f"    🔍 调试未分配用户:")
            for user in users_without_grid:
                print(f"      用户{user.id}({user.type}): 位置{user.position}")
                self._debug_user_grid_assignment(user)
        
        if occupied_grids:
            avg_users_per_grid = total_users_in_grids / len(occupied_grids)
            print(f"    平均用户/网格: {avg_users_per_grid:.1f}")
    
    def _find_user_grid(self, user: User) -> Optional[int]:
        """找到用户所属的网格"""
        pos = user.position
        
        for grid_cell in self.grid_cells:
            bounds = grid_cell.bounds
            if (bounds['x'][0] <= pos[0] <= bounds['x'][1] and
                bounds['y'][0] <= pos[1] <= bounds['y'][1] and
                bounds['z'][0] <= pos[2] <= bounds['z'][1]):
                return grid_cell.grid_id
        
        return None
    
    def _debug_user_grid_assignment(self, user: User):
        """调试用户网格分配问题"""
        pos = user.position
        print(f"        调试用户{user.id}: 位置{pos}")
        
        # 检查是否在环境边界内
        env_size = self.params.environment_size
        if not (0 <= pos[0] <= env_size[0] and 0 <= pos[1] <= env_size[1] and 0 <= pos[2] <= env_size[2]):
            print(f"        ❌ 用户超出环境边界 {env_size}")
            return
        
        # 找到最接近的网格
        closest_grid = None
        min_distance = float('inf')
        
        for grid_cell in self.grid_cells:
            bounds = grid_cell.bounds
            center = grid_cell.center_position
            distance = np.linalg.norm(pos - center)
            
            if distance < min_distance:
                min_distance = distance
                closest_grid = grid_cell
                
            # 检查边界
            x_in = bounds['x'][0] <= pos[0] <= bounds['x'][1]
            y_in = bounds['y'][0] <= pos[1] <= bounds['y'][1] 
            z_in = bounds['z'][0] <= pos[2] <= bounds['z'][1]
            
            if x_in and y_in and z_in:
                print(f"        ✅ 应该分配到网格{grid_cell.grid_id}({grid_cell.grid_type})")
                return
        
        if closest_grid:
            print(f"        🔍 最接近网格{closest_grid.grid_id}({closest_grid.grid_type}): 距离{min_distance:.2f}m")
            print(f"           网格边界: x{closest_grid.bounds['x']}, y{closest_grid.bounds['y']}, z{closest_grid.bounds['z']}")
    
    def _perform_initial_antenna_allocation(self):
        """执行初始天线分配（完全基于优化数据）"""
        print(f"  执行初始天线分配...")
        
        # 获取有用户的网格
        occupied_grids = {grid_id: info for grid_id, info in self.grid_user_info.items() 
                         if info.user_count > 0}
        
        if not occupied_grids:
            print(f"    ⚠️  没有用户占用的网格，跳过分配")
            return
        
        print(f"    需要覆盖的网格: {len(occupied_grids)}个")
        print(f"    可用天线表面: {self.num_surfaces}个")
        
        # 必须基于优化结果分配天线
        if self.optimization_data and 'analysis_results' in self.optimization_data:
            self._allocate_antennas_with_optimization()
        else:
            print(f"    ⚠️  没有优化数据，无法进行天线分配")
            print(f"    请先运行网格优化生成优化结果")
            self.antenna_allocations = []
            return
        
        # 分析分配结果
        self._analyze_allocation_results()
    
    def _allocate_antennas_with_optimization(self):
        """基于优化结果分配天线"""
        print(f"    使用优化结果进行分配...")
        
        analysis_results = self.optimization_data['analysis_results']
        grid_analysis = analysis_results.get('grid_analysis', {})
        antenna_ranking = analysis_results.get('antenna_ranking', [])
        
        # 获取有用户的网格及其最优天线配置
        grid_antenna_candidates = []
        
        for grid_id, grid_info in self.grid_user_info.items():
            if grid_info.user_count == 0:
                continue
                
            grid_id_str = str(grid_id)
            if grid_id_str in grid_analysis:
                grid_data = grid_analysis[grid_id_str]
                top_configs = grid_data.get('top_10_configs', [])
                
                # 为该网格添加候选天线配置
                for rank, config in enumerate(top_configs[:5]):  # 取前5个配置
                    candidate = {
                        'grid_id': grid_id,
                        'user_count': grid_info.user_count,
                        'position_idx': config['position_idx'],
                        'position': np.array(config['position']),
                        'normal': np.array(config['normal']),
                        'rotation_type': config['rotation_type'],
                        'rotation_idx': self._infer_rotation_idx_from_type(config['rotation_type']),  # 添加旋转索引
                        'expected_rate': config['average_rate_mbps'],
                        'rank': rank,
                        'priority_score': config['average_rate_mbps'] * grid_info.user_count * (6 - rank)  # 综合评分
                    }
                    grid_antenna_candidates.append(candidate)
        
        # 按优先级排序
        grid_antenna_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 贪心选择天线位置（避免重复）
        selected_positions = set()
        allocations = []
        
        for candidate in grid_antenna_candidates:
            if len(allocations) >= self.num_surfaces:
                break
                
            pos_idx = candidate['position_idx']
            if pos_idx in selected_positions:
                continue  # 避免重复选择同一位置
            
            # 计算该天线能覆盖的所有网格（基于预计算的映射关系）
            covered_grids = self._find_covered_grids(candidate['position_idx'], candidate.get('rotation_idx'))
            total_covered_users = sum(self.grid_user_info[gid].user_count for gid in covered_grids)
            
            allocation = AntennaAllocation(
                surface_id=len(allocations),
                antenna_position_idx=pos_idx,
                antenna_position=candidate['position'],
                antenna_normal=candidate['normal'],
                rotation_type=candidate['rotation_type'],
                rotation_idx=candidate.get('rotation_idx', 0),  # 获取旋转索引，默认为0
                covered_grids=covered_grids,
                total_users_covered=total_covered_users,
                expected_average_rate=candidate['expected_rate'],
                allocation_score=candidate['priority_score']
            )
            
            allocations.append(allocation)
            selected_positions.add(pos_idx)
        
        self.antenna_allocations = allocations
        print(f"    ✅ 基于优化结果分配了{len(allocations)}个天线表面")
    
    def _allocate_antennas_with_adaptive_mapping(self):
        """基于自适应映射分配天线"""
        print(f"    使用自适应映射进行天线分配...")
        
        if not self.adaptive_grid_antenna_mapping:
            print(f"    ⚠️  没有自适应映射数据，回退到优化结果分配")
            self._allocate_antennas_with_optimization()
            return
        
        # 收集所有有用户的网格的候选天线
        grid_antenna_candidates = []
        
        for grid_id, grid_info in self.grid_user_info.items():
            if grid_info.user_count == 0 or grid_id not in self.adaptive_grid_antenna_mapping:
                continue
            
            # 从自适应映射中获取该网格的优势天线配置
            antenna_stack = self.adaptive_grid_antenna_mapping[grid_id]
            
            for rank, antenna_config in enumerate(antenna_stack):
                candidate = {
                    'grid_id': grid_id,
                    'user_count': grid_info.user_count,
                    'position_idx': antenna_config['position_idx'],
                    'position': antenna_config['position'],
                    'normal': antenna_config['normal'],
                    'rotation_type': antenna_config['rotation_type'],
                    'rotation_idx': antenna_config.get('rotation_idx', 0),  # 获取旋转索引
                    'expected_rate': antenna_config['quality_score'],  # 使用实时质量评分
                    'rank': rank,
                    'priority_score': (antenna_config['quality_score'] * 
                                     grid_info.user_count * 
                                     (self.grid_antenna_stack_size - rank) *
                                     (1 + 0.1 * antenna_config['update_count']))  # 考虑更新频次
                }
                grid_antenna_candidates.append(candidate)
        
        # 按优先级排序
        grid_antenna_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 贪心选择天线位置（避免重复）
        selected_positions = set()
        allocations = []
        
        for candidate in grid_antenna_candidates:
            if len(allocations) >= self.num_surfaces:
                break
                
            pos_idx = candidate['position_idx']
            if pos_idx in selected_positions:
                continue  # 避免重复选择同一位置
            
            # 计算该天线能覆盖的所有网格（基于预计算的映射关系）
            covered_grids = self._find_covered_grids(candidate['position_idx'], candidate.get('rotation_idx'))
            total_covered_users = sum(self.grid_user_info[gid].user_count for gid in covered_grids)
            
            allocation = AntennaAllocation(
                surface_id=len(allocations),
                antenna_position_idx=pos_idx,
                antenna_position=candidate['position'],
                antenna_normal=candidate['normal'],
                rotation_type=candidate['rotation_type'],
                rotation_idx=candidate.get('rotation_idx', 0),  # 获取旋转索引，默认为0
                covered_grids=covered_grids,
                total_users_covered=total_covered_users,
                expected_average_rate=candidate['expected_rate'],
                allocation_score=candidate['priority_score']
            )
            
            allocations.append(allocation)
            selected_positions.add(pos_idx)
        
        self.antenna_allocations = allocations
        adaptive_count = len([c for c in grid_antenna_candidates if c['rank'] == 0])
        # print(f"    ✅ 基于自适应映射分配了{len(allocations)}个天线表面")
        # print(f"    其中 {adaptive_count} 个使用了自适应更新的最优配置")
    

    
    def _find_covered_grids(self, antenna_position_idx: int, antenna_rotation_idx: int = None) -> Set[int]:
        """基于预计算的网格-天线映射关系找到天线覆盖的网格
        
        Args:
            antenna_position_idx: 天线位置索引
            antenna_rotation_idx: 天线旋转索引（可选）
            
        Returns:
            Set[int]: 该天线覆盖的网格ID集合
        """
        covered_grids = set()
        
        if not self.enable_adaptive_mapping or not self.adaptive_grid_antenna_mapping:
            # 如果没有自适应映射，回退到从优化数据中查找
            return self._find_covered_grids_from_optimization_data(antenna_position_idx, antenna_rotation_idx)
        
        # 从自适应映射的堆栈中查找该天线覆盖的网格
        for grid_id, antenna_stack in self.adaptive_grid_antenna_mapping.items():
            # 检查该网格是否有用户
            if grid_id not in self.grid_user_info or self.grid_user_info[grid_id].user_count == 0:
                continue
                
            # 检查该天线是否在该网格的堆栈中
            for antenna_config in antenna_stack:
                if antenna_config['position_idx'] == antenna_position_idx:
                    # 如果指定了旋转索引，还需要匹配旋转
                    if antenna_rotation_idx is not None:
                        if antenna_config.get('rotation_idx', 0) == antenna_rotation_idx:
                            covered_grids.add(grid_id)
                            break
                    else:
                        # 没有指定旋转索引，只要位置匹配就认为覆盖
                        covered_grids.add(grid_id)
                        break
        
        return covered_grids
    
    def _find_covered_grids_from_optimization_data(self, antenna_position_idx: int, antenna_rotation_idx: int = None) -> Set[int]:
        """从优化数据中查找天线覆盖的网格（备用方法）"""
        covered_grids = set()
        
        if not self.optimization_data or 'analysis_results' not in self.optimization_data:
            return covered_grids
        
        analysis_results = self.optimization_data['analysis_results']
        grid_analysis = analysis_results.get('grid_analysis', {})
        
        # 遍历所有网格，查找包含该天线的网格
        for grid_id_str, grid_data in grid_analysis.items():
            grid_id = int(grid_id_str)
            
            # 检查该网格是否有用户
            if grid_id not in self.grid_user_info or self.grid_user_info[grid_id].user_count == 0:
                continue
                
            top_configs = grid_data.get('top_10_configs', [])
            
            # 检查该天线是否在该网格的优势天线列表中
            for config in top_configs:
                if config['position_idx'] == antenna_position_idx:
                    # 如果指定了旋转索引，还需要匹配旋转
                    if antenna_rotation_idx is not None:
                        config_rotation_idx = self._infer_rotation_idx_from_type(config['rotation_type'])
                        if config_rotation_idx == antenna_rotation_idx:
                            covered_grids.add(grid_id)
                            break
                    else:
                        # 没有指定旋转索引，只要位置匹配就认为覆盖
                        covered_grids.add(grid_id)
                        break
        
        return covered_grids
    
    def _perform_antenna_reallocation(self):
        """重新分配天线（基于当前用户分布）"""
        if not self.optimization_data or 'analysis_results' not in self.optimization_data:
            return  # 没有优化数据则跳过
        
        # 根据自适应映射标志和更新次数选择分配策略
        if (self.enable_adaptive_mapping and 
            self.stats['total_updates'] > 2 and 
            self.adaptive_grid_antenna_mapping):
            # 使用自适应映射进行分配
            self._allocate_antennas_with_adaptive_mapping()
        else:
            # 使用原始优化结果进行分配
            self._allocate_antennas_with_optimization()
    
    def _calculate_total_user_rate(self) -> float:
        """计算所有用户的总速率（完整系统版本）"""
        if not self.antenna_allocations or not self.current_users:
            return 0.0
        
        # 构建完整的系统信道矩阵（所有分配的天线表面 × 4个天线 × 所有用户）
        num_users = len(self.current_users)
        num_antennas = len(self.antenna_allocations) * 4  # 每个表面4个天线
        
        if num_antennas == 0:
            return 0.0
            
        H = np.zeros((num_antennas, num_users), dtype=complex)
        
        # 为每个分配的天线表面生成4天线阵列并计算信道系数
        antenna_idx = 0
        for allocation in self.antenna_allocations:
            # 生成4天线矩形阵列位置
            antenna_array_positions = self._generate_4_antenna_array(
                allocation.antenna_position, allocation.antenna_normal
            )
            
            # 计算该表面4个天线对所有用户的信道系数
            for ant_pos in antenna_array_positions:
                # 创建天线对象
                antenna = Antenna(
                    surface_id=allocation.surface_id,
                    global_id=antenna_idx,
                    local_id=antenna_idx % 4,
                    position=ant_pos,
                    normal=allocation.antenna_normal,
                    surface_center=allocation.antenna_position
                )
                
                for user_idx, user in enumerate(self.current_users):
                    # 计算距离（使用表面中心距离）
                    distance = np.linalg.norm(user.position - allocation.antenna_position)
                    
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
        
        # 使用6DMA环境中的正确速率计算函数
        # 创建临时环境实例来调用速率计算函数（只创建一次并缓存）
        if not hasattr(self, '_temp_env'):
            self._temp_env = OptimizedSixDMAEnvironment(self.params)
        user_rates = self._temp_env._calculate_theoretical_rates_vectorized(H, self.transmit_power_dbm)
        
        # 更新自适应网格-天线映射（基于速率变化）
        if (self.enable_adaptive_mapping and 
            self.stats['total_updates'] > 0):  # 跳过初始计算
            self._update_adaptive_mapping_with_rates(user_rates, H)
        
        return np.sum(user_rates)
    
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
    
    def _update_adaptive_mapping_with_rates(self, user_rates: np.ndarray, channel_matrix: np.ndarray):
        """基于用户速率变化更新自适应网格-天线映射"""
        if not self.enable_adaptive_mapping or user_rates is None:
            return
        
        # 存储当前状态
        current_state = {
            'user_rates': user_rates.copy(),
            'user_grid_mapping': self._get_current_user_grid_mapping(),
            'antenna_allocations': [(alloc.antenna_position_idx, alloc.rotation_idx, alloc.surface_id) 
                                   for alloc in self.antenna_allocations],  # 包含旋转索引
            'timestamp': self.stats['total_updates'],
            'total_rate': np.sum(user_rates)
        }
        
        # 存储历史记录
        self.rate_history.append(current_state)
        
        # 保持历史记录长度
        if len(self.rate_history) > self.max_history_length:
            self.rate_history.pop(0)
        
        # 计算网格用户速率并存储历史
        self._update_grid_user_rate_history(user_rates)
        
        # 分析速率变化并更新映射（需要至少2个历史记录进行对比）
        if len(self.rate_history) >= 2:
            self._analyze_rate_changes_and_update_mapping()
    
    def _get_current_user_grid_mapping(self) -> Dict[int, int]:
        """获取当前用户到网格的映射"""
        user_grid_mapping = {}
        for grid_id, grid_info in self.grid_user_info.items():
            for user_id in grid_info.user_ids:
                user_grid_mapping[user_id] = grid_id
        return user_grid_mapping
    
    def _update_grid_user_rate_history(self, user_rates: np.ndarray):
        """更新网格用户速率历史"""
        user_grid_mapping = self._get_current_user_grid_mapping()
        
        # 按网格聚合用户速率
        grid_rates = defaultdict(list)
        for user_idx, user in enumerate(self.current_users):
            if user.id in user_grid_mapping and user_idx < len(user_rates):
                grid_id = user_grid_mapping[user.id]
                grid_rates[grid_id].append(user_rates[user_idx])
        
        # 计算每个网格的平均速率并存储
        for grid_id, rates in grid_rates.items():
            if rates:
                avg_rate = np.mean(rates)
                if grid_id not in self.grid_user_rate_history:
                    self.grid_user_rate_history[grid_id] = []
                
                self.grid_user_rate_history[grid_id].append({
                    'timestamp': self.stats['total_updates'],
                    'average_rate': avg_rate,
                    'user_count': len(rates),
                    'rate_std': np.std(rates) if len(rates) > 1 else 0.0
                })
                
                # 保持历史记录长度
                if len(self.grid_user_rate_history[grid_id]) > self.max_history_length:
                    self.grid_user_rate_history[grid_id].pop(0)
    
    def _analyze_rate_changes_and_update_mapping(self):
        """分析速率变化并更新天线映射"""
        if len(self.rate_history) < 2:
            return
        
        current_state = self.rate_history[-1]
        previous_state = self.rate_history[-2]
        
        # 检查天线配置是否发生变化
        current_antennas = set(current_state['antenna_allocations'])
        previous_antennas = set(previous_state['antenna_allocations'])
        
        if current_antennas == previous_antennas:
            return  # 天线配置未变化，无法进行归因分析
        
        # 识别变化的天线（现在包含旋转信息）
        added_antennas = current_antennas - previous_antennas
        removed_antennas = previous_antennas - current_antennas
        
        # 分析速率变化对各网格的影响
        grid_rate_changes = self._calculate_grid_rate_changes(current_state, previous_state)
        
        # 根据速率变化更新天线映射
        self._update_mapping_based_on_rate_changes(grid_rate_changes, added_antennas, removed_antennas)
    
    def _calculate_grid_rate_changes(self, current_state: Dict, previous_state: Dict) -> Dict[int, float]:
        """计算各网格的速率变化"""
        grid_rate_changes = {}
        
        current_mapping = current_state['user_grid_mapping']
        previous_mapping = previous_state['user_grid_mapping']
        current_rates = current_state['user_rates']
        previous_rates = previous_state['user_rates']
        
        # 按网格聚合速率变化
        current_grid_rates = defaultdict(list)
        previous_grid_rates = defaultdict(list)
        
        # 当前状态的网格速率
        for user_idx, user in enumerate(self.current_users):
            if (user.id in current_mapping and 
                user_idx < len(current_rates)):
                grid_id = current_mapping[user.id]
                current_grid_rates[grid_id].append(current_rates[user_idx])
        
        # 上一状态的网格速率（需要匹配相同用户）
        for user_idx, user in enumerate(self.current_users):
            if (user.id in previous_mapping and 
                user_idx < len(previous_rates)):
                grid_id = previous_mapping[user.id]
                previous_grid_rates[grid_id].append(previous_rates[user_idx])
        
        # 计算每个网格的速率变化
        all_grids = set(current_grid_rates.keys()) | set(previous_grid_rates.keys())
        for grid_id in all_grids:
            current_avg = np.mean(current_grid_rates[grid_id]) if current_grid_rates[grid_id] else 0.0
            previous_avg = np.mean(previous_grid_rates[grid_id]) if previous_grid_rates[grid_id] else 0.0
            
            # 计算相对变化率
            if previous_avg > 0:
                rate_change = (current_avg - previous_avg) / previous_avg
            else:
                rate_change = 1.0 if current_avg > 0 else 0.0
            
            grid_rate_changes[grid_id] = rate_change
        
        return grid_rate_changes
    
    def _update_mapping_based_on_rate_changes(self, grid_rate_changes: Dict[int, float], 
                                            added_antennas: set, removed_antennas: set):
        """基于速率变化更新天线映射"""
        updated_grids = 0
        stack_updates_detail = {
            'score_updates': 0,      # 评分更新次数
            'new_additions': 0,      # 新天线添加次数  
            'stack_reorders': 0,     # 堆栈重排序次数
            'grids_affected': set()  # 受影响的网格ID
        }
        
        # 对于速率提升的网格，提升新增天线的评分
        for grid_id, rate_change in grid_rate_changes.items():
            if rate_change > 0.05 and grid_id in self.adaptive_grid_antenna_mapping:  # 30%以上提升
                # 找到可能负责提升的天线（新增的天线）
                for antenna_pos_idx, rotation_idx, surface_id in added_antennas:
                    if self._is_antenna_serving_grid(antenna_pos_idx, grid_id):
                        if self._update_grid_antenna_score(grid_id, antenna_pos_idx, 
                                                         rate_change, 'positive'):
                            updated_grids += 1
                            # print(f"      网格 {grid_id}: 天线 {antenna_pos_idx}(旋转{rotation_idx}) "
                            #       f"带来 {rate_change:.1%} 速率提升")
            
            elif rate_change < -0.05 and grid_id in self.adaptive_grid_antenna_mapping:  # 30%以上下降
                # 找到可能负责下降的天线（被移除的天线）
                for antenna_pos_idx, rotation_idx, surface_id in removed_antennas:
                    if self._was_antenna_serving_grid(antenna_pos_idx, grid_id):
                        if self._update_grid_antenna_score(grid_id, antenna_pos_idx, 
                                                         abs(rate_change), 'negative'):
                            updated_grids += 1
                            # print(f"      网格 {grid_id}: 移除天线 {antenna_pos_idx}(旋转{rotation_idx}) "
                            #       f"导致 {abs(rate_change):.1%} 速率下降")
        
        if updated_grids > 0:
            print(f"    📊 更新了 {updated_grids} 个网格的天线评分")
    
    def _is_antenna_serving_grid(self, antenna_pos_idx: int, grid_id: int) -> bool:
        """判断天线是否服务于指定网格（简化实现）"""
        # 简化判断：检查天线是否在当前分配中，且能覆盖该网格
        for allocation in self.antenna_allocations:
            if allocation.antenna_position_idx == antenna_pos_idx:
                return grid_id in allocation.covered_grids
        return False
    
    def _was_antenna_serving_grid(self, antenna_pos_idx: int, grid_id: int) -> bool:
        """判断天线之前是否服务于指定网格"""
        # 简化实现：假设之前服务过（实际中可以查历史记录）
        return True
    
    def _update_grid_antenna_score(self, grid_id: int, antenna_pos_idx: int, 
                                 impact_magnitude: float, impact_type: str) -> bool:
        """更新网格中天线的评分"""
        if grid_id not in self.adaptive_grid_antenna_mapping:
            return False
        
        antenna_stack = self.adaptive_grid_antenna_mapping[grid_id]
        stack_size_before = len(antenna_stack)
        
        # 查找现有天线配置
        for config in antenna_stack:
            if config['position_idx'] == antenna_pos_idx:
                if impact_type == 'positive':
                    # 正面影响：提升评分
                    config['quality_score'] = min(1.0, config['quality_score'] + impact_magnitude * 0.5)
                    config['update_count'] += 1
                else:
                    # 负面影响：降低评分
                    config['quality_score'] = max(0.0, config['quality_score'] - impact_magnitude * 0.5)
                    config['update_count'] += 1
                
                # 重新排序堆栈（按质量评分降序）
                antenna_stack.sort(key=lambda x: x['quality_score'], reverse=True)
                return True
        
        # 如果天线不在堆栈中且影响为正，尝试添加
        if impact_type == 'positive' and impact_magnitude > 0.1:
            # 尝试找到当前使用的旋转索引
            current_rotation_idx = self._find_current_rotation_idx(antenna_pos_idx)
            
            new_config = self._find_antenna_config_by_idx(antenna_pos_idx, current_rotation_idx)
            if new_config:
                new_config['quality_score'] = impact_magnitude * 0.8  # 基于影响程度设置初始评分
                new_config['update_count'] = 1
                
                # 插入到合适位置并保持堆栈大小
                antenna_stack.append(new_config)
                antenna_stack.sort(key=lambda x: x['quality_score'], reverse=True)
                
                if len(antenna_stack) > self.grid_antenna_stack_size:
                    removed = antenna_stack.pop()  # 移除最差的
                    print(f"        ↔️ 天线替换: pos_{antenna_pos_idx}(旋转{current_rotation_idx}) 替换 pos_{removed['position_idx']}(旋转{removed.get('rotation_idx', 0)})")
                else:
                    print(f"        ➕ 新增天线: pos_{antenna_pos_idx}(旋转{current_rotation_idx})")
                
                return True
        
        return False
    
    def _find_current_rotation_idx(self, antenna_pos_idx: int) -> int:
        """找到当前分配中使用的旋转索引"""
        for allocation in self.antenna_allocations:
            if allocation.antenna_position_idx == antenna_pos_idx:
                return allocation.rotation_idx
        return 0  # 默认返回径向旋转
    
    def _find_antenna_config_by_idx(self, position_idx: int, rotation_idx: int = 0) -> Optional[Dict]:
        """根据位置索引和旋转索引找到准确的天线配置"""
        # 首先尝试从ActionSpace获取准确的位置和旋转信息
        if hasattr(self, '_temp_env') and hasattr(self._temp_env, 'action_space_manager'):
            action_space_manager = self._temp_env.action_space_manager
        else:
            # 创建临时ActionSpace管理器
            from sixDMA_Environment_core_class import ActionSpace
            action_space_manager = ActionSpace(self.params)
        
        # 验证position_idx的有效性
        if position_idx < 0 or position_idx >= len(action_space_manager.all_positions):
            return None
        
        # 获取准确的位置坐标
        position = action_space_manager.all_positions[position_idx]
        
        # 获取该位置的9种旋转
        rotations = action_space_manager._generate_9_rotations(position_idx, position)
        
        # 验证rotation_idx的有效性
        if rotation_idx < 0 or rotation_idx >= len(rotations):
            rotation_idx = 0  # 默认使用径向旋转
        
        # 获取指定旋转的信息
        rotation_matrix, normal, rotation_type = rotations[rotation_idx]
        
        # 尝试从优化数据获取期望速率
        expected_rate = 0.0
        if self.optimization_data and 'analysis_results' in self.optimization_data:
            analysis_results = self.optimization_data['analysis_results']
            antenna_ranking = analysis_results.get('antenna_ranking', [])
            
            for antenna_info in antenna_ranking:
                if antenna_info['position_idx'] == position_idx:
                    expected_rate = antenna_info.get('average_rate_mbps', 0.0)
                    break
        
        return {
            'position_idx': position_idx,
            'position': position.copy(),  # 使用ActionSpace中的准确位置
            'normal': normal.copy(),      # 使用准确的法向量
            'rotation_type': rotation_type,  # 使用准确的旋转类型
            'rotation_idx': rotation_idx,    # 添加旋转索引
            'expected_rate': expected_rate,
            'quality_score': 0.0,
            'update_count': 0
        }
    

    
    def _analyze_allocation_results(self):
        """分析分配结果"""
        print(f"\n📊 天线分配结果分析:")
        
        total_covered_grids = set()
        total_covered_users = 0
        total_expected_rate = 0
        
        for i, allocation in enumerate(self.antenna_allocations):
            total_covered_grids.update(allocation.covered_grids)
            total_covered_users += allocation.total_users_covered
            total_expected_rate += allocation.expected_average_rate
            
            print(f"  天线{i}: 位置{allocation.antenna_position_idx}, "
                  f"覆盖{len(allocation.covered_grids)}个网格, "
                  f"{allocation.total_users_covered}个用户, "
                  f"预期速率{allocation.expected_average_rate:.1f}Mbps")
        
        # 覆盖统计
        occupied_grids = len([g for g in self.grid_user_info.values() if g.user_count > 0])
        total_users = sum(g.user_count for g in self.grid_user_info.values())
        
        coverage_ratio = len(total_covered_grids) / occupied_grids if occupied_grids > 0 else 0
        user_coverage_ratio = total_covered_users / total_users if total_users > 0 else 0
        
        print(f"\n  📈 覆盖统计:")
        print(f"    网格覆盖率: {len(total_covered_grids)}/{occupied_grids} ({coverage_ratio:.1%})")
        print(f"    用户覆盖率: {total_covered_users}/{total_users} ({user_coverage_ratio:.1%})")
        print(f"    平均预期速率: {total_expected_rate/len(self.antenna_allocations):.1f}Mbps")
        print(f"    覆盖效率: {len(total_covered_grids)/len(self.antenna_allocations):.1f}网格/天线")
        
        # 更新统计信息
        self.stats.update({
            'avg_occupied_grids': occupied_grids,
            'coverage_efficiency': len(total_covered_grids)/len(self.antenna_allocations),
            'grid_coverage_ratio': coverage_ratio,
            'user_coverage_ratio': user_coverage_ratio
        })
    
    def update_scenario(self, time_step: float):
        """更新动态场景"""
        # 更新用户位置（使用UserMobility类的静态方法）
        # 为每次更新生成确定性但不同的种子
        seed_for_update = self.random_seed + self.stats['total_updates'] + 1000
        self.current_users = UserMobility.update_user_positions(self.current_users, time_step, random_seed=seed_for_update)
        
        # 更新网格-用户映射（只更新用户密度，网格结构不变）
        self._update_grid_user_mapping()
        
        # 基于新的用户分布重新分配天线（天线位置固定，只改变分配策略）
        self._perform_antenna_reallocation()
        
        # 计算总用户速率
        total_rate = self._calculate_total_user_rate()
        self.stats['update_rates'].append(total_rate)
        
        self.stats['total_updates'] += 1
        
        print(f"    更新{self.stats['total_updates']}: 总用户速率 {total_rate:.2f} Mbps")
    
    def run_dynamic_scenario(self, duration: float = None, max_updates: int = None):
        """运行动态场景"""
        if max_updates is not None:
            # 基于更新次数的运行模式
            print(f"\n🎬 开始运行动态场景 (更新次数: {max_updates})")
            
            for update_count in range(max_updates):
                print(f"\n  --- 更新 {update_count + 1}/{max_updates} ---")
                
                # 执行场景更新
                time_step = self.update_interval
                self.update_scenario(time_step)
                
                # 每5次更新输出一次详细状态
                if (update_count + 1) % 5 == 0:
                    self._print_scenario_status()
        else:
            # 基于时间的运行模式
            if duration is None:
                duration = self.scenario_duration
                
            print(f"\n🎬 开始运行动态场景 (时长: {duration}秒)")
            
            start_time = time.time()
            last_update = start_time
            
            while time.time() - start_time < duration:
                current_time = time.time()
                
                if current_time - last_update >= self.update_interval:
                    time_step = current_time - last_update
                    self.update_scenario(time_step)
                    last_update = current_time
                    
                    # 每30秒输出一次状态
                    if self.stats['total_updates'] % 30 == 0:
                        self._print_scenario_status()
                
                time.sleep(0.1)  # 避免过度占用CPU
        
        print(f"✅ 动态场景运行完成")
        self._print_final_statistics()
    
    def _print_scenario_status(self):
        """打印场景状态"""
        occupied_grids = len([g for g in self.grid_user_info.values() if g.user_count > 0])
        total_users = sum(g.user_count for g in self.grid_user_info.values())
        
        print(f"⏱️  更新{self.stats['total_updates']}: "
              f"占用网格{occupied_grids}, 总用户{total_users}")
    
    def _print_final_statistics(self):
        """打印最终统计"""
        print(f"\n📊 最终统计:")
        print(f"  总更新次数: {self.stats['total_updates']}")
        print(f"  平均占用网格: {self.stats['avg_occupied_grids']}")
        print(f"  覆盖效率: {self.stats['coverage_efficiency']:.1f}网格/天线")
        print(f"  网格覆盖率: {self.stats.get('grid_coverage_ratio', 0):.1%}")
        print(f"  用户覆盖率: {self.stats.get('user_coverage_ratio', 0):.1%}")
        
        # 速率统计
        if self.stats['update_rates']:
            print(f"\n📈 速率统计:")
            print(f"  每次更新的总用户速率: {self.stats['update_rates']}")
            print(f"  最大总速率: {max(self.stats['update_rates']):.2f} Mbps")
            print(f"  最小总速率: {min(self.stats['update_rates']):.2f} Mbps")
            print(f"  平均总速率: {np.mean(self.stats['update_rates']):.2f} Mbps")
            print(f"  速率标准差: {np.std(self.stats['update_rates']):.2f} Mbps")
    
    def save_scenario_results(self, output_dir: str):
        """保存场景结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存网格用户信息
        grid_user_data = {}
        for grid_id, info in self.grid_user_info.items():
            if info.user_count > 0:
                grid_user_data[grid_id] = {
                    'grid_type': info.grid_type,
                    'center_position': info.center_position.tolist(),
                    'user_count': info.user_count,
                    'user_ids': info.user_ids
                }
        
        with open(f"{output_dir}/grid_user_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(grid_user_data, f, indent=2, ensure_ascii=False)
        
        # 保存天线分配结果
        allocation_data = []
        for allocation in self.antenna_allocations:
            allocation_data.append({
                'surface_id': allocation.surface_id,
                'antenna_position_idx': allocation.antenna_position_idx,
                'antenna_position': allocation.antenna_position.tolist(),
                'antenna_normal': allocation.antenna_normal.tolist(),
                'rotation_type': allocation.rotation_type,
                'rotation_idx': allocation.rotation_idx,  # 包含旋转索引
                'covered_grids': list(allocation.covered_grids),
                'total_users_covered': allocation.total_users_covered,
                'expected_average_rate': allocation.expected_average_rate,
                'allocation_score': allocation.allocation_score
            })
        
        with open(f"{output_dir}/antenna_allocation.json", 'w', encoding='utf-8') as f:
            json.dump(allocation_data, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        with open(f"{output_dir}/scenario_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # 保存自适应映射状态
        self._save_adaptive_mapping_state(output_dir)
        
        print(f"📁 场景结果已保存至: {output_dir}/")
    
    def _save_adaptive_mapping_state(self, output_dir: str):
        """保存自适应映射状态"""
        if not self.enable_adaptive_mapping or not self.adaptive_grid_antenna_mapping:
            return
        
        adaptive_mapping_data = {}
        for grid_id, antenna_stack in self.adaptive_grid_antenna_mapping.items():
            adaptive_mapping_data[str(grid_id)] = []
            for config in antenna_stack:
                config_data = {
                    'position_idx': config['position_idx'],
                    'position': config['position'].tolist(),
                    'normal': config['normal'].tolist(),
                    'rotation_type': config['rotation_type'],
                    'rotation_idx': config.get('rotation_idx', 0),  # 包含旋转索引
                    'expected_rate': config['expected_rate'],
                    'quality_score': config['quality_score'],
                    'update_count': config['update_count']
                }
                adaptive_mapping_data[str(grid_id)].append(config_data)
        
        with open(f"{output_dir}/adaptive_grid_antenna_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(adaptive_mapping_data, f, indent=2, ensure_ascii=False)
        
        # 保存映射更新统计
        mapping_stats = {
            'total_grids_with_mapping': len(self.adaptive_grid_antenna_mapping),
            'stack_size': self.grid_antenna_stack_size,
            'total_updates': self.stats['total_updates'],
            'channel_history_length': len(self.channel_history),
            'grid_update_summary': {}
        }
        
        for grid_id, antenna_stack in self.adaptive_grid_antenna_mapping.items():
            total_updates = sum(config['update_count'] for config in antenna_stack)
            avg_quality = np.mean([config['quality_score'] for config in antenna_stack])
            mapping_stats['grid_update_summary'][str(grid_id)] = {
                'total_antenna_updates': total_updates,
                'average_quality_score': avg_quality,
                'current_best_antenna': antenna_stack[0]['position_idx'] if antenna_stack else None
            }
        
        with open(f"{output_dir}/adaptive_mapping_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(mapping_stats, f, indent=2, ensure_ascii=False)
        
        print(f"    💾 自适应映射状态已保存")
        print(f"    映射网格数: {len(self.adaptive_grid_antenna_mapping)}")
        print(f"    历史记录长度: {len(self.channel_history)}")
    
    def visualize_scenario(self, output_dir: str):
        """可视化场景（简化版本）"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制有用户的网格（简化）
            ground_grids = [(info.center_position, info.user_count) for info in self.grid_user_info.values() 
                           if info.user_count > 0 and info.grid_type == 'ground']
            air_grids = [(info.center_position, info.user_count) for info in self.grid_user_info.values() 
                        if info.user_count > 0 and info.grid_type == 'air']
            
            if ground_grids:
                positions = np.array([pos for pos, _ in ground_grids])
                sizes = [min(100, count * 20) for _, count in ground_grids]
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c='blue', s=sizes, alpha=0.6, label='Ground grids')
            
            if air_grids:
                positions = np.array([pos for pos, _ in air_grids])
                sizes = [min(100, count * 20) for _, count in air_grids]
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c='green', s=sizes, alpha=0.6, label='Air grids')
            
            # 绘制天线位置
            if self.antenna_allocations:
                antenna_positions = np.array([alloc.antenna_position for alloc in self.antenna_allocations])
                ax.scatter(antenna_positions[:, 0], antenna_positions[:, 1], antenna_positions[:, 2], 
                          c='red', s=200, marker='^', label='Antennas')
            
            # 绘制基站
            bs_pos = self.params.base_station_pos
            ax.scatter(bs_pos[0], bs_pos[1], bs_pos[2], 
                      c='gold', s=300, marker='*', label='Base Station')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Dynamic Scenario: Grid Users and Antenna Allocation')
            ax.legend()
            
            plt.savefig(f"{output_dir}/scenario_visualization.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 场景可视化已保存至: {output_dir}/scenario_visualization.png")
            
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")


def main():
    """主函数 - 演示动态场景管理"""
    print("🚀 动态场景管理器演示")
    
    # 初始化系统参数
    params = SystemParams()
    
    # 创建场景管理器
    scenario_manager = DynamicScenarioManager(
        params=params,
        optimization_results_path="demo_optimization_results"  # 假设已有优化结果
    )
    
    # 初始化场景
    scenario_manager.initialize_scenario()
    
    # 运行短期动态场景（演示）
    scenario_manager.run_dynamic_scenario(duration=60.0)  # 1分钟演示
    
    # 保存结果
    scenario_manager.save_scenario_results("dynamic_scenario_results")
    
    # 生成可视化
    scenario_manager.visualize_scenario("dynamic_scenario_results")
    
    print("🎉 动态场景演示完成！")


if __name__ == "__main__":
    main()
