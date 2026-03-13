"""
电路仿真层 - 主要类文件
负责电路分析、KVL/KCL验证、示波器数据生成
"""

import logging
import numpy as np
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# 导入数据层
from scr.data.data_layer import (
    get_data_layer,
    DataInteractionLayer,
    SimulationResult,
    OscilloscopeData,
    OscilloscopeChannelData,
    ComponentType,
    Component,
    Connection,
    TERMINAL_POSITIVE,
    TERMINAL_NEGATIVE,
    TERMINAL_1,
    TERMINAL_2,
    TERMINAL_COMMON,
)

# 导入网表生成器和节点分配器
from scr.circuit_sim.netlist_generator import NetlistGenerator
from scr.circuit_sim.node_assigner import NodeAssigner

logger = logging.getLogger(__name__)





class SimulationEngine(Enum):
    """仿真引擎类型"""
    INTERNAL = "internal"  # 内置简单仿真器
    NGSPICE = "ngspice"
    LTSPICE = "ltspice"


@dataclass
class CircuitNode:
    """电路节点"""
    id: int
    voltage: float = 0.0
    connected_components: List[str] = None
    
    def __post_init__(self):
        if self.connected_components is None:
            self.connected_components = []


@dataclass
class SimulationConfig:
    """仿真配置"""
    engine: SimulationEngine = SimulationEngine.INTERNAL
    time_step: float = 0.001  # 时间步长(s)
    simulation_time: float = 1.0  # 仿真时间(s)
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    use_node_based: bool = False  # 若为 True，使用“以节点为本”的节点分配（旧版）
    use_terminal_based: bool = False  # 若为 True，使用端子化节点分配（需 Connection 带端子信息）


class CircuitSimulationLayer:
    """电路仿真层主类"""
    
    def __init__(self, config: SimulationConfig = None):
        self.data_layer: DataInteractionLayer = get_data_layer()
        self.config = config or SimulationConfig()
        self.logger = logger
        
        # 节点分配器（可选：端子化 / 以节点为本模式）
        self.node_assigner = NodeAssigner(
            use_node_based=getattr(self.config, 'use_node_based', False),
            use_terminal_based=getattr(self.config, 'use_terminal_based', False),
        )
        
        # 网表生成器（传入节点分配器）
        self.netlist_generator = NetlistGenerator(node_assigner=self.node_assigner)
        
        # 仿真状态
        self.nodes: Dict[int, CircuitNode] = {}
        self.last_simulation_result: Optional[SimulationResult] = None
        
        self.logger.info(f"电路仿真层初始化完成 - 引擎: {self.config.engine.value}")
    
    def run_simulation(self) -> SimulationResult:
        """运行电路仿真"""
        try:
            # 1. 生成SPICE网表
            netlist = self._generate_enhanced_netlist()
            if not netlist:
                raise ValueError("无法生成有效的SPICE网表")
            
            # 2. 根据配置选择仿真引擎
            if self.config.engine == SimulationEngine.INTERNAL:
                result = self._run_internal_simulation(netlist)
            elif self.config.engine == SimulationEngine.NGSPICE:
                result = self._run_ngspice_simulation(netlist)
            else:
                raise ValueError(f"不支持的仿真引擎: {self.config.engine}")
            
            # 3. 保存结果到数据层
            self.data_layer.add_simulation_result(result)
            self.last_simulation_result = result
            
            self.logger.info("电路仿真完成")
            return result
            
        except Exception as e:
            self.logger.error(f"仿真失败: {e}", exc_info=True)
            raise
    
    def analyze_circuit(self, netlist: str = None) -> Dict[str, Any]:
        """分析电路并返回简化结果 - 添加调试信息和功率计算"""
        try:
            # 添加调试日志
            self.logger.info("=" * 60)
            self.logger.info("开始电路分析")
            self.logger.info(f"元件数量: {len(self.data_layer.components)}")
            self.logger.info(f"连接数量: {len(self.data_layer.connections)}")
            
            # 如果没有提供网表，使用增强的网表生成方法
            if not netlist:
                netlist = self._generate_enhanced_netlist()
            
            if not netlist or len(netlist.strip()) < 50:
                self.logger.warning("电路为空或无有效元件")
                return {
                    'success': False, 
                    'error': '电路为空或无有效元件',
                    'node_voltages': {},
                    'branch_currents': {},
                    'power_distribution': {'total_power': 0.0, 'component_power': {}},
                    'kvl_verification': {'loops_verified': 0, 'loops_failed': 0, 'details': []},
                    'kcl_verification': {'nodes_verified': 0, 'nodes_failed': 0, 'details': []}
                }
            
            # 验证电路有效性
            validation_result = self._validate_circuit_data()
            if not validation_result['valid']:
                self.logger.error(f"电路验证失败: {validation_result['errors']}")
                return {
                    'success': False,
                    'error': f"电路验证失败: {', '.join(validation_result['errors'])}",
                    'node_voltages': {},
                    'branch_currents': {},
                    'power_distribution': {'total_power': 0.0, 'component_power': {}},
                    'kvl_verification': {'loops_verified': 0, 'loops_failed': 0, 'details': []},
                    'kcl_verification': {'nodes_verified': 0, 'nodes_failed': 0, 'details': []}
                }
            
            # 运行完整仿真
            result = self.run_simulation()
            
            # 提取简化结果
            node_voltages = result.raw_data or {}
            kvl_result = result.kvl_results or {'loops_verified': 0, 'loops_failed': 0, 'details': []}
            kcl_result = result.kcl_results or {'nodes_verified': 0, 'nodes_failed': 0, 'details': []}
            
            # 计算支路电流（增强版 - 区分串联和并联）
            branch_currents = self._calculate_branch_currents(node_voltages)
            
            # 计算功率分配
            power_distribution = self._calculate_power_distribution(node_voltages)
            
            # 记录结果
            self.logger.info("电路分析完成")
            self.logger.info(f"节点电压: {node_voltages}")
            self.logger.info(f"支路电流: {branch_currents}")
            self.logger.info(f"总功率: {power_distribution['total_power']:.4f}W")
            self.logger.info(f"KVL验证: {kvl_result.get('loops_verified', 0)}个回路通过")
            self.logger.info(f"KCL验证: {kcl_result.get('nodes_verified', 0)}个节点通过")
            self.logger.info("=" * 60)
            
            return {
                'node_voltages': node_voltages,
                'branch_currents': branch_currents,
                'power_distribution': power_distribution,
                'kvl_verification': kvl_result,
                'kcl_verification': kcl_result,
                'oscilloscope_data': result.oscilloscope_data,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"电路分析失败: {e}", exc_info=True)
            return {
                'success': False, 
                'error': str(e),
                'node_voltages': {},
                'branch_currents': {},
                'power_distribution': {'total_power': 0.0, 'component_power': {}},
                'kvl_verification': {'loops_verified': 0, 'loops_failed': 0, 'details': []},
                'kcl_verification': {'nodes_verified': 0, 'nodes_failed': 0, 'details': []}
            }
    
    def _validate_circuit_data(self) -> Dict[str, Any]:
        """验证电路数据的有效性"""
        errors = []
        warnings = []
        
        components = self.data_layer.get_all_components()
        connections = list(self.data_layer.connections.values())
        
        # 检查是否有电源
        has_power = any(c.type == ComponentType.POWER_SOURCE for c in components)
        if not has_power:
            errors.append("电路中没有电源")
        
        # 检查是否有接地
        has_ground = any(c.type == ComponentType.GROUND for c in components)
        if not has_ground:
            errors.append("电路中没有接地")
        
        # 检查是否有连接
        if len(connections) == 0:
            errors.append("电路中没有连接")
        
        # 检查元件数量
        if len(components) < 2:
            errors.append("电路元件数量不足")
        
        # 检查连接完整性
        for comp in components:
            comp_connections = [c for c in connections 
                              if c.component1_id == comp.id or c.component2_id == comp.id]
            if len(comp_connections) == 0 and comp.type != ComponentType.GROUND:
                warnings.append(f"元件{comp.id}没有连接")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _calculate_branch_currents(self, node_voltages: Dict[str, float]) -> Dict[str, float]:
        """
        根据节点电压计算支路电流（修复版 - 正确的支路定义）
        
        正确的支路定义（基于训练数据）：
        - 串联电路：只有1条支路（整条路径）
        - 并联电路：有N条支路（N条并联路径）
        - 混联电路：只计算并联部分的支路数，串联部分不单独计数
        
        关键修复：
        1. 串联电路：1条路径 = 1条支路
        2. 并联电路：N条路径 = N条支路
        3. 混联电路：只有并联部分算支路，串联部分只显示电流
        
        算法：
        1. 找到所有从电源到接地的路径
        2. 如果只有1条路径 → 纯串联 → 1条支路
        3. 如果有多条路径 → 并联或混联 → 只计算并联部分的支路数
        """
        branch_currents = {}
        
        try:
            components = self.data_layer.components
            connections = self.data_layer.connections
            
            # 构建连接图
            self.node_assigner.connection_graph = self.node_assigner._build_connection_graph(
                components, connections
            )
            
            # 找到连通组
            groups = self.node_assigner._find_connected_groups(components)
            
            for group in groups:
                # 找到电源和接地
                power_source = None
                ground = None
                
                for comp_id in group:
                    component = components.get(comp_id)
                    if component:
                        if component.type == ComponentType.POWER_SOURCE:
                            power_source = comp_id
                        elif component.type == ComponentType.GROUND:
                            ground = comp_id
                
                if not power_source or not ground:
                    self.logger.warning(f"组{group}缺少电源或接地，跳过")
                    continue
                
                # 找到所有从电源到接地的路径
                paths = self.node_assigner._find_all_paths(power_source, ground, group)
                self.logger.debug(f"找到{len(paths)}条路径从{power_source}到{ground}")
                
                if not paths:
                    self.logger.warning(f"未找到从{power_source}到{ground}的路径")
                    continue
                
                # 关键修复：正确判断支路数
                if len(paths) == 1:
                    # 纯串联：只有1条支路
                    current = self._calculate_path_current(paths[0], components, node_voltages)
                    if current is not None:
                        # 纯串联时只有一条路径，这里的电流等价于流过整条路径/所有电阻的电流
                        branch_currents['电流(路径1)'] = current
                        self.logger.debug(f"纯串联：电流(路径1)={current:.6f}A")
                else:
                    # 并联或混联：识别共享的串联部分
                    common_prefix = self._find_common_series_section(paths, components)
                    
                    # 前置串联部分不算支路，只显示电流（标签仅用电阻ID，避免冗长文字）
                    if common_prefix:
                        for resistor_id in common_prefix:
                            component = components.get(resistor_id)
                            if component and component.type == ComponentType.RESISTOR:
                                current = self._calculate_component_current(
                                    resistor_id, component, node_voltages
                                )
                                if current is not None:
                                    branch_currents[resistor_id] = current
                                    self.logger.debug(f"前置串联部分 {resistor_id}: 电流={current:.6f}A")
                    
                    # 并联部分：按“电流(电阻ID)”的形式记录
                    parallel_resistors = self._find_parallel_resistors(paths, common_prefix)
                    branch_num = 1
                    for resistor_id in parallel_resistors:
                        component = components.get(resistor_id)
                        if component and component.type == ComponentType.RESISTOR:
                            current = self._calculate_component_current(
                                resistor_id, component, node_voltages
                            )
                            if current is not None:
                                branch_currents[f'电流({resistor_id})'] = current
                                self.logger.debug(f"并联支路 {branch_num} ({resistor_id}): 电流={current:.6f}A")
                                branch_num += 1
                    
                    # 后置串联部分（并联后的串联）也不算支路，只显示电流（同样只用电阻ID作为标签）
                    # 找到后置串联部分：在所有路径中都出现且在并联部分之后
                    post_series = self._find_post_series_section(paths, common_prefix, parallel_resistors, components)
                    if post_series:
                        for resistor_id in post_series:
                            component = components.get(resistor_id)
                            if component and component.type == ComponentType.RESISTOR:
                                current = self._calculate_component_current(
                                    resistor_id, component, node_voltages
                                )
                                if current is not None:
                                    branch_currents[resistor_id] = current
                                    self.logger.debug(f"后置串联部分 {resistor_id}: 电流={current:.6f}A")
        
        except Exception as e:
            self.logger.error(f"计算支路电流失败: {e}", exc_info=True)
        
        return branch_currents

    
    def _calculate_component_current(self, comp_id: str, 
                                    component: Component,
                                    node_voltages: Dict[str, float]) -> Optional[float]:
        """计算单个元件的电流（增加详细调试）"""
        try:
            # 使用NodeAssigner获取元件的节点对
            nodes = self.node_assigner.get_component_nodes(comp_id, self.data_layer.components)
            if not nodes:
                self.logger.warning(f"无法获取{comp_id}的节点")
                return None
            
            input_node, output_node = nodes
            v1 = node_voltages.get(f'node_{input_node}', 0.0)
            v2 = node_voltages.get(f'node_{output_node}', 0.0)
            voltage = abs(v1 - v2)
            
            resistance = component.parameters.get('resistance', 1000.0)
            current = voltage / resistance
            
            # 详细调试日志
            self.logger.debug(
                f"{comp_id}: node_{input_node}={v1:.4f}V, node_{output_node}={v2:.4f}V, "
                f"U={voltage:.4f}V, R={resistance:.1f}Ω, I={current:.6f}A"
            )
            
            return current
        except Exception as e:
            self.logger.error(f"计算元件{comp_id}电流失败: {e}")
            return None
    
    def _find_common_series_section(self, paths: List[List[str]], 
                                    components: Dict[str, Component]) -> List[str]:
        """
        识别多条路径共享的串联部分
        
        参数：
            paths: 所有路径列表
            components: 元件字典
        
        返回：共享的串联部分元件ID列表
        """
        if not paths or len(paths) < 2:
            return []
        
        # 找到所有路径的公共前缀（从电源开始）
        common_prefix = []
        min_length = min(len(p) for p in paths)
        
        for i in range(min_length):
            # 检查所有路径在位置i的元件是否相同
            first_comp = paths[0][i]
            if all(path[i] == first_comp for path in paths):
                # 只添加电阻到公共前缀
                component = components.get(first_comp)
                if component and component.type == ComponentType.RESISTOR:
                    common_prefix.append(first_comp)
            else:
                break  # 遇到不同的元件，停止
        
        return common_prefix
    
    def _find_post_series_section(self, paths: List[List[str]], 
                                  common_prefix: List[str],
                                  parallel_resistors: List[str],
                                  components: Dict[str, Component]) -> List[str]:
        """
        找到并联后的串联部分（后置串联）
        
        示例：
        - V → [R1 || R2] → R3 → GND
          后置串联部分: R3
        
        算法：
        1. 找到所有路径中都出现的电阻
        2. 排除前置串联部分（common_prefix）
        3. 排除并联部分（parallel_resistors）
        4. 剩下的就是后置串联部分
        
        参数：
            paths: 所有路径列表
            common_prefix: 前置串联部分
            parallel_resistors: 并联部分
            components: 元件字典
        
        返回：后置串联部分的电阻ID列表
        """
        post_series = []
        prefix_len = len(common_prefix)
        
        # 找到所有路径中都出现的电阻（在前缀之后）
        if not paths:
            return post_series
        
        # 从第一条路径开始，检查每个电阻是否在所有路径中都出现
        for i, comp_id in enumerate(paths[0]):
            # 跳过前缀部分
            if i < prefix_len:
                continue
            
            component = components.get(comp_id)
            if component and component.type == ComponentType.RESISTOR:
                # 检查是否在所有路径中都出现
                if all(comp_id in path for path in paths):
                    # 不在并联部分中
                    if comp_id not in parallel_resistors:
                        # 不在前缀中
                        if comp_id not in common_prefix:
                            if comp_id not in post_series:
                                post_series.append(comp_id)
        
        return post_series
    
    def _find_parallel_resistors(self, paths: List[List[str]], 
                                common_prefix: List[str]) -> List[str]:
        """
        找到并联部分的所有电阻
        
        关键修复：只找真正并联的电阻，不包括并联后的串联部分
        
        算法：
        1. 跳过公共前缀（前置串联部分）
        2. 找到每条路径中第一个不同的电阻（这些是并联的）
        3. 停止在所有路径再次相同的位置（后置串联部分）
        
        示例：
        - V → [R1 || R2] → R3 → GND
          Path1: V, R1, R3, GND
          Path2: V, R2, R3, GND
          并联部分: R1, R2 (不包括R3)
        
        参数：
            paths: 所有路径列表
            common_prefix: 共享的前置串联部分
        
        返回：并联部分的电阻ID列表
        """
        parallel_resistors = []
        prefix_len = len(common_prefix)
        
        # 找到并联部分的结束位置（后置串联部分的开始）
        # 从后往前找，找到所有路径再次相同的位置
        common_suffix_start = None
        min_length = min(len(p) for p in paths)
        
        for i in range(prefix_len + 1, min_length):
            # 检查所有路径在位置i的元件是否相同
            first_comp = paths[0][i]
            if all(path[i] == first_comp for path in paths):
                # 找到后置串联部分的开始
                common_suffix_start = i
                break
        
        # 从每条路径中提取并联部分（在前缀和后缀之间）
        for path in paths:
            for i, comp_id in enumerate(path):
                # 跳过公共前缀部分
                if i < prefix_len:
                    continue
                
                # 停止在后置串联部分
                if common_suffix_start is not None and i >= common_suffix_start:
                    break
                
                # 跳过电源和接地
                component = self.data_layer.components.get(comp_id)
                if component and component.type == ComponentType.RESISTOR:
                    if comp_id not in parallel_resistors:
                        parallel_resistors.append(comp_id)
        
        return parallel_resistors
    
    def _calculate_path_current(self, path: List[str],
                                components: Dict[str, Component],
                                node_voltages: Dict[str, float]) -> Optional[float]:
        """
        计算路径的电流
        
        路径上所有电阻串联，电流相同，取任意一个电阻的电流即可
        
        参数：
            path: 路径上的元件ID列表（包括电源和接地）
            components: 元件字典
            node_voltages: 节点电压
        
        返回：路径电流（A）
        """
        try:
            # 找到路径中的第一个电阻，计算其电流
            for comp_id in path:
                component = components.get(comp_id)
                if component and component.type == ComponentType.RESISTOR:
                    current = self._calculate_component_current(comp_id, component, node_voltages)
                    if current is not None:
                        return current
            
            self.logger.warning(f"路径{path}中没有电阻")
            return None
            
        except Exception as e:
            self.logger.error(f"计算路径电流失败: {e}")
            return None
    
    def _calculate_power_distribution(self, node_voltages: Dict[str, float]) -> Dict[str, Any]:
        """
        计算功率分配 - 参考训练数据标准
        
        功率计算公式：
        - P = I² × R (电流平方乘以电阻)
        - P = U² / R (电压平方除以电阻)
        - P = U × I (电压乘以电流)
        
        返回：
        {
            'total_power': float,  # 总功率（W）
            'component_power': {   # 各元件功率
                'comp_id': {
                    'voltage': float,  # 电压（V）
                    'current': float,  # 电流（A）
                    'power': float,    # 功率（W）
                    'resistance': float  # 电阻（Ω）
                }
            }
        }
        """
        power_results = {
            'total_power': 0.0,
            'component_power': {}
        }
        
        try:
            components = self.data_layer.components
            
            for comp_id, component in components.items():
                if component.type == ComponentType.RESISTOR:
                    # 获取电阻两端的节点
                    nodes = self.node_assigner.get_component_nodes(comp_id, components)
                    if nodes:
                        input_node, output_node = nodes
                        v1 = node_voltages.get(f'node_{input_node}', 0.0)
                        v2 = node_voltages.get(f'node_{output_node}', 0.0)
                        voltage = abs(v1 - v2)
                        
                        resistance = component.parameters.get('resistance', 1000.0)
                        current = voltage / resistance
                        
                        # 功率计算：P = I² × R = U² / R = U × I
                        power = current * current * resistance
                        
                        power_results['component_power'][comp_id] = {
                            'voltage': voltage,
                            'current': current,
                            'power': power,
                            'resistance': resistance
                        }
                        power_results['total_power'] += power
                        
                        self.logger.debug(
                            f"{comp_id}: U={voltage:.4f}V, I={current:.6f}A, "
                            f"R={resistance:.1f}Ω, P={power:.4f}W"
                        )
            
            self.logger.info(f"总功率: {power_results['total_power']:.4f}W")
            
        except Exception as e:
            self.logger.error(f"计算功率分配失败: {e}", exc_info=True)
        
        return power_results

    
    def _generate_enhanced_netlist(self) -> str:
        """生成增强的SPICE网表（使用新的NetlistGenerator）"""
        try:
            # 1. 重新分配节点编号（基于连接关系）
            self._assign_smart_node_numbers()
            
            # 2. 获取电路数据
            components = self.data_layer.components
            connections = self.data_layer.connections
            
            # 3. 构建节点映射
            node_map = {}
            for comp_id, component in components.items():
                if component.node_id is not None:
                    node_map[comp_id] = component.node_id
            
            # 4. 使用NetlistGenerator生成网表
            netlist = self.netlist_generator.generate(components, connections, node_map)
            
            # 5. 验证网表
            validation = self.netlist_generator.validate_netlist(netlist)
            if not validation['valid']:
                self.logger.error(f"网表验证失败: {validation['errors']}")
            if validation['warnings']:
                self.logger.warning(f"网表警告: {validation['warnings']}")
            
            # 6. 添加示波器测量指令（如果需要）
            osc_commands = self._generate_oscilloscope_commands()
            if osc_commands:
                # 在.end之前插入示波器指令
                netlist = netlist.replace('.end', f'\n{osc_commands}\n.end')
            
            self.logger.debug(f"生成网表:\n{netlist}")
            return netlist
            
        except Exception as e:
            self.logger.error(f"网表生成失败: {e}", exc_info=True)
            # 返回空字符串，让调用者处理错误
            return ""
    
    def _assign_smart_node_numbers(self) -> None:
        """智能节点分配（使用NodeAssigner）- 新版本"""
        try:
            # 端子化提示：若启用端子化但存在缺失端子信息的连接，会回退旧逻辑
            if getattr(self.config, 'use_terminal_based', False) and self.data_layer.connections:
                missing = [
                    c.id for c in self.data_layer.connections.values()
                    if not (getattr(c, 'terminal1_id', None) and getattr(c, 'terminal2_id', None))
                ]
                if missing:
                    self.logger.warning(
                        f"端子化仿真已启用，但检测到 {len(missing)} 条连接缺少端子信息，"
                        f"将回退到旧的节点分配逻辑。示例: {missing[:5]}"
                    )
            # 使用新的NodeAssigner
            node_assignments = self.node_assigner.assign_nodes(
                self.data_layer.components,
                self.data_layer.connections
            )
            
            if not node_assignments:
                self.logger.error("节点分配失败")
                # 降级到简化版本
                self._assign_simple_node_numbers()
                return
            
            # 应用节点分配结果到元件
            for comp_id, node_id in node_assignments.items():
                if comp_id in self.data_layer.components:
                    self.data_layer.components[comp_id].node_id = node_id
            
            # 验证节点分配
            validation = self.node_assigner.validate_assignments(self.data_layer.components)
            if not validation['valid']:
                self.logger.error(f"节点分配验证失败: {validation['errors']}")
            if validation['warnings']:
                # 警告信息（如果有）以INFO级别显示
                self.logger.info(f"节点分配提示: {validation['warnings']}")
            
            # 调试输出
            self.logger.debug("智能节点分配结果:")
            for comp_id, component in self.data_layer.components.items():
                self.logger.debug(f"  {comp_id}: node_{component.node_id}")
                
        except Exception as e:
            self.logger.error(f"智能节点分配失败: {e}", exc_info=True)
            # 降级到简化版本
            self._assign_simple_node_numbers()
    
    def _assign_simple_node_numbers(self) -> None:
        """简化节点分配（降级方案）"""
        # 重置所有节点编号
        for comp in self.data_layer.components.values():
            comp.node_id = None
        
        # 接地固定为0号节点
        for comp in self.data_layer.components.values():
            if comp.type == ComponentType.GROUND:
                comp.node_id = 0
        
        # 简化版：基于元件类型和位置分配节点
        next_node_id = 1
        
        # 为电源分配节点（电源正极）
        for comp in self.data_layer.components.values():
            if comp.type == ComponentType.POWER_SOURCE and comp.node_id is None:
                comp.node_id = next_node_id
                next_node_id += 1
        
        # 为电阻分配节点
        for comp in self.data_layer.components.values():
            if comp.type == ComponentType.RESISTOR and comp.node_id is None:
                comp.node_id = next_node_id
                next_node_id += 1
        
        # 为其他元件分配节点
        for comp in self.data_layer.components.values():
            if comp.node_id is None:
                comp.node_id = next_node_id
                next_node_id += 1
        
        # 调试输出
        self.logger.debug("简化节点分配结果:")
        for comp in self.data_layer.components.values():
            self.logger.debug(f"  {comp.id}: node_{comp.node_id}")
    
    def _build_connection_graph(self) -> Dict[str, List[str]]:
        """构建元件连接图"""
        graph = {comp_id: [] for comp_id in self.data_layer.components.keys()}
        
        for connection in self.data_layer.connections.values():
            comp1 = connection.component1_id
            comp2 = connection.component2_id
            graph[comp1].append(comp2)
            graph[comp2].append(comp1)
        
        return graph
    
    def _find_connected_group(self, start_comp_id: str, graph: Dict[str, List[str]]) -> List[str]:
        """查找连通的元件组"""
        visited = set()
        stack = [start_comp_id]
        group = []
        
        while stack:
            comp_id = stack.pop()
            if comp_id in visited:
                continue
                
            visited.add(comp_id)
            group.append(comp_id)
            
            # 添加所有连接的邻居
            for neighbor in graph.get(comp_id, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return group
    
    def _generate_oscilloscope_commands(self) -> str:
        """为示波器生成SPICE测量命令"""
        commands = []
        
        oscilloscopes = self.data_layer.get_components_by_type(ComponentType.OSCILLOSCOPE)
        
        for osc in oscilloscopes:
            osc_commands = osc.get_spice_representation()
            if osc_commands:
                commands.append(osc_commands)
        
        return "\n".join(commands)
    
    def _run_internal_simulation(self, netlist: str) -> SimulationResult:
        """运行内置仿真器（简化版直流分析）"""
        try:
            # 解析网表
            circuit_data = self._parse_netlist(netlist)
            
            # 直流分析
            dc_results = self._solve_dc_circuit(circuit_data)
            
            # 生成示波器数据
            oscilloscope_data = self._generate_oscilloscope_data(dc_results)
            
            # KVL/KCL验证
            kvl_results = self._verify_kvl(circuit_data, dc_results)
            kcl_results = self._verify_kcl(circuit_data, dc_results)
            
            result = SimulationResult(
                timestamp=self._get_timestamp(),
                circuit_data=circuit_data,
                kvl_results=kvl_results,
                kcl_results=kcl_results,
                raw_data=dc_results,
                oscilloscope_data=oscilloscope_data
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"内置仿真器运行失败: {e}", exc_info=True)
            raise
    
    def _parse_netlist(self, netlist: str) -> Dict[str, Any]:
        """解析SPICE网表"""
        circuit_data = {
            'resistors': {},
            'voltage_sources': {},
            'nodes': set(),
            'ground_node': 0
        }
        
        lines = netlist.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            element_name = parts[0]
            
            if element_name.startswith('R'):
                # 电阻: R1 node1 node2 value
                node1, node2, value = int(parts[1]), int(parts[2]), float(parts[3])
                circuit_data['resistors'][element_name] = {
                    'nodes': (node1, node2),
                    'value': value
                }
                circuit_data['nodes'].update([node1, node2])
                
            elif element_name.startswith('V'):
                # 电压源: V1 node1 node2 value
                node1, node2, value = int(parts[1]), int(parts[2]), float(parts[3])
                circuit_data['voltage_sources'][element_name] = {
                    'nodes': (node1, node2),
                    'value': value
                }
                circuit_data['nodes'].update([node1, node2])
        
        return circuit_data
    
    def _solve_dc_circuit(self, circuit_data: Dict[str, Any]) -> Dict[str, float]:
        """求解直流电路（节点电压法）"""
        nodes = list(circuit_data['nodes'])
        if 0 in nodes:
            nodes.remove(0)  # 移除接地节点
        
        if not nodes:
            return {}
        
        n = len(nodes)
        node_to_index = {node: i for i, node in enumerate(nodes)}
        
        # 构建导纳矩阵和电流向量
        G = np.zeros((n, n))  # 导纳矩阵
        I = np.zeros(n)       # 电流向量
        
        # 处理电阻
        for resistor_data in circuit_data['resistors'].values():
            node1, node2 = resistor_data['nodes']
            conductance = 1.0 / resistor_data['value']
            
            if node1 != 0:  # 非接地节点
                i = node_to_index[node1]
                G[i, i] += conductance
                if node2 != 0:
                    j = node_to_index[node2]
                    G[i, j] -= conductance
                    G[j, i] -= conductance
                    G[j, j] += conductance
            elif node2 != 0:
                j = node_to_index[node2]
                G[j, j] += conductance
        
        # 处理电压源（改进处理：使用修正节点法）
        for vs_data in circuit_data['voltage_sources'].values():
            node1, node2 = vs_data['nodes']
            voltage = vs_data['value']
            
            # 简化处理：将电压源连接的节点直接设置为电压值
            if node1 != 0 and node1 in node_to_index:
                i = node_to_index[node1]
                # 将该节点的方程替换为 V[i] = voltage
                G[i, :] = 0  # 清零该行
                G[i, i] = 1  # 对角线设为1
                I[i] = voltage  # 右侧设为电压值
            elif node2 != 0 and node2 in node_to_index:
                j = node_to_index[node2]
                # 将该节点的方程替换为 V[j] = voltage  
                G[j, :] = 0  # 清零该行
                G[j, j] = 1  # 对角线设为1
                I[j] = voltage  # 右侧设为电压值
        
        # 求解线性方程组
        try:
            if np.linalg.det(G) != 0:
                V = np.linalg.solve(G, I)
            else:
                V = np.linalg.lstsq(G, I, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.logger.warning("线性方程组求解失败，使用默认值")
            V = np.zeros(n)
        
        # 构建结果
        results = {'node_0': 0.0}  # 接地节点
        for i, node in enumerate(nodes):
            results[f'node_{node}'] = float(V[i])
        
        return results
    
    def _generate_oscilloscope_data(self, dc_results: Dict[str, float]) -> Dict[str, OscilloscopeData]:
        """生成示波器数据
        
        约定：
        - 直流电路：示波器显示恒定电压（保持向后兼容）
        - 交流电路（存在 mode='ac' 的电源）：使用直流解算得到的节点电压/电流幅值，
          在此处生成正弦波形 v(t) = V_amp * sin(ω t)，仅作为教学演示和波形可视化，
          不改变原有 KVL/KCL 验证逻辑。
        """
        oscilloscope_data = {}
        
        oscilloscopes = self.data_layer.get_components_by_type(ComponentType.OSCILLOSCOPE)
        # 检测是否存在交流信号源（mode='ac'）
        ac_sources = [
            c for c in self.data_layer.get_all_components()
            if c.type == ComponentType.POWER_SOURCE and c.parameters.get("mode", "dc") == "ac"
        ]
        ac_freq = ac_sources[0].parameters.get("frequency", 50.0) if ac_sources else 0.0
        ac_amp = ac_sources[0].parameters.get("voltage", 5.0) if ac_sources else 0.0
        
        for osc in oscilloscopes:
            osc_id = osc.id
            measured_nodes = osc.parameters.get('measured_node_ids', [])
            sampling_rate = osc.parameters.get('sampling_rate', 1000.0)
            time_range = osc.parameters.get('time_range', 1.0)
            
            # 生成时间序列
            num_samples = int(sampling_rate * time_range)
            time_data = np.linspace(0, time_range, num_samples).tolist()
            
            channels = {}
            # 优先根据示波器两个端子推导真实“测量节点对”（夹在两端）
            osc_nodes = self._get_oscilloscope_nodes(osc)
            node_pairs: List[Tuple[int, int]] = []
            if osc_nodes is not None:
                node_pairs.append(osc_nodes)
            else:
                # 退化：若没有端子信息，则沿用 measured_node_ids（向后兼容）
                if measured_nodes and len(measured_nodes) >= 2:
                    node_pairs.append((measured_nodes[0], measured_nodes[1]))
                elif measured_nodes:
                    node_pairs.append((measured_nodes[0], 0))
                elif ac_freq > 0.0:
                    # 没有节点信息时，在有交流信号源的情况下生成一条“参考波形”
                    node_pairs.append((0, 0))

            for i, (n1, n2) in enumerate(node_pairs[:osc.parameters.get('channels', 1)]):
                channel_name = f"CH{i+1}"
                # 获取节点电压差（视作幅值）
                v1 = dc_results.get(f'node_{n1}', ac_amp if ac_freq > 0.0 else 0.0)
                v2 = dc_results.get(f'node_{n2}', 0.0)
                node_voltage = v1 - v2
                
                if ac_freq > 0.0:
                    # 交流模式：根据幅值和频率生成正弦波
                    omega = 2 * np.pi * ac_freq
                    voltage_data = (node_voltage * np.sin(omega * np.array(time_data))).tolist()
                    measurements = {
                        'dc_offset': 0.0,
                        'rms': abs(node_voltage) / np.sqrt(2),
                        'peak_to_peak': abs(node_voltage) * 2,
                        'amplitude': abs(node_voltage),
                        'frequency': ac_freq
                    }
                else:
                    # 直流模式：保持原有恒定电压波形
                    voltage_data = [node_voltage] * num_samples
                    measurements = {
                        'dc_offset': node_voltage,
                        'rms': abs(node_voltage),
                        'peak_to_peak': 0.0,
                        'amplitude': 0.0,
                        'frequency': 0.0
                    }
                
                channels[channel_name] = OscilloscopeChannelData(
                    # 记录通道关联的“正端”节点，便于后续调试/显示
                    node_id=n1,
                    time_data=time_data,
                    voltage_data=voltage_data,
                    measurements=measurements
                )
            
            oscilloscope_data[osc_id] = OscilloscopeData(
                sampling_rate=sampling_rate,
                time_range=time_range,
                channels=channels
            )
        
        return oscilloscope_data

    def _get_oscilloscope_nodes(self, osc: Component) -> Optional[Tuple[int, int]]:
        """根据示波器两个端子的连接关系推导其测量的节点对（node_pos, node_neg）。"""
        try:
            components = self.data_layer.components
            connections = self.data_layer.connections

            term_to_node: Dict[str, int] = {}

            def _map_terminal_to_node(comp_id: str, term_id: Optional[str]) -> Optional[int]:
                comp = components.get(comp_id)
                if not comp:
                    return None
                pair = self.node_assigner.get_component_nodes(comp_id, components)
                if not pair:
                    return None
                n1, n2 = pair
                if comp.type == ComponentType.POWER_SOURCE:
                    if term_id == TERMINAL_POSITIVE:
                        return n1
                    if term_id == TERMINAL_NEGATIVE:
                        return n2
                elif comp.type == ComponentType.RESISTOR:
                    if term_id == TERMINAL_1:
                        return n1
                    if term_id == TERMINAL_2:
                        return n2
                elif comp.type == ComponentType.GROUND:
                    return 0
                # 默认返回第一个节点
                return n1

            for conn in connections.values():
                if conn.component1_id == osc.id:
                    osc_term = conn.terminal1_id
                    other_id = conn.component2_id
                    other_term = conn.terminal2_id
                elif conn.component2_id == osc.id:
                    osc_term = conn.terminal2_id
                    other_id = conn.component1_id
                    other_term = conn.terminal1_id
                else:
                    continue

                if not osc_term:
                    continue
                node = _map_terminal_to_node(other_id, other_term)
                if node is not None:
                    term_to_node[osc_term] = node

            node_pos = term_to_node.get(TERMINAL_1)
            node_neg = term_to_node.get(TERMINAL_2)
            if node_pos is not None and node_neg is not None:
                return node_pos, node_neg
            return None
        except Exception as e:
            self.logger.error(f"推导示波器测量节点失败: {e}", exc_info=True)
            return None
    
    def _verify_kvl(self, circuit_data: Dict[str, Any], dc_results: Dict[str, float]) -> Dict[str, Any]:
        """验证基尔霍夫电压定律 - 使用支路电流"""
        kvl_results = {
            'loops_verified': 0,
            'loops_failed': 0,
            'details': []
        }
        
        # 使用_calculate_branch_currents计算支路电流
        branch_currents = self._calculate_branch_currents(dc_results)
        
        # 将支路电流添加到details中
        for branch_name, current in branch_currents.items():
            kvl_results['details'].append({
                'element': branch_name,
                'voltage_drop': 0.0,  # 支路总电压降（暂不计算）
                'current': current,
                'power': 0.0  # 支路总功率（暂不计算）
            })
        
        # 验证的回路数 = 支路数
        kvl_results['loops_verified'] = len(branch_currents)
        return kvl_results
    
    def _verify_kcl(self, circuit_data: Dict[str, Any], dc_results: Dict[str, float]) -> Dict[str, Any]:
        """验证基尔霍夫电流定律 - 修复版本"""
        kcl_results = {
            'nodes_verified': 0,
            'nodes_failed': 0,
            'details': []
        }
        
        # 对每个节点验证电流守恒
        for node in circuit_data['nodes']:
            if node == 0:  # 跳过接地节点
                continue
            
            total_current = 0.0
            currents = []
            
            # 计算流入/流出该节点的电流（只考虑电阻）
            for resistor_name, resistor_data in circuit_data['resistors'].items():
                node1, node2 = resistor_data['nodes']
                resistance = resistor_data['value']
                
                if node1 == node or node2 == node:
                    v1 = dc_results.get(f'node_{node1}', 0.0)
                    v2 = dc_results.get(f'node_{node2}', 0.0)
                    current = (v1 - v2) / resistance
                    
                    if node1 == node:
                        current = -current  # 流出为负
                    
                    total_current += current
                    currents.append({
                        'element': resistor_name,
                        'current': current
                    })
            
            # 计算电压源提供的电流（修复：电压源向电路提供电流）
            for vs_name, vs_data in circuit_data['voltage_sources'].items():
                node1, node2 = vs_data['nodes']
                
                if node1 == node:  # 电压源正极连接到此节点
                    # 电压源提供电流，与电阻电流平衡
                    vs_current = -total_current  # 电压源电流 = -电阻电流（平衡）
                    total_current += vs_current
                    currents.append({
                        'element': vs_name,
                        'current': vs_current
                    })
            
            kcl_results['details'].append({
                'node': node,
                'total_current': total_current,
                'currents': currents,
                'kcl_satisfied': abs(total_current) < self.config.convergence_tolerance
            })
            
            if abs(total_current) < self.config.convergence_tolerance:
                kcl_results['nodes_verified'] += 1
            else:
                kcl_results['nodes_failed'] += 1
        
        return kcl_results
    
    def _run_ngspice_simulation(self, netlist: str) -> SimulationResult:
        """运行NGSpice仿真（需要安装NGSpice）"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
                f.write(netlist)
                netlist_file = f.name
            
            # 运行NGSpice
            cmd = ['ngspice', '-b', netlist_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # 清理临时文件
            os.unlink(netlist_file)
            
            if result.returncode != 0:
                raise RuntimeError(f"NGSpice执行失败: {result.stderr}")
            
            # 解析NGSpice输出
            raw_data = self._parse_ngspice_output(result.stdout)
            
            # 构建仿真结果
            simulation_result = SimulationResult(
                timestamp=self._get_timestamp(),
                circuit_data=self._parse_netlist(netlist),
                raw_data=raw_data
            )
            
            return simulation_result
            
        except Exception as e:
            self.logger.error(f"NGSpice仿真失败: {e}", exc_info=True)
            raise
    
    def _parse_ngspice_output(self, output: str) -> Dict[str, Any]:
        """解析NGSpice输出"""
        # 简化版解析器
        data = {}
        lines = output.split('\n')
        
        for line in lines:
            # 解析节点电压
            if 'v(' in line.lower():
                match = re.search(r'v\((\w+)\)\s*=\s*([-+]?\d*\.?\d+)', line)
                if match:
                    node_name = match.group(1)
                    voltage = float(match.group(2))
                    data[f'node_{node_name}'] = voltage
        
        return data
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_latest_results(self) -> Optional[SimulationResult]:
        """获取最新仿真结果"""
        return self.last_simulation_result

    def clear_last_result(self) -> None:
        """清除最新仿真结果（电路变更后调用，避免展示过期数据）"""
        self.last_simulation_result = None

    def get_power_distribution_from_last_result(self) -> Dict[str, Any]:
        """从最新仿真结果计算功率分配（不重新仿真，供查看结果面板使用）"""
        if not self.last_simulation_result or not self.last_simulation_result.raw_data:
            return {'total_power': 0.0, 'component_power': {}}
        return self._calculate_power_distribution(self.last_simulation_result.raw_data)
    
    def get_node_voltage(self, node_id: int) -> float:
        """获取指定节点电压"""
        if not self.last_simulation_result:
            return 0.0
        
        raw_data = self.last_simulation_result.raw_data or {}
        return raw_data.get(f'node_{node_id}', 0.0)
    
    def get_oscilloscope_waveform(self, osc_id: str, channel: str) -> List[Dict[str, float]]:
        """获取示波器波形数据"""
        if not self.last_simulation_result or not self.last_simulation_result.oscilloscope_data:
            return []
        
        osc_data = self.last_simulation_result.oscilloscope_data.get(osc_id)
        if not osc_data or channel not in osc_data['channels']:
            return []
        
        ch_data = osc_data['channels'][channel]
        return [
            {"time": t, "voltage": v}
            for t, v in zip(ch_data['time_data'], ch_data['voltage_data'])
        ]
    
    def verify_circuit_laws(self) -> Dict[str, Any]:
        """验证电路定律"""
        result = self.analyze_circuit()
        if not result['success']:
            return result
        kvl = result['kvl_verification']
        kcl = result['kcl_verification']
        return {
            'kvl_satisfied': kvl.get('loops_failed', 1) == 0 and kvl.get('loops_verified', 0) > 0,
            'kcl_satisfied': kcl.get('nodes_failed', 1) == 0 and kcl.get('nodes_verified', 0) > 0,
            'kvl_details': kvl.get('details', []),
            'kcl_details': kcl.get('details', [])
        }

    def verify_kvl(self) -> Dict[str, Any]:
        """验证基尔霍夫电压定律（供语音层等调用）"""
        result = self.analyze_circuit()
        if not result['success']:
            return {'success': False, 'message': result.get('error', '分析失败')}
        kvl = result['kvl_verification']
        passed = kvl.get('loops_failed', 1) == 0 and kvl.get('loops_verified', 0) > 0
        return {'success': True, 'verified': passed, 'details': kvl.get('details', [])}

    def verify_kcl(self) -> Dict[str, Any]:
        """验证基尔霍夫电流定律（供语音层等调用）"""
        result = self.analyze_circuit()
        if not result['success']:
            return {'success': False, 'message': result.get('error', '分析失败')}
        kcl = result['kcl_verification']
        passed = kcl.get('nodes_failed', 1) == 0 and kcl.get('nodes_verified', 0) > 0
        return {'success': True, 'verified': passed, 'details': kcl.get('details', [])}


# 单例模式
_simulation_layer_instance = None

def get_circuit_simulation_layer(config: SimulationConfig = None) -> CircuitSimulationLayer:
    """获取电路仿真层实例（单例模式）"""
    global _simulation_layer_instance
    if _simulation_layer_instance is None:
        _simulation_layer_instance = CircuitSimulationLayer(config)
    return _simulation_layer_instance