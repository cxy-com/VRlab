"""
数据交互层 - 全局数据管理中心
负责存储和管理所有电路元件、连接关系、仿真结果和指令上下文
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# 示波器相关类型定义
class OscilloscopeChannelData(TypedDict):
    """示波器通道数据结构"""
    node_id: int
    time_data: List[float]
    voltage_data: List[float]
    measurements: Dict[str, float]


class OscilloscopeData(TypedDict):
    """示波器数据结构"""
    sampling_rate: float
    time_range: float
    channels: Dict[str, OscilloscopeChannelData]


class ComponentType(Enum):
    """电路元件类型枚举"""
    RESISTOR = "resistor"
    POWER_SOURCE = "power_source"
    GROUND = "ground"
    NODE = "node"
    WIRE = "wire"
    OSCILLOSCOPE = "oscilloscope"  # 新增：示波器元件类型


# === 端子定义（用于 VR 实体标记与节点分配）===
# 每个元件类型有固定的端子 ID，VR 场景中导线只能连到这些端子上
TERMINAL_POSITIVE = "positive"   # 电源正极
TERMINAL_NEGATIVE = "negative"   # 电源负极
TERMINAL_1 = "terminal_1"        # 电阻/二端元件 端子1
TERMINAL_2 = "terminal_2"        # 电阻/二端元件 端子2
TERMINAL_COMMON = "common"       # 接地 公共端子

# 各元件类型的端子列表（供 VR 实体创建 Collider 时使用）
COMPONENT_TERMINALS: Dict[ComponentType, List[str]] = {
    ComponentType.POWER_SOURCE: [TERMINAL_POSITIVE, TERMINAL_NEGATIVE],
    ComponentType.RESISTOR: [TERMINAL_1, TERMINAL_2],
    ComponentType.GROUND: [TERMINAL_COMMON],
    # 示波器为二端测量元件，本身不参与仿真，只用于 VR 端子连线与测量
    ComponentType.OSCILLOSCOPE: [TERMINAL_1, TERMINAL_2],
}


@dataclass
class Component:
    """电路元件基类"""
    id: str
    type: ComponentType
    position: tuple  # (x, y, z) 坐标
    parameters: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[int] = None  # SPICE节点编号

    def get_spice_representation(self) -> str:
        """获取SPICE网表表示"""
        raise NotImplementedError("子类必须实现此方法")


@dataclass
class Resistor(Component):
    """电阻元件"""

    def __init__(self, id: str, position: tuple, resistance: float):
        super().__init__(id, ComponentType.RESISTOR, position, {"resistance": resistance})

    def get_spice_representation(self) -> str:
        # 修复：使用正确的双节点系统
        # 电阻连接在两个不同的节点之间
        if not hasattr(self, 'node_id') or self.node_id is None:
            return f"R{self.id} 0 0 {self.parameters['resistance']}"
        
        # 简化处理：假设电阻一端连接到分配的节点，另一端连接到接地(0)
        # 这需要根据实际连接关系来确定，但先用这个简化版本
        node1 = self.node_id
        node2 = 0  # 暂时假设另一端接地
        return f"R{self.id} {node1} {node2} {self.parameters['resistance']}"


@dataclass
class PowerSource(Component):
    """电源元件"""

    def __init__(self, id: str, position: tuple, voltage: float):
        super().__init__(
            id,
            ComponentType.POWER_SOURCE,
            position,
            {
                "voltage": voltage,   # 直流电压或交流幅值（V_peak）
                "mode": "dc",         # 'dc' 或 'ac'；缺省为直流，保持向后兼容
                "frequency": 50.0,    # 仅在 mode='ac' 时使用，单位 Hz
            },
        )

    def get_spice_representation(self) -> str:
        # 修复：使用正确的双节点系统
        # 电压源连接在两个不同的节点之间
        if not hasattr(self, 'node_id') or self.node_id is None:
            return f"V{self.id} 0 0 {self.parameters['voltage']}"
        
        # 简化处理：假设电压源正极连接到分配的节点，负极连接到接地(0)
        node1 = self.node_id  # 正极
        node2 = 0  # 负极接地
        return f"V{self.id} {node1} {node2} {self.parameters['voltage']}"


@dataclass
class Ground(Component):
    """接地元件"""

    def __init__(self, id: str, position: tuple):
        super().__init__(id, ComponentType.GROUND, position)
        self.node_id = 0  # 接地固定为0号节点

    def get_spice_representation(self) -> str:
        return ""


@dataclass
class Oscilloscope(Component):
    """示波器元件（VR场景专用，支持信号测量和波形显示）"""

    def __init__(self, id: str, position: tuple, sampling_rate: float = 1000.0,
                 measured_node_ids: List[int] = None, channels: int = 2):
        """
        初始化示波器元件
        :param id: 元件唯一ID
        :param position: VR场景中坐标 (x, y, z)
        :param sampling_rate: 采样率（Hz）
        :param measured_node_ids: 测量的电路节点ID列表
        :param channels: 测量通道数
        """
        # 初始化基类
        super().__init__(
            id=id,
            type=ComponentType.OSCILLOSCOPE,
            position=position,
            parameters={
                "sampling_rate": sampling_rate,
                "measured_node_ids": measured_node_ids or [],
                "channels": channels,
                "time_range": 1.0,
                "voltage_range": 10.0,
                "trigger_level": 0.0,
                "coupling": "DC",
                # 运行时数据统一存入parameters
                "waveform_data": {},
                "measurements": {}
            }
        )

        # 参数校验
        self._validate_parameters()

    def _validate_parameters(self):
        """参数校验"""
        if not isinstance(self.parameters["measured_node_ids"], list):
            raise TypeError("measured_node_ids must be a list of integers")

        if not all(isinstance(nid, int) for nid in self.parameters["measured_node_ids"]):
            raise TypeError("all measured_node_ids must be integers")

        if self.parameters["channels"] < 1 or self.parameters["channels"] > 4:
            raise ValueError("channels must be 1-4")

        if self.parameters["coupling"] not in ["DC", "AC", "GND"]:
            raise ValueError("coupling must be DC/AC/GND")

        if self.parameters["sampling_rate"] <= 0:
            raise ValueError("sampling_rate must be positive")

    def get_spice_representation(self) -> str:
        """生成可执行的SPICE测量指令"""
        if not self.parameters["measured_node_ids"]:
            return f"* Oscilloscope {self.id}: no nodes to measure"

        # 生成瞬态分析指令
        time_step = 1.0 / self.parameters["sampling_rate"]
        time_stop = self.parameters["time_range"]

        measure_commands = [
            f"* Oscilloscope {self.id} measurements",
            f".tran {time_step:.6f} {time_stop:.3f}"  # 瞬态分析
        ]

        # 为每个测量节点生成测量指令
        for idx, node_id in enumerate(self.parameters["measured_node_ids"][:self.parameters["channels"]]):
            ch_name = f"CH{idx + 1}"
            measure_commands.extend([
                f".measure tran {ch_name}_MIN MIN V({node_id}) FROM=0 TO={time_stop}",
                f".measure tran {ch_name}_MAX MAX V({node_id}) FROM=0 TO={time_stop}",
                f".measure tran {ch_name}_AVG AVG V({node_id}) FROM=0 TO={time_stop}",
                f".measure tran {ch_name}_RMS RMS V({node_id}) FROM=0 TO={time_stop}"
            ])

        return "\n".join(measure_commands)

    def get_waveform(self, simulation_result: 'SimulationResult', channel: str) -> List[Dict[str, float]]:
        """从仿真结果读取指定通道波形（单一数据源）"""
        if (not simulation_result.oscilloscope_data or
                self.id not in simulation_result.oscilloscope_data):
            return []

        osc_data = simulation_result.oscilloscope_data[self.id]
        if channel not in osc_data["channels"]:
            return []

        ch_data = osc_data["channels"][channel]
        return [
            {"time": t, "voltage": v}
            for t, v in zip(ch_data["time_data"], ch_data["voltage_data"])
        ]

    def get_measurements(self, simulation_result: 'SimulationResult', channel: str) -> Dict[str, float]:
        """基于仿真结果计算测量值（计算型方法，不存储）"""
        import math  # 模块级导入优化

        waveform = self.get_waveform(simulation_result, channel)
        if not waveform:
            return {}

        # 提取数据
        voltages = [p["voltage"] for p in waveform]
        time_data = [p["time"] for p in waveform]

        # 边界处理
        if len(voltages) < 2:
            return {"dc_offset": voltages[0] if voltages else 0.0}

        v_min, v_max = min(voltages), max(voltages)
        time_span = time_data[-1] - time_data[0]

        # 基础测量
        measurements = {
            "peak_to_peak": v_max - v_min,
            "amplitude": (v_max - v_min) / 2,
            "dc_offset": sum(voltages) / len(voltages),
            "rms": math.sqrt(sum(v ** 2 for v in voltages) / len(voltages))
        }

        # 频率计算（仅适用于交流信号，避免直流信号误判）
        frequency = 0.0
        if v_min < -0.1 and v_max > 0.1 and time_span > 0:  # 确认为交流信号
            zero_crossings = sum(
                1 for i in range(1, len(voltages))
                if (voltages[i - 1] <= 0 <= voltages[i]) or (voltages[i - 1] >= 0 >= voltages[i])
            )
            if zero_crossings > 0:
                frequency = zero_crossings / (2 * time_span)

        measurements["frequency"] = frequency
        return measurements


@dataclass
class Connection:
    """
    元件连接关系
    
    支持端子化连接：当 terminal1_id / terminal2_id 存在时，表示导线连接的是
    元件1 的 terminal1 与 元件2 的 terminal2。VR 场景中应只在端子处允许连线。
    
    若未指定端子（向后兼容），则视为“元件整体相连”，节点分配将回退到旧逻辑。
    """
    id: str
    component1_id: str
    component2_id: str
    connection_point1: tuple  # 连接点坐标（用于导线渲染）
    connection_point2: tuple
    terminal1_id: Optional[str] = None  # 元件1 的端子，如 "positive", "terminal_1"
    terminal2_id: Optional[str] = None  # 元件2 的端子


@dataclass
class SimulationResult:
    """仿真结果"""
    timestamp: str
    circuit_data: Dict[str, Any]
    kvl_results: Optional[Dict] = None
    kcl_results: Optional[Dict] = None
    raw_data: Optional[Dict] = None
    # 增强类型约束的示波器数据
    oscilloscope_data: Optional[Dict[str, OscilloscopeData]] = None


@dataclass
class VoiceCommandContext:
    """语音指令上下文"""
    command_id: str
    timestamp: str
    original_text: str
    parsed_command: Dict[str, Any]
    execution_result: str
    referenced_components: List[str]  # 引用的元件ID


class DataInteractionLayer:
    """数据交互层主类"""

    def __init__(self):
        self.logger = logger

        # 数据存储
        self.components: Dict[str, Component] = {}
        self.connections: Dict[str, Connection] = {}
        self.simulation_results: List[SimulationResult] = []
        self.voice_context: List[VoiceCommandContext] = []

        # 上下文管理
        self.max_context_history = 5  # 最大历史指令数

        self.logger.info("数据交互层初始化完成")

    # === 元件管理接口 ===

    def add_component(self, component: Component) -> bool:
        """添加元件"""
        if component.id in self.components:
            self.logger.warning(f"元件ID {component.id} 已存在")
            return False

        # 示波器专用校验：测量节点是否存在
        if component.type == ComponentType.OSCILLOSCOPE:
            measured_nodes = component.parameters.get("measured_node_ids", [])
            if measured_nodes:
                # 获取当前电路中的有效节点ID
                valid_node_ids = set()
                for comp in self.components.values():
                    if comp.node_id is not None:
                        valid_node_ids.add(comp.node_id)

                # 检查是否有无效节点引用
                invalid_nodes = [nid for nid in measured_nodes if nid not in valid_node_ids]
                if invalid_nodes:
                    self.logger.warning(f"示波器 {component.id} 引用了不存在的节点: {invalid_nodes}")
                    # 可以选择抛出异常或仅警告，这里选择警告但仍允许添加
                    # 如果需要严格校验，可以改为：raise ValueError(f"Invalid node IDs: {invalid_nodes}")

        self.components[component.id] = component
        self.logger.info(f"添加元件: {component.id} ({component.type.value})")
        return True

    def remove_component(self, component_id: str) -> bool:
        """移除元件"""
        if component_id not in self.components:
            return False

        # 移除相关连接
        connections_to_remove = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.component1_id == component_id or conn.component2_id == component_id
        ]

        for conn_id in connections_to_remove:
            del self.connections[conn_id]

        del self.components[component_id]
        self.logger.info(f"移除元件: {component_id}")
        return True

    def get_component(self, component_id: str) -> Optional[Component]:
        """获取元件"""
        return self.components.get(component_id)

    def get_components_by_type(self, component_type: ComponentType) -> List[Component]:
        """按类型获取元件列表"""
        return [comp for comp in self.components.values() if comp.type == component_type]

    def get_all_components(self) -> List[Component]:
        """获取所有元件"""
        return list(self.components.values())

    # === 连接管理接口 ===

    def add_connection(self, connection: Connection) -> bool:
        """添加连接"""
        # 检查元件是否存在
        if (connection.component1_id not in self.components or
                connection.component2_id not in self.components):
            self.logger.warning("连接包含不存在的元件")
            return False

        self.connections[connection.id] = connection
        self.logger.info(f"添加连接: {connection.id}")
        return True

    def remove_connection(self, connection_id: str) -> bool:
        """移除连接"""
        if connection_id not in self.connections:
            return False

        del self.connections[connection_id]
        self.logger.info(f"移除连接: {connection_id}")
        return True

    def get_connections_for_component(self, component_id: str) -> List[Connection]:
        """获取元件的所有连接"""
        return [
            conn for conn in self.connections.values()
            if conn.component1_id == component_id or conn.component2_id == component_id
        ]
    
    def get_all_connections(self) -> List[Connection]:
        """获取所有连接"""
        return list(self.connections.values())

    # === 仿真结果管理 ===

    def add_simulation_result(self, result: SimulationResult) -> None:
        """添加仿真结果"""
        self.simulation_results.append(result)
        self.logger.info("添加新的仿真结果")

    def get_latest_simulation_result(self) -> Optional[SimulationResult]:
        """获取最新仿真结果"""
        if not self.simulation_results:
            return None
        return self.simulation_results[-1]

    def has_new_simulation_results(self) -> bool:
        """检查是否有新仿真结果"""
        # 这里可以添加更复杂的逻辑来检测新结果
        return len(self.simulation_results) > 0

    def get_simulation_results(self) -> List[SimulationResult]:
        """获取所有仿真结果"""
        return self.simulation_results

    # === 语音指令上下文管理 ===

    def add_voice_context(self, context: VoiceCommandContext) -> None:
        """添加上下文记录"""
        self.voice_context.append(context)

        # 限制历史记录数量
        if len(self.voice_context) > self.max_context_history:
            self.voice_context = self.voice_context[-self.max_context_history:]

        self.logger.info(f"添加上下文: {context.command_id}")

    def get_recent_voice_context(self, limit: int = 3) -> List[VoiceCommandContext]:
        """获取最近的上下文记录"""
        return self.voice_context[-limit:]

    def get_context_by_component(self, component_id: str) -> List[VoiceCommandContext]:
        """获取涉及特定元件的上下文记录"""
        return [
            ctx for ctx in self.voice_context
            if component_id in ctx.referenced_components
        ]

    # === 电路数据导出 ===

    def get_circuit_snapshot(self) -> Dict[str, Any]:
        """
        获取电路快照（供仿真层使用）
        
        返回电路的当前状态，包括所有元件和连接
        这是数据层提供给仿真层的标准接口
        """
        return {
            'components': dict(self.components),
            'connections': dict(self.connections),
            'timestamp': datetime.now().isoformat()
        }

    # === 定理验证支持 ===

    def get_circuit_data_for_verification(self) -> Dict[str, Any]:
        """获取用于定理验证的电路数据"""
        return {
            "components": {comp.id: {
                "type": comp.type.value,
                "parameters": comp.parameters,
                "node_id": comp.node_id
            } for comp in self.components.values()},
            "connections": {conn.id: {
                "component1": conn.component1_id,
                "component2": conn.component2_id
            } for conn in self.connections.values()}
        }

    def clear_all_data(self) -> None:
        """清空所有数据"""
        self.components.clear()
        self.connections.clear()
        self.simulation_results.clear()
        self.voice_context.clear()
        self.logger.info("清空所有数据")


# 单例模式
data_layer_instance = None


def get_data_layer() -> DataInteractionLayer:
    """获取数据交互层实例（单例模式）"""
    global data_layer_instance
    if data_layer_instance is None:
        data_layer_instance = DataInteractionLayer()
    return data_layer_instance
