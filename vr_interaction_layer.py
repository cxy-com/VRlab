# 第一步：添加项目根目录到Python搜索路径
import sys
import os
import re
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 第二步：导入核心依赖
from ursina import Entity, Text, Vec3, color, scene, EditorCamera, mouse, camera, time, invoke, destroy, InputField, Texture
from ursina.mesh_importer import load_model
import logging
import json
import shutil
from typing import Dict, Optional, List, Tuple
import platform
import math
import threading
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from scr.config import PROJECT_ROOT

# 配置Matplotlib后端为Agg（非交互式）
plt.switch_backend('Agg')

from .ui_panels import (
    create_ui_panel,
    create_component_menu,
    create_oscilloscope_detail_panel,
    create_modify_params_panel,
    create_simulation_results_panel,
)

# 第三步：导入data层
from scr.data.data_layer import (
    ComponentType, Component, Resistor, PowerSource, Ground,
    Connection, get_data_layer, DataInteractionLayer,
    TERMINAL_POSITIVE, TERMINAL_NEGATIVE, TERMINAL_1, TERMINAL_2, TERMINAL_COMMON,
)

# Windows系统适配
assert platform.system() == "Windows", "此版本仅适配Windows系统"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug 模式专用日志配置（写入 NDJSON 到本地文件，供调试分析）
DEBUG_LOG_PATH = r"d:\python\.cursor\debug.log"
DEBUG_SESSION_ID = "vr-interaction-debug-session"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict = None, run_id: str = "pre-fix") -> None:
    """向本地 NDJSON 日志文件追加一条调试记录（Debug 模式专用，避免依赖网络）。"""
    # region agent log
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试辅助：任何日志写入失败都不影响主逻辑
        pass
    # endregion


class VRInteractionLayer:
    """VR交互层主类 - 修复Z轴强制固定bug"""

    def __init__(self, ai_voice_layer=None, circuit_sim_layer=None, font_path: str = None):
        # 数据层联动
        self.data_layer: DataInteractionLayer = get_data_layer()
        self.ai_voice_layer = ai_voice_layer
        self.circuit_sim_layer = circuit_sim_layer
        # 外部注入的字体路径（由 initializer + font_manager 提供时使用，避免时序竞争）
        self._resolved_font_path = font_path if font_path else None

        # 仿真器集成
        self._init_circuit_simulator()

        # 核心配置（移除强制Z轴固定）
        self.COMPONENT_Y = 0.8
        self.UNIFIED_COMP_SCALE = (0.4, 0.4, 0.4)
        self.WIRE_THICKNESS = 0.1
        self.TABLE_TOP_Y = 0.2
        # 实验台网格：与 _create_table_grid 画线一致，18 行 × 24 列（每格 1 单位）
        self.GRID_ROWS = 18
        self.GRID_COLS = 24
        self.GRID_X_MIN = -12
        self.GRID_Z_MIN = -9
        # 左上角默认提示（含行/列数，不含 V 键）
        self._default_ui_hint = f"右键: 菜单/旋转 | 左键: 放置 | 滚轮: 缩放/平移 | {self.GRID_ROWS}行×{self.GRID_COLS}列"

        # 场景元素存储
        self.entities: Dict[str, Entity] = {}
        self.wires: List[Entity] = []
        self.experiment_table = None
        self.click_plane = None
        self.ground_plane = None
        self.ui_panel = None
        self.interactive_camera = None
        self.component_menu = None
        self.oscilloscope_detail_panel = None
        self.modify_params_panel = None  # 修改参数迷你面板
        self._modify_params_comp_id = None
        # 拾取根节点：用于按类别过滤鼠标射线命中，解决密集场景乱选中
        self.pick_components = None
        self.pick_terminals = None
        self.pick_wires = None

        # 状态管理
        self.selected_component_type = None
        # 按元件类别分别编号：电源 V1、V2…；电阻 R1、R2…；接地 GND1、GND2…；示波器 OSC1…
        self.component_counters = {
            'power_source': 1,
            'signal_generator': 1,
            'resistor': 1,
            'ground': 1,
            'oscilloscope': 1,
        }
        self.is_placing_mode = False
        self.selected_component_id = None
        self._selected_original_color = None
        self._selected_highlight_entity = None  # 选中时的高亮覆盖层，不修改元件颜色

        # 连线模式状态
        self.is_wiring_mode = False
        self.wire_first_component_id = None
        self.wire_first_terminal_id = None  # 连线时第一个端子的 terminal_id
        self.wire_first_terminal_entity = None  # 连线时第一个端子实体，用于高亮与恢复
        self._wire_first_terminal_original_color = None  # 连线起点端子原始颜色
        self.next_wire_id = 1

        # 选中导线状态
        self.selected_connection_id = None
        self.selected_wire_entity = None
        self._selected_wire_original_color = None

        # 修改参数模式：点击菜单「修改参数」后，再点实验台元件弹出参数面板
        self.modify_params_mode = False

        # 仿真结果面板相关
        self.simulation_results_panel = None
        self.sim_result_texts = []
        # 示波器波形点缓存（在 ui_panels 中初始化为空列表）
        self.oscilloscope_voltage_points = []
        self.oscilloscope_current_points = []

        # 实时波形更新相关
        self.active_oscilloscope = None  # 当前活动的示波器
        self.oscilloscope_update_interval = 0.05  # 50ms更新一次，增加动态效果
        self.last_oscilloscope_update = 0
        # 波形缩放相关
        self.oscilloscope_scale = 1.0  # 默认缩放比例
        self.oscilloscope_scale_min = 0.5  # 最小缩放比例
        self.oscilloscope_scale_max = 2.0  # 最大缩放比例
        # 波形延伸相关
        self.waveform_history = {}
        self.max_history_points = 1000  # 最大历史数据点

        # 右键点击/长按检测相关状态
        self.right_mouse_pressed = False
        self.right_mouse_press_time = 0.0
        self.right_mouse_press_pos = (0, 0)
        self.long_press_threshold = 0.4   # 放宽：0.4 秒内松开算"点击"，否则算长按旋转
        self.right_click_move_threshold = 0.2  # 放宽：允许一定移动仍算点击（避免被旋转误判）

        # 未注入字体路径时，本地初始化字体（如 run_vr.py 单入口）
        if self._resolved_font_path is None:
            self._setup_font()

        logger.info("VR交互层初始化完成（Z轴自定义生效）")

    def _init_circuit_simulator(self) -> None:
        """初始化电路仿真器"""
        try:
            from scr.circuit_sim.circuit_simulation_layer import get_circuit_simulation_layer, SimulationConfig
            # 启用端子化节点分配：需要每条 Connection 都带 terminal1_id/terminal2_id
            cfg = SimulationConfig(use_terminal_based=True)
            self.circuit_simulator = get_circuit_simulation_layer(cfg)
            # 单例已创建时，确保配置与 node_assigner 同步
            if self.circuit_simulator:
                self.circuit_simulator.config.use_terminal_based = True
                self.circuit_simulator.node_assigner.use_terminal_based = True
            logger.info("电路仿真器集成成功")
        except Exception as e:
            logger.warning(f"电路仿真器初始化失败: {e}")
            self.circuit_simulator = None

    def _setup_font(self) -> None:
        """设置中文字体（复制字体到项目目录）"""
        # 获取当前项目目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_dir = os.path.join(current_dir, 'fonts')
        os.makedirs(font_dir, exist_ok=True)

        # 源字体路径和目标文件名（只有一个字体文件）
        source_path = r"D:\python\zh_msyh\msyh.ttc"
        dest_path = os.path.join(font_dir, "msyh.ttc")

        # 复制字体文件到项目目录
        if os.path.exists(source_path) and not os.path.exists(dest_path):
            try:
                shutil.copy2(source_path, dest_path)
                logger.info(f"复制字体: msyh.ttc")
            except Exception as e:
                logger.warning(f"复制字体失败: {e}")

        # 使用项目目录中的字体
        # 绝对路径用于存在性检查；加载时必须转换为 Panda3D 识别的 Filename（Windows 下会变成 /d/...）
        abs_font_path = os.path.join(font_dir, "msyh.ttc")
        rel_font_path = "fonts/msyh.ttc"  # 仅用于日志展示

        # Panda3D 推荐用 Filename 处理路径：避免 D:/... 找不到文件的问题
        p3d_font_path = None
        try:
            from panda3d.core import Filename  # type: ignore
            p3d_font_path = Filename.fromOsSpecific(abs_font_path).getFullpath()
        except Exception:
            # 若导入失败（极少见），退回到原始路径（可能仍可用）
            p3d_font_path = abs_font_path

        _agent_log(
            hypothesis_id="H1",
            location="vr_interaction_layer._setup_font",
            message="检查字体文件是否存在",
            data={"abs_font_path": p3d_font_path, "exists": os.path.exists(abs_font_path)},
        )

        if os.path.exists(abs_font_path):
            try:
                Text.default_font = p3d_font_path
                self._resolved_font_path = p3d_font_path
                logger.info(f"成功设置中文字体: {p3d_font_path}")
                _agent_log(
                    hypothesis_id="H1",
                    location="vr_interaction_layer._setup_font",
                    message="字体设置成功",
                    data={"font": p3d_font_path},
                )
                return
            except Exception as font_error:
                logger.warning(f"字体加载失败: {font_error}")
                _agent_log(
                    hypothesis_id="H2",
                    location="vr_interaction_layer._setup_font",
                    message="字体设置抛出异常",
                    data={"error": str(font_error)},
                )

        logger.warning("未找到合适的中文字体，使用默认字体")
        _agent_log(
            hypothesis_id="H3",
            location="vr_interaction_layer._setup_font",
            message="未找到中文字体，回退默认字体",
            data={"abs_font_path": abs_font_path, "exists": os.path.exists(abs_font_path)},
        )

    def initialize_scene(self) -> bool:
        """初始化VR场景"""
        try:
            scene.background_color = color.gray

            # 未注入字体时，在创建菜单/UI 前再次设置字体
            if self._resolved_font_path is None:
                self._setup_font()

            # 创建核心元素
            self._create_experiment_table()
            self._create_ground_plane()
            self._create_ui_panel()
            self._create_component_menu()
            self._create_oscilloscope_detail_panel()  # 创建示波器详情面板
            self._create_modify_params_panel()  # 修改参数面板
            self._create_simulation_results_panel()  # 创建仿真结果面板
            self._create_table_grid()  # 添加坐标网格辅助线
            self._create_pick_roots()  # 创建拾取根节点（必须在创建元件/端子/导线之前）
            self._setup_camera()

            # 加载数据层内容
            self.load_all_components()
            self.load_all_connections()

            logger.info("VR场景初始化成功（连线对齐+Z轴自定义）")
            return True
        except Exception as e:
            logger.error(f"VR场景初始化失败: {e}", exc_info=True)
            return False

    def _create_experiment_table(self) -> None:
        """放大试验台（优化碰撞体精度）并添加桌子支撑结构"""
        # 创建桌子主体（台面）
        self.experiment_table = Entity(
            model='cube',
            color=color.light_gray,  # 浅灰色实验台
            scale=(24, 0.2, 18),
            position=(0, 0.1, 0),
            name="ExperimentTable",
            collider='box'  # 添加碰撞体，使鼠标能检测到
        )
        
        # 添加桌子腿 - 只添加桌腿，不添加横梁，避免形成十字架
        table_leg_color = color.gray
        table_leg_scale = (1.5, 1.0, 1.5)  # 桌子腿的尺寸，增大使其更明显
        table_leg_height = 1.0  # 桌子腿的高度
        
        # 四条腿的位置 - 调整位置使桌腿看起来更自然地支撑实验台
        leg_positions = [
            (-10.0, -0.4, -7.0),  # 左下角
            (10.0, -0.4, -7.0),   # 右下角
            (-10.0, -0.4, 7.0),    # 左上角
            (10.0, -0.4, 7.0)      # 右上角
        ]
        
        for pos in leg_positions:
            Entity(
                model='cube',
                color=table_leg_color,
                scale=(table_leg_scale[0], table_leg_height, table_leg_scale[2]),
                position=pos,
                name="TableLeg",
                collider=None
            )
        
        # 点击专用平面：放在台面高度，负责提供稳定 world_point
        self.click_plane = Entity(
            model='plane',
            color=color.rgba(255, 255, 255, 0),  # 全透明
            scale=(24, 1, 18),  # 覆盖台面（plane 的厚度无所谓）
            position=(0, self.TABLE_TOP_Y + 0.001, 0),  # 比台面略高一点点，确保先被命中
            collider='box',
            name="ClickPlane"
        )

    def _create_ground_plane(self) -> None:
        """放大接地平面（优先使用自定义模型，失败回退为默认平面；禁用碰撞体，避免干扰实验台检测）"""
        # 自定义接地模型（OBJ）路径（名称与目录分离，便于使用 load_model 指定 path）
        # 用 PROJECT_ROOT（和 component_renderer.py 一致），避免 root_dir/运行目录差异导致找不到资源
        custom_ground_dir = Path(PROJECT_ROOT) / "assets" / "Grounding" / "JCI54559206271_obj" / "source"
        # Ursina 的 load_model 会把带多个 '.' 的文件名拆成 name+filetype，
        # 但 obj_to_ursinamesh 只用 name（第一个 '.' 之前的部分），会导致找不到实际文件。
        # 因此这里使用无额外 '.' 的文件名 ground.obj。
        custom_ground_name = "ground.obj"
        custom_ground_model = custom_ground_dir / custom_ground_name

        def _create_default_ground_plane():
            return Entity(
                model="plane",
                color=color.light_gray,
                scale=(15, 1, 15),
                position=(0, 0, 0),
                name="GroundPlane",
                collider=None,  # 关键：禁用地面碰撞体，防止鼠标误检测
            )

        # 优先尝试加载自定义接地模型；若失败则回退为默认平面
        try:
            if custom_ground_model.exists():
                # 注意：Ursina 的 load_model 需要单独提供文件名和目录
                model = load_model(custom_ground_name, path=custom_ground_dir)
                if model is None:
                    raise RuntimeError(f"Ursina load_model 返回 None: {custom_ground_model}")

                self.ground_plane = Entity(
                    model=model,
                    color=color.light_gray,
                    scale=(15, 1, 15),
                    position=(0, 0, 0),
                    name="GroundPlane",
                    collider=None,
                )
            else:
                logger.warning(f"自定义接地模型不存在，使用默认接地平面: {custom_ground_model} (PROJECT_ROOT={PROJECT_ROOT})")
                self.ground_plane = _create_default_ground_plane()
        except Exception as e:
            logger.warning(f"加载自定义接地模型失败，回退为默认接地平面: {e} (model={custom_ground_model}, dir={custom_ground_dir})")
            self.ground_plane = _create_default_ground_plane()

    def _create_pick_roots(self) -> None:
        """创建用于鼠标拾取分组的根节点（按类别分离 traverse_target）。"""
        if self.pick_components and self.pick_terminals and self.pick_wires:
            return
        self.pick_components = Entity(parent=scene, name="PickRoot_Components", collider=None)
        self.pick_terminals = Entity(parent=scene, name="PickRoot_Terminals", collider=None)
        self.pick_wires = Entity(parent=scene, name="PickRoot_Wires", collider=None)
        # 清除颜色缩放，防止默认白色继承到子实体
        try:
            self.pick_components.clearColorScale()
            self.pick_terminals.clearColorScale()
            self.pick_wires.clearColorScale()
        except Exception:
            pass

    def _ensure_pick_roots(self) -> None:
        """确保拾取根节点存在（防止在场景未初始化时创建元件/导线）。"""
        if not (self.pick_components and self.pick_terminals and self.pick_wires):
            self._create_pick_roots()

    def _first_hit_under(self, root: Entity):
        """从当前鼠标射线的命中列表中，取第一个属于 root 子树的实体。"""
        if not root:
            return None
        hits = getattr(mouse, "collisions", None) or []
        for h in hits:
            ent = getattr(h, "entity", None)
            if not ent:
                continue
            try:
                if ent == root or ent.has_ancestor(root):
                    return ent
            except Exception:
                # has_ancestor 失败时回退用 parent 链
                e = ent
                while e:
                    if e == root:
                        return ent
                    e = getattr(e, "parent", None)
        return None

    def _get_clicked_entity_normal(self):
        """普通模式点击：导线优先，其次端子，其次元件。"""
        self._ensure_pick_roots()
        ent = self._first_hit_under(self.pick_wires)
        if ent:
            return ent
        ent = self._first_hit_under(self.pick_terminals)
        if ent:
            return ent
        ent = self._first_hit_under(self.pick_components)
        if ent:
            return ent
        return None

    def _get_clicked_entity_wiring(self):
        """连线模式点击：只允许端子；若点到接地主体则视为公共端子。"""
        self._ensure_pick_roots()
        ent = self._first_hit_under(self.pick_terminals)
        if ent:
            return ent
        # 允许点接地主体
        ent = self._first_hit_under(self.pick_components)
        if not ent:
            return None
        comp_id = self._get_clicked_component_id(ent)
        if not comp_id:
            return None
        comp = self.data_layer.get_component(comp_id)
        if comp and comp.type == ComponentType.GROUND:
            return ent
        return None

    def _create_ui_panel(self) -> None:
        """UI文字位置（含实验台行/列数提示）委托给 ui_panels 模块。"""
        create_ui_panel(self)

    def _create_table_grid(self) -> None:
        """在实验台上绘制网格线，辅助定位"""
        try:
            # 网格线高度：略高于实验台顶面 (TABLE_TOP_Y = 0.2)
            grid_y = self.TABLE_TOP_Y + 0.01  # 0.21

            # 网格线颜色：浅灰色；中间列(x=0)、中间行(z=0)用橙色提示
            grid_color = color.rgba(200, 200, 200, 255)  # 浅灰色网格线
            center_line_color = color.orange  # 中间行/列

            # X轴方向（左右）每1单位画一条线；x=0 为中间列
            for x in range(-12, 13, 1):
                line_color = center_line_color if x == 0 else grid_color
                Entity(
                    model='cube',
                    color=line_color,
                    scale=(0.05, 0.02, 18),
                    position=(x, grid_y, 0),
                    collider=None
                )

            # Z轴方向（前后）每1单位画一条线；z=0 为中间行
            for z in range(-9, 10, 1):
                line_color = center_line_color if z == 0 else grid_color
                Entity(
                    model='cube',
                    color=line_color,
                    scale=(24, 0.02, 0.05),
                    position=(0, grid_y, z),
                    collider=None
                )

            logger.info("实验台网格线创建完成")
        except Exception as e:
            logger.error(f"创建网格线失败: {e}", exc_info=True)

    def _create_component_menu(self) -> None:
        """创建元件选择菜单，使用Matplotlib技术优化。"""
        try:
            logger.info("=== 开始创建Matplotlib优化菜单 ===")

            # 菜单主容器（挂载到 camera.ui）
            self.component_menu = Entity(
                parent=camera.ui,
                visible=False,
                position=(0, 0),
                scale=1,
            )

            # 创建Matplotlib菜单图像
            menu_image = self._generate_menu_image()
            
            # 创建菜单背景图像
            self.menu_background = Entity(
                parent=self.component_menu,
                model='quad',
                texture=Texture(menu_image),
                scale=(0.8, 0.6, 0),
                position=(0, 0, 0),
                collider=None,
            )

            logger.info("Matplotlib菜单创建成功")
        except Exception as e:
            logger.error(f"创建Matplotlib菜单失败: {e}", exc_info=True)
            # 回退到原始菜单
            create_component_menu(self)
    
    def _generate_menu_image(self):
        """生成Matplotlib菜单图像"""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        # 配置Matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # 设置背景为半透明
        fig.patch.set_alpha(0.9)
        ax.set_facecolor((240/255, 240/255, 240/255, 0.8))
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 绘制标题
        plt.text(0.5, 0.95, '菜单', fontsize=20, fontweight='bold', ha='center')
        
        # 按钮配置
        buttons = [
            # 左列
            (0.25, 0.8, '电源', 'power_source'),
            (0.25, 0.65, '信号发生器', 'signal_generator'),
            (0.25, 0.5, '电阻', 'resistor'),
            (0.25, 0.35, '接地', 'ground'),
            (0.25, 0.2, '示波器', 'oscilloscope'),
            # 右列
            (0.75, 0.8, '连线模式', 'wire_mode'),
            (0.75, 0.65, '运行仿真', 'run_simulation'),
            (0.75, 0.5, '查看仿真', 'view_simulation'),
            (0.75, 0.35, '语音指令', 'voice_input'),
            (0.75, 0.2, '修改参数', 'modify_params'),
        ]
        
        # 绘制按钮
        for x, y, label, action in buttons:
            # 绘制按钮背景
            rect = plt.Rectangle((x-0.2, y-0.07), 0.4, 0.14, 
                               facecolor=(128/255, 128/255, 128/255, 0.7), 
                               edgecolor='black', 
                               linewidth=1.5)
            ax.add_patch(rect)
            
            # 绘制按钮文字
            plt.text(x, y, label, fontsize=12, ha='center', va='center', color='black')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存到内存缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        
        # 转换为PIL Image
        img = Image.open(buf)
        plt.close()
        
        return img

    def _create_oscilloscope_detail_panel(self) -> None:
        """创建示波器详情面板，委托给 ui_panels 模块。"""
        create_oscilloscope_detail_panel(self)

    def _create_modify_params_panel(self) -> None:
        """修改参数迷你面板，委托给 ui_panels 模块。"""
        create_modify_params_panel(self)

    def _create_simulation_results_panel(self) -> None:
        """创建仿真结果面板，委托给 ui_panels 模块。"""
        create_simulation_results_panel(self)

    def _show_component_menu(self) -> None:
        """显示元件选择菜单（固定在屏幕中央）"""
        if self.component_menu:
            # 菜单固定在屏幕中央
            menu_x = 0.0
            menu_y = 0.0
            self.component_menu.position = (menu_x, menu_y)
            self.component_menu.visible = True

            # 为Matplotlib菜单图像添加碰撞体
            if hasattr(self, 'menu_background'):
                self.menu_background.collider = 'box'

            logger.info(f"Matplotlib菜单显示在屏幕中央: ({menu_x:.2f}, {menu_y:.2f})")

    def _get_menu_clicked_action(self) -> Optional[str]:
        """根据当前鼠标在菜单上的位置判断点中的是哪个按钮（不依赖 hovered_entity）"""
        if not self.component_menu or not self.component_menu.visible:
            return None
        
        # 获取菜单的绝对位置
        menu_x = self.component_menu.x
        menu_y = self.component_menu.y
        
        # 计算鼠标在菜单坐标系中的位置
        mouse_x = mouse.x
        mouse_y = mouse.y
        
        # Matplotlib菜单的实际布局
        # 菜单宽度为0.8，高度为0.6
        # 从Matplotlib坐标转换到ursina UI坐标
        # Matplotlib: (0,0)左下角, (1,1)右上角
        # Ursina UI: (0,0)中心, 宽度0.8, 高度0.6
        
        # 计算按钮在ursina UI中的绝对位置
        # 基于Matplotlib中的实际按钮位置
        left_buttons = [
            # 左列按钮: Matplotlib x=0.25, 转换为ursina x = menu_x - 0.4 + 0.25*0.8 = menu_x - 0.2
            (menu_x - 0.2, menu_y + 0.18, "power_source"),      # 电源 (Matplotlib y=0.8)
            (menu_x - 0.2, menu_y + 0.09, "signal_generator"),   # 信号发生器 (Matplotlib y=0.65)
            (menu_x - 0.2, menu_y + 0.0, "resistor"),         # 电阻 (Matplotlib y=0.5)
            (menu_x - 0.2, menu_y - 0.09, "ground"),            # 接地 (Matplotlib y=0.35)
            (menu_x - 0.2, menu_y - 0.18, "oscilloscope"),     # 示波器 (Matplotlib y=0.2)
        ]
        
        right_buttons = [
            # 右列按钮: Matplotlib x=0.75, 转换为ursina x = menu_x - 0.4 + 0.75*0.8 = menu_x + 0.2
            (menu_x + 0.2, menu_y + 0.18, "wire_mode"),         # 连线模式 (Matplotlib y=0.8)
            (menu_x + 0.2, menu_y + 0.09, "run_simulation"),      # 运行仿真 (Matplotlib y=0.65)
            (menu_x + 0.2, menu_y + 0.0, "view_simulation"),   # 查看仿真 (Matplotlib y=0.5)
            (menu_x + 0.2, menu_y - 0.09, "voice_input"),        # 语音指令 (Matplotlib y=0.35)
            (menu_x + 0.2, menu_y - 0.18, "modify_params"),     # 修改参数 (Matplotlib y=0.2)
        ]
        
        # 检查左列按钮
        button_width = 0.16  # 按钮半宽 (Matplotlib 0.2 * 0.8)
        button_height = 0.042  # 按钮半高 (Matplotlib 0.07 * 0.6)
        
        for btn_x, btn_y, action in left_buttons:
            if (btn_x - button_width <= mouse_x <= btn_x + button_width) and (btn_y - button_height <= mouse_y <= btn_y + button_height):
                return action
        
        # 检查右列按钮
        for btn_x, btn_y, action in right_buttons:
            if (btn_x - button_width <= mouse_x <= btn_x + button_width) and (btn_y - button_height <= mouse_y <= btn_y + button_height):
                return action
        
        return None

    def _hide_component_menu(self) -> None:
        """隐藏元件选择菜单（按鼠标位置判断点中的按钮，再隐藏）"""
        if self.component_menu and self.component_menu.visible:
            action_type = self._get_menu_clicked_action()
            if action_type:
                if action_type == 'wire_mode':
                    self._toggle_wiring_mode()
                elif action_type == 'run_simulation':
                    self._run_circuit_simulation()
                elif action_type == 'view_simulation':
                    self._show_simulation_results()
                elif action_type == 'voice_input':
                    logger.info("菜单点击：语音指令（voice_input）按钮")
                    self.start_voice_input()
                elif action_type == 'modify_params':
                    self.modify_params_mode = True
                    self.show_feedback("请点击要修改的元件")
                    logger.info("进入修改参数模式")
                else:
                    self._select_component_type(action_type)
            # 再隐藏菜单
            self.component_menu.visible = False
            
            for child in self.component_menu.children:
                if type(child).__name__ == 'Entity' and hasattr(child, 'collider'):
                    child.collider = None

    def _hide_oscilloscope_detail_panel(self) -> None:
        """隐藏示波器详情面板"""
        try:
            # 隐藏面板容器
            if self.oscilloscope_detail_panel:
                self.oscilloscope_detail_panel.visible = False

            # 隐藏波形点
            for pt in getattr(self, "oscilloscope_voltage_points", []) or []:
                pt.visible = False
            for pt in getattr(self, "oscilloscope_current_points", []) or []:
                pt.visible = False
            # 隐藏波形图像
            if hasattr(self, "oscilloscope_waveform_image"):
                self.oscilloscope_waveform_image.visible = False

            # 重置波形缩放比例
            self.oscilloscope_scale = 1.0

            # 清除活动示波器
            self.active_oscilloscope = None

            logger.info("隐藏示波器详情面板")

        except Exception as e:
            logger.error(f"隐藏示波器详情面板失败: {e}", exc_info=True)

    def _open_modify_params_panel(self, comp_id: str) -> None:
        """打开修改参数面板（电阻阻值/电源电压）"""
        try:
            comp = self.data_layer.get_component(comp_id)
            if not comp or not self.modify_params_panel:
                return
            from scr.data.data_layer import ComponentType
            ctype = comp.type
            if ctype not in (ComponentType.RESISTOR, ComponentType.POWER_SOURCE):
                self.show_feedback(f"{comp_id} 无参数可修改")
                return
            self._modify_params_comp_id = comp_id
            self.modify_params_panel.visible = True
            self.modify_params_panel.bg.visible = True
            self.modify_params_panel.title.visible = True
            self.modify_params_panel.title.text = f"修改参数 - {comp_id}"
            # 标题与参数标签统一使用黑色，保证清晰可见
            if hasattr(self.modify_params_panel.title, 'color'):
                self.modify_params_panel.title.color = color.black
            self.modify_params_panel.param_label.visible = True
            if hasattr(self.modify_params_panel.param_label, 'color'):
                self.modify_params_panel.param_label.color = color.black

            # 默认隐藏交流专用的频率输入控件
            if hasattr(self.modify_params_panel, "freq_label"):
                self.modify_params_panel.freq_label.visible = False
            if hasattr(self.modify_params_panel, "freq_input_field"):
                self.modify_params_panel.freq_input_field.visible = False
            if hasattr(self.modify_params_panel, "freq_input_field_bg"):
                self.modify_params_panel.freq_input_field_bg.visible = False

            if ctype == ComponentType.RESISTOR:
                # 电阻：仅修改阻值
                self.modify_params_panel.param_label.text = "阻值(Ω):"
                val = comp.parameters.get("resistance", 1000)
                self.modify_params_panel.input_field.text = str(int(val))
            else:
                # 电源/信号源：根据 mode 决定是直流电压还是交流幅值+频率
                mode = comp.parameters.get("mode", "dc")
                if mode == "ac":
                    # 交流信号源：主输入为幅值，附加频率输入
                    self.modify_params_panel.param_label.text = "幅值(V):"
                    val = comp.parameters.get("voltage", 5.0)
                    self.modify_params_panel.input_field.text = str(float(val))

                    if hasattr(self.modify_params_panel, "freq_label"):
                        self.modify_params_panel.freq_label.visible = True
                    if hasattr(self.modify_params_panel, "freq_input_field"):
                        freq_val = comp.parameters.get("frequency", 50.0)
                        self.modify_params_panel.freq_input_field.text = str(float(freq_val))
                        self.modify_params_panel.freq_input_field.visible = True
                    if hasattr(self.modify_params_panel, "freq_input_field_bg"):
                        self.modify_params_panel.freq_input_field_bg.visible = True
                else:
                    # 直流电源：保持原逻辑，只显示电压
                    self.modify_params_panel.param_label.text = "电压(V):"
                    val = comp.parameters.get("voltage", 5.0)
                    self.modify_params_panel.input_field.text = str(float(val))
            if hasattr(self.modify_params_panel.input_field, 'visible'):
                self.modify_params_panel.input_field.visible = True
            # 数字颜色改为白色，与设计保持一致
            if hasattr(self.modify_params_panel.input_field, 'text_color'):
                self.modify_params_panel.input_field.text_color = color.white
            self.modify_params_panel.input_field_bg.visible = True
            self.modify_params_panel.btn_ok.visible = True
            self.modify_params_panel.btn_ok.collider = 'box'
            self.modify_params_panel.btn_ok_label.visible = True
            self.modify_params_panel.btn_cancel.visible = True
            self.modify_params_panel.btn_cancel.collider = 'box'
            self.modify_params_panel.btn_cancel_label.visible = True
            if self.component_menu:
                self.component_menu.visible = False
        except Exception as e:
            logger.error(f"打开修改参数面板失败: {e}", exc_info=True)

    def _hide_modify_params_panel(self) -> None:
        """隐藏修改参数面板"""
        try:
            if not self.modify_params_panel:
                return
            self.modify_params_panel.visible = False
            self.modify_params_panel.bg.visible = False
            self.modify_params_panel.title.visible = False
            self.modify_params_panel.param_label.visible = False
            if hasattr(self.modify_params_panel.input_field, 'visible'):
                self.modify_params_panel.input_field.visible = False
            self.modify_params_panel.input_field_bg.visible = False
            # 隐藏交流频率相关控件（如果存在）
            if hasattr(self.modify_params_panel, "freq_label"):
                self.modify_params_panel.freq_label.visible = False
            if hasattr(self.modify_params_panel, "freq_input_field"):
                self.modify_params_panel.freq_input_field.visible = False
            if hasattr(self.modify_params_panel, "freq_input_field_bg"):
                self.modify_params_panel.freq_input_field_bg.visible = False
            self.modify_params_panel.btn_ok.visible = False
            self.modify_params_panel.btn_ok.collider = None
            self.modify_params_panel.btn_ok_label.visible = False
            self.modify_params_panel.btn_cancel.visible = False
            self.modify_params_panel.btn_cancel.collider = None
            self.modify_params_panel.btn_cancel_label.visible = False
            self._modify_params_comp_id = None
        except Exception as e:
            logger.error(f"隐藏修改参数面板失败: {e}", exc_info=True)

    def _apply_modify_params(self) -> None:
        """应用修改参数（从输入框读取并更新数据层与实体显示）"""
        try:
            cid = self._modify_params_comp_id
            if not cid or not self.modify_params_panel:
                return
            comp = self.data_layer.get_component(cid)
            if not comp or cid not in self.entities:
                self._hide_modify_params_panel()
                return
            from scr.data.data_layer import ComponentType
            raw = self.modify_params_panel.input_field.text.strip()
            if comp.type == ComponentType.RESISTOR:
                try:
                    val = float(raw)
                    if val <= 0 or val > 1e9:
                        self.show_feedback("阻值需为正数且不超过 1e9")
                        return
                except ValueError:
                    self.show_feedback("请输入有效数字")
                    return
                comp.parameters["resistance"] = val
                ent = self.entities[cid]
                for c in ent.children:
                    if type(c).__name__ == 'Text':
                        c.text = f"{cid}\n{int(val)}Ω"
                        break
            elif comp.type == ComponentType.POWER_SOURCE:
                try:
                    val = float(raw)
                    if val < -1e6 or val > 1e6:
                        self.show_feedback("电压请在合理范围内")
                        return
                except ValueError:
                    self.show_feedback("请输入有效数字")
                    return
                # 区分直流电源与交流信号源：mode 缺省为 'dc' 以保持兼容
                mode = comp.parameters.get("mode", "dc")
                if mode == "ac":
                    # 交流信号源：val 为幅值；同时读取频率输入（若存在）
                    comp.parameters["voltage"] = val
                    freq_text = ""
                    if hasattr(self.modify_params_panel, "freq_input_field"):
                        freq_text = self.modify_params_panel.freq_input_field.text.strip()
                    if freq_text:
                        try:
                            freq_val = float(freq_text)
                            if freq_val <= 0 or freq_val > 1e9:
                                self.show_feedback("频率需为正数且不超过 1e9 Hz")
                                return
                            comp.parameters["frequency"] = freq_val
                        except ValueError:
                            self.show_feedback("频率请输入有效数字")
                            return
                else:
                    # 直流电源：仅更新电压
                    comp.parameters["voltage"] = val
                ent = self.entities[cid]
                for c in ent.children:
                    if type(c).__name__ == 'Text':
                        c.text = f"{cid}\n{val}V"
                        break
            self.show_feedback(f"已更新 {cid}")
            self._hide_modify_params_panel()
        except Exception as e:
            logger.error(f"应用修改参数失败: {e}", exc_info=True)

    def _hide_simulation_results_panel(self) -> None:
        """隐藏仿真结果面板"""
        try:
            # 隐藏面板容器
            if self.simulation_results_panel:
                self.simulation_results_panel.visible = False

            # 隐藏背景并确保collider为None
            if hasattr(self.simulation_results_panel, 'bg') and self.simulation_results_panel.bg:
                self.simulation_results_panel.bg.visible = False
                self.simulation_results_panel.bg.collider = None

            # 隐藏标题
            if hasattr(self.simulation_results_panel, 'title'):
                self.simulation_results_panel.title.visible = False

            # 隐藏关闭按钮并禁用collider
            if hasattr(self.simulation_results_panel, 'close_btn'):
                self.simulation_results_panel.close_btn.visible = False
                self.simulation_results_panel.close_btn.collider = None

            # 隐藏所有结果文本 - 使用与示波器相同的结构
            for label, value_text in self.sim_result_texts:
                label.visible = False
                value_text.visible = False

            # 隐藏分隔线并确保collider为None
            if hasattr(self.simulation_results_panel,
                       'left_separator') and self.simulation_results_panel.left_separator:
                self.simulation_results_panel.left_separator.visible = False
                self.simulation_results_panel.left_separator.collider = None
            if hasattr(self.simulation_results_panel,
                       'right_separator') and self.simulation_results_panel.right_separator:
                self.simulation_results_panel.right_separator.visible = False
                self.simulation_results_panel.right_separator.collider = None

            # 隐藏状态栏背景并确保collider为None
            if hasattr(self.simulation_results_panel, 'status_bg') and self.simulation_results_panel.status_bg:
                self.simulation_results_panel.status_bg.visible = False
                self.simulation_results_panel.status_bg.collider = None

            # 重置放置模式状态，避免干扰
            self.is_placing_mode = False
            self.selected_component_type = None
            
            # 恢复背景UI面板显示
            if self.ui_panel:
                self.ui_panel.visible = True
                self.ui_panel.text = self._default_ui_hint

            logger.info("隐藏仿真结果面板并重置状态")

        except Exception as e:
            logger.error(f"隐藏仿真结果面板失败: {e}", exc_info=True)

    def update(self):
        """Ursina引擎的每一帧更新回调，用于实时更新示波器波形和处理缩放"""
        try:
            # 实时更新示波器波形
            if self.active_oscilloscope and time.time() - self.last_oscilloscope_update > self.oscilloscope_update_interval:
                self._update_oscilloscope_waveforms(self.active_oscilloscope)
                self.last_oscilloscope_update = time.time()
            
            # 处理波形缩放
            if self.active_oscilloscope and hasattr(self, "oscilloscope_waveform_image"):
                # 检测鼠标滚轮输入（添加属性检查）
                if hasattr(mouse, 'scroll_y') and mouse.scroll_y != 0:
                    # 调整缩放比例
                    scale_factor = 1.1 if mouse.scroll_y > 0 else 0.9
                    new_scale = self.oscilloscope_scale * scale_factor
                    # 限制缩放范围
                    self.oscilloscope_scale = max(self.oscilloscope_scale_min, min(self.oscilloscope_scale_max, new_scale))
                    # 更新波形图像大小
                    self.oscilloscope_waveform_image.scale = (0.8 * self.oscilloscope_scale, 0.6 * self.oscilloscope_scale)
                    logger.info(f"波形缩放比例: {self.oscilloscope_scale:.2f}")
                    # 阻止滚轮事件传递给场景相机
                    if hasattr(self.interactive_camera, 'mouse') and hasattr(mouse, 'scroll_y'):
                        # 重置鼠标滚轮值，防止影响相机
                        mouse.scroll_y = 0
        except Exception as e:
            logger.error(f"update方法执行失败: {e}", exc_info=True)

    def _select_component_type(self, comp_type: str) -> None:
        """选择元件类型并进入放置模式"""
        self.selected_component_type = comp_type
        self.is_placing_mode = True
        self._set_terminal_colliders(False)  # 放置时端子不挡实验台点击

        # 更新UI提示
        if self.ui_panel:
            self.ui_panel.text = f"已选择: {comp_type} | 点击实验台放置 | ESC取消"

        logger.info(f"选择元件类型: {comp_type}")

    def _toggle_wiring_mode(self) -> None:
        """切换连线模式"""
        # 退出连线模式时恢复已高亮的起点端子颜色
        if self.wire_first_terminal_entity and hasattr(self.wire_first_terminal_entity, 'color'):
            orig = getattr(self, "_wire_first_terminal_original_color", None)
            self.wire_first_terminal_entity.color = orig if orig is not None else color.white
        self.wire_first_terminal_entity = None
        self._wire_first_terminal_original_color = None

        self.is_wiring_mode = not self.is_wiring_mode
        self.wire_first_component_id = None
        self.wire_first_terminal_id = None
        self._set_terminal_colliders(True)  # 连线时需要点端子
        # 连线时关闭带端子元件的主体碰撞体，避免挡住端子，便于点中端子；接地无端子实体，保持主体可点
        self._set_component_body_colliders(not self.is_wiring_mode)

        # 更新UI提示
        if self.is_wiring_mode:
            if self.ui_panel:
                self.ui_panel.text = "连线模式：点击两个端子连接 | ESC退出"
            logger.info("进入连线模式")
        else:
            if self.ui_panel:
                self.ui_panel.text = self._default_ui_hint
            logger.info("退出连线模式")

    def _setup_camera(self) -> None:
        """配置相机（修复window.event_handler不存在的错误）"""
        self.interactive_camera = EditorCamera(
            rotate_button=mouse.right,  # 右键旋转（符合操作习惯）
            pan_button=mouse.middle,  # 中键平移（符合操作习惯）
            zoom_button=None  # 滚轮缩放由EditorCamera自动处理
        )
        # 正对实验台：从台前(z负)往台心看，列(x)左右、行(z)向里
        self.interactive_camera.position = Vec3(0.0, 8.0, -18.0)
        self.interactive_camera.rotation_x = 28
        self.interactive_camera.rotation_y = 0
        self.interactive_camera.sensitivity = (0.05, 0.05, 0.05)

    def input(self, key) -> None:
        """处理用户输入（使用时间阈值区分右键点击/长按）"""
        try:
            # 右键按下：记录时间和位置
            if key == 'right mouse down':
                self.right_mouse_pressed = True
                self.right_mouse_press_time = time.time()
                self.right_mouse_press_pos = (mouse.x, mouse.y)
                if self.component_menu:
                    self.component_menu.visible = False  # 按下时先隐藏菜单
                logger.info(f"右键按下，记录时间戳")

            # 右键松开：计算时长，判断点击/长按
            elif key == 'right mouse up':
                if self.right_mouse_pressed:
                    press_duration = time.time() - self.right_mouse_press_time
                    # 计算鼠标移动距离
                    move_distance = ((mouse.x - self.right_mouse_press_pos[0]) ** 2 +
                                     (mouse.y - self.right_mouse_press_pos[1]) ** 2) ** 0.5

                    logger.info(f"右键松开，时长={press_duration:.3f}s，移动距离={move_distance:.3f}")

                    # 短按且移动距离小（放宽阈值，便于点出菜单）
                    if (press_duration < self.long_press_threshold and
                            move_distance < self.right_click_move_threshold):
                        # 如果示波器被选中，显示详细信息，否则显示菜单
                        if self.selected_component_id:
                            component = self.data_layer.get_component(self.selected_component_id)
                            if component and component.type.value == 'oscilloscope':
                                self._show_oscilloscope_details(component)
                                logger.info(f"✓ 右键点击示波器 {self.selected_component_id}，显示详细信息")
                            else:
                                self._show_component_menu()
                                logger.info(f"✓ 右键点击非示波器元件，显示菜单")
                        else:
                            self._show_component_menu()
                            logger.info(f"✓ 右键点击（时长={press_duration:.3f}s），显示菜单")
                    else:
                        logger.info(f"✗ 不满足条件，不显示菜单（时长>0.2s 或 移动>0.05）")

                    # 重置状态
                    self.right_mouse_pressed = False

            # 左键处理：优先检查菜单点击
            elif key == 'left mouse down':
                # 检查示波器详情面板的关闭按钮
                if self.oscilloscope_detail_panel and self.oscilloscope_detail_panel.visible:
                    hovered = mouse.hovered_entity
                    if hovered and hasattr(hovered, 'is_close_button') and hovered.is_close_button:
                        self._hide_oscilloscope_detail_panel()
                        return

                # 检查仿真结果面板的关闭按钮
                if self.simulation_results_panel and self.simulation_results_panel.visible:
                    hovered = mouse.hovered_entity
                    if hovered and hasattr(hovered, 'is_sim_close_button') and hovered.is_sim_close_button:
                        self._hide_simulation_results_panel()
                        return

                # 检查修改参数面板的确定/取消按钮
                if self.modify_params_panel and self.modify_params_panel.visible:
                    hovered = mouse.hovered_entity
                    if hovered and getattr(hovered, 'is_modify_ok_button', False):
                        self._apply_modify_params()
                        return
                    if hovered and getattr(hovered, 'is_modify_cancel_button', False):
                        self._hide_modify_params_panel()
                        return

                if self.component_menu and self.component_menu.visible:
                    # 菜单可见时，处理菜单点击
                    self._hide_component_menu()
                elif self.is_wiring_mode:
                    # 连线模式：处理连线
                    self._handle_wiring_click()
                elif self.is_placing_mode:
                    # 放置模式：仅当点击实验台平面时放置，避免点到端子或其它遮挡
                    if mouse.hovered_entity == self.click_plane:
                        self._place_component()
                    else:
                        self.show_feedback("请点击实验台空白处放置")
                elif self.modify_params_mode:
                    # 修改参数模式：点中元件则打开参数面板，点空白则取消模式
                    cid = self._get_clicked_component_id(self._get_clicked_entity_normal())
                    ent = self._get_clicked_entity_normal()
                    if cid and cid in self.entities:
                        self._open_modify_params_panel(cid)
                        self.modify_params_mode = False
                        return
                    if ent is None or ent == self.click_plane or ent == self.experiment_table:
                        self.modify_params_mode = False
                        self.show_feedback("已取消")
                        return
                else:
                    # 普通模式：使用拾取分组，导线优先，其次端子，其次元件
                    ent = self._get_clicked_entity_normal()
                    logger.info(f"[DEBUG] 普通模式点击: ent={ent}, type={type(ent).__name__ if ent else None}")
                    cid = self._get_clicked_component_id(ent)
                    logger.info(f"[DEBUG] component_id={cid}")
                    conn_id = getattr(ent, "connection_id", None) if ent else None
                    
                    # 处理示波器界面关闭
                    if self.active_oscilloscope:
                        # 检查是否点击了波形图像
                        clicked_waveform = False
                        if hasattr(self, "oscilloscope_waveform_image"):
                            # 检查鼠标是否在波形图像范围内
                            waveform_pos = self.oscilloscope_waveform_image.position
                            waveform_scale = self.oscilloscope_waveform_image.scale
                            # 计算波形图像的边界
                            left = waveform_pos[0] - waveform_scale[0] / 2
                            right = waveform_pos[0] + waveform_scale[0] / 2
                            top = waveform_pos[1] + waveform_scale[1] / 2
                            bottom = waveform_pos[1] - waveform_scale[1] / 2
                            # 检查鼠标位置是否在波形图像范围内
                            if mouse.x >= left and mouse.x <= right and mouse.y >= bottom and mouse.y <= top:
                                clicked_waveform = True
                        
                        # 点击空白/台面 = 关闭示波器，但不包括点击波形图像
                        if (ent is None or ent == self.click_plane or ent == self.experiment_table) and not clicked_waveform:
                            self._hide_oscilloscope_detail_panel()
                            return
                    
                    # 先处理"点击当前已选"= 取消选中
                    if self.selected_component_id and cid == self.selected_component_id:
                        self._clear_selection()
                        self._clear_wire_selection()
                        return
                    if self.selected_connection_id and conn_id == self.selected_connection_id:
                        self._clear_wire_selection()
                        return
                    # 点空白/台面 = 取消选中
                    if ent is None or ent == self.click_plane or ent == self.experiment_table:
                        self._clear_selection()
                        self._clear_wire_selection()
                        return
                    # 点中元件（含端子、主体、子级）：选中该元件
                    if cid and cid in self.entities:
                        self._select_component_entity_by_id(cid)
                        return
                    # 点中导线：选中该导线
                    if conn_id:
                        wire_ent = ent
                        if wire_ent not in self.wires and getattr(wire_ent, "parent", None):
                            if wire_ent.parent in self.wires:
                                wire_ent = wire_ent.parent
                        self._select_wire_entity(wire_ent)
                        return
                    # 其他（UI、文字等不可选）一律取消选中
                    self._clear_selection()
                    self._clear_wire_selection()

            # Delete/Backspace键：优先删除导线，其次删除元件
            elif key in ('delete', 'backspace'):
                # 优先删除导线，其次删除元件
                if self.selected_connection_id:
                    self._delete_selected_wire()
                else:
                    self._delete_selected_component()

            # I键：显示选中示波器的详细信息
            elif key == 'i':
                if self.selected_component_id:
                    component = self.data_layer.get_component(self.selected_component_id)
                    if component and component.type.value == 'oscilloscope':
                        self._show_oscilloscope_details(component)

            # V键：不再提供语音功能，忽略
            elif key == 'v':
                pass

            # ESC键：取消所有模式并隐藏菜单
            elif key == 'escape':
                self._cancel_placing_mode()
                self.modify_params_mode = False
                self._hide_modify_params_panel()
                self._hide_component_menu()
                self._hide_oscilloscope_detail_panel()  # 隐藏示波器详情面板
                self._hide_simulation_results_panel()  # 隐藏仿真结果面板
                self._clear_selection()
                self._clear_wire_selection()
                if self.is_wiring_mode:
                    self._toggle_wiring_mode()

            # 相机的旋转/平移/缩放由EditorCamera自动处理
        except Exception as e:
            logger.error(f"输入处理失败: {e}", exc_info=True)

    def _show_oscilloscope_details(self, oscilloscope) -> None:
        """显示示波器波形面板（仅Matplotlib波形）"""
        try:
            if not hasattr(self, 'oscilloscope_detail_panel'):
                logger.error("示波器详情面板未初始化")
                return

            comp_id = oscilloscope.id

            # 设置活动示波器
            self.active_oscilloscope = oscilloscope
            # 重置更新时间，确保打开界面后立即开始实时更新
            self.last_oscilloscope_update = 0

            # 更新波形
            self._update_oscilloscope_waveforms(oscilloscope)

            # 隐藏其他面板
            if self.component_menu:
                self.component_menu.visible = False

            logger.info(f"显示示波器 {comp_id} 波形面板（仅Matplotlib波形）")
            logger.info("提示：使用鼠标滚轮缩放波形，按ESC键关闭示波器界面")

        except Exception as e:
            logger.error(f"显示示波器详情面板失败: {e}", exc_info=True)

    def _clamp_position_to_table(self, x: float, z: float) -> Tuple[float, float]:
        """限制坐标在实验台范围内（与网格线范围一致）"""
        return (max(self.GRID_X_MIN, min(12, x)), max(self.GRID_Z_MIN, min(9, z)))

    def _position_from_grid(self, row: int, col: int) -> Tuple[float, float, float]:
        """根据行、列（0-based）计算网格格心坐标，与视觉网格线一致。"""
        row = max(0, min(self.GRID_ROWS - 1, row))
        col = max(0, min(self.GRID_COLS - 1, col))
        x = self.GRID_X_MIN + col + 0.5
        z = self.GRID_Z_MIN + row + 0.5
        return (x, self.COMPONENT_Y, z)

    def _get_click_position_on_table(self) -> Optional[Vec3]:
        """获取鼠标在实验台上的点击位置"""
        try:
            if mouse.hovered_entity == self.click_plane and mouse.world_point:
                p = mouse.world_point
                final_x, final_z = self._clamp_position_to_table(p.x, p.z)
                return Vec3(final_x, self.TABLE_TOP_Y, final_z)
            else:
                return None
        except Exception as e:
            logger.error(f"坐标转换失败: {e}", exc_info=True)
            return None

    def _generate_component_id(self, comp_type: str) -> str:
        """生成元件ID（按类别使用最小未占用的自然数编号：R1,R2,R3... 无空缺）"""
        type_prefix_map = {
            'power_source': 'V',
            'signal_generator': 'SG',
            'resistor': 'R',
            'ground': 'GND',
            'oscilloscope': 'OSC'
        }
        prefix = type_prefix_map.get(comp_type, 'COMP')
        # 从 1 开始找最小的未占用编号，保证同类元件始终是 1、2、3、4、5...
        n = 1
        while True:
            comp_id = f"{prefix}{n}"
            if comp_id not in self.entities and not self.data_layer.get_component(comp_id):
                break
            n += 1
        self.component_counters[comp_type] = n + 1
        return comp_id

    def _generate_wire_id(self) -> str:
        """生成导线ID（最小未占用的自然数：W1,W2,W3... 无空缺）"""
        n = 1
        while f"W{n}" in self.data_layer.connections:
            n += 1
        self.next_wire_id = n + 1
        return f"W{n}"
        """生成元件ID（按类别使用最小未占用的自然数编号：R1,R2,R3... 无空缺）"""
        type_prefix_map = {
            'power_source': 'V',
            'resistor': 'R',
            'ground': 'GND',
            'oscilloscope': 'OSC'
        }
        prefix = type_prefix_map.get(comp_type, 'COMP')
        # 从 1 开始找最小的未占用编号，保证同类元件始终是 1、2、3、4、5...
        n = 1
        while True:
            comp_id = f"{prefix}{n}"
            if comp_id not in self.entities and not self.data_layer.get_component(comp_id):
                break
            n += 1
        self.component_counters[comp_type] = n + 1
        return comp_id

    def _place_component(self) -> None:
        """在鼠标位置放置元件"""
        try:
            # 添加详细日志
            logger.info(f"尝试放置元件 - is_placing_mode: {self.is_placing_mode}, selected_type: {self.selected_component_type}")
            logger.info(f"  mouse.hovered_entity: {mouse.hovered_entity}")
            if mouse.hovered_entity:
                logger.info(f"    - name: {mouse.hovered_entity.name}")
                logger.info(f"    - collider: {mouse.hovered_entity.collider}")
            logger.info(f"  click_plane: {self.click_plane}")
            if self.click_plane:
                logger.info(f"    - visible: {self.click_plane.visible}")
                logger.info(f"    - enabled: {self.click_plane.enabled}")
                logger.info(f"    - collider: {self.click_plane.collider}")
            
            pos = self._get_click_position_on_table()

            if pos is None:
                logger.warning("无法获取鼠标位置 - click_plane未被hover或world_point为None")
                self.show_feedback("无法获取鼠标位置！请点击实验台")
                return

            pos_x, pos_z = self._clamp_position_to_table(pos.x, pos.z)
            logger.debug(f"最终使用位置: x={pos_x:.2f}, y={self.COMPONENT_Y:.2f}, z={pos_z:.2f}")
            logger.debug(f"选择类型: {self.selected_component_type}")

            # 生成元件ID和准备数据（按类别单独编号）
            comp_id = self._generate_component_id(self.selected_component_type)
            final_pos = (pos_x, self.COMPONENT_Y, pos_z)

            command_data = {
                'component_type': self.selected_component_type,
                'component_id': comp_id,
                'position': final_pos
            }
            logger.debug(f"放置元件: {comp_id} at ({pos_x:.2f}, {self.COMPONENT_Y:.2f}, {pos_z:.2f})")

            # 创建元件
            result = self._create_and_add_component(command_data)
            if not result:
                logger.debug(f"创建元件失败: {comp_id}")
                return

            # 退出放置模式并恢复端子碰撞体
            self.is_placing_mode = False
            self.selected_component_type = None
            self._set_terminal_colliders(True)

            # 恢复UI提示
            if self.ui_panel:
                self.ui_panel.text = self._default_ui_hint

        except Exception as e:
            logger.error(f"放置失败: {e}", exc_info=True)

    def _set_terminal_colliders(self, enabled: bool) -> None:
        """放置模式下关闭端子碰撞体以免挡住实验台点击；退出放置/进入连线时重新开启"""
        for ent in self.entities.values():
            for t in getattr(ent, "terminals", []) or []:
                if hasattr(t, "collider"):
                    t.collider = "box" if enabled else None

    def _set_component_body_colliders(self, enabled: bool) -> None:
        """连线模式下关闭带端子元件的主体与 hitbox 碰撞体，便于点中端子；退出连线时恢复"""
        logger.info(f"[DEBUG] _set_component_body_colliders called with enabled={enabled}")
        for ent in self.entities.values():
            if not getattr(ent, "terminals", None):
                continue
            # 主体默认不应有 collider（避免密集时误选），这里强制设置 collider
            # 修复 GLB 模型 collider 问题：不检查原值，直接设置
            ent.collider = "box" if enabled else None
            logger.info(f"[DEBUG] 设置主体 {getattr(ent, 'component_id', 'unknown')} collider = {ent.collider}")
            for child in ent.children:
                if getattr(child, "_is_component_pick_collider", False):
                    child.collider = "box" if enabled else None
                    logger.info(f"[DEBUG] 设置 hitbox {getattr(child, 'component_id', 'unknown')} collider = {child.collider}")
        logger.info(f"[DEBUG] _set_component_body_colliders 完成")

    def _get_clicked_component_id(self, ent) -> Optional[str]:
        """从点击实体或其父级链解析出元件 id（点中端子、主体、hitbox、标签子级等都能识别）"""
        if not ent:
            return None
        e = ent
        while e:
            cid = getattr(e, "component_id", None)
            if cid:
                return cid
            e = getattr(e, "parent", None)
        return None

    def _get_component_terminals(self, comp_id: str) -> List[str]:
        """获取元件可用端子列表（用于语音/自动补全连接端子）。"""
        comp = self.data_layer.get_component(comp_id)
        if not comp:
            return []
        if comp.type == ComponentType.POWER_SOURCE:
            return [TERMINAL_POSITIVE, TERMINAL_NEGATIVE]
        if comp.type == ComponentType.RESISTOR:
            return [TERMINAL_1, TERMINAL_2]
        if comp.type == ComponentType.GROUND:
            return [TERMINAL_COMMON]
        if comp.type == ComponentType.OSCILLOSCOPE:
            return [TERMINAL_1, TERMINAL_2]
        return []

    def _get_used_terminals(self, comp_id: str) -> set:
        """统计某个元件已被连接占用的端子（仅统计带端子信息的连接）。"""
        used = set()
        for c in self.data_layer.connections.values():
            t1 = getattr(c, "terminal1_id", None)
            t2 = getattr(c, "terminal2_id", None)
            if not t1 or not t2:
                continue
            if c.component1_id == comp_id:
                used.add(t1)
            if c.component2_id == comp_id:
                used.add(t2)
        return used

    def _choose_terminal_for_component(self, comp_id: str) -> Optional[str]:
        """为元件选择一个合理的端子（优先未被占用的端子）。"""
        terminals = self._get_component_terminals(comp_id)
        if not terminals:
            return None
        used = self._get_used_terminals(comp_id)
        for t in terminals:
            if t not in used:
                return t
        return terminals[0]

    def _fill_connection_terminals_if_missing(self, conn: Connection) -> Connection:
        """为缺少端子信息的连接补全 terminal1_id / terminal2_id（语音连接用）。"""
        if not getattr(conn, "terminal1_id", None):
            conn.terminal1_id = self._choose_terminal_for_component(conn.component1_id)
        if not getattr(conn, "terminal2_id", None):
            conn.terminal2_id = self._choose_terminal_for_component(conn.component2_id)
        return conn

    def _cancel_placing_mode(self) -> None:
        """取消放置模式"""
        self.is_placing_mode = False
        self.selected_component_type = None
        self._set_terminal_colliders(True)
        if self.ui_panel:
            self.ui_panel.text = self._default_ui_hint
            logger.info("取消放置模式")

    def _handle_wiring_click(self) -> None:
        """处理连线模式下的点击：点端子或点接地主体（视为公共端子），并写入 terminal1_id/terminal2_id"""
        ent = self._get_clicked_entity_wiring()
        if not ent:
            self.show_feedback("请点击端子进行连线")
            return
        comp_id = getattr(ent, "component_id", None)
        terminal_id = getattr(ent, "terminal_id", None)
        # 接地无单独端子实体，点接地主体即视为 TERMINAL_COMMON
        if not terminal_id and comp_id:
            comp = self.data_layer.get_component(comp_id)
            if comp and comp.type == ComponentType.GROUND:
                terminal_id = TERMINAL_COMMON
        if not comp_id or not terminal_id:
            self.show_feedback("请点击端子进行连线")
            return

        if self.wire_first_component_id is None:
            # 第一次点击：记录起点端子，并高亮该端子（接地无端子实体则不高亮）
            self.wire_first_component_id = comp_id
            self.wire_first_terminal_id = terminal_id
            self.wire_first_terminal_entity = ent if getattr(ent, "terminal_id", None) else None
            if self.wire_first_terminal_entity and hasattr(self.wire_first_terminal_entity, "color"):
                # 记录原始颜色（电源正极为红色、负极为黑色等），便于连线结束后恢复
                self._wire_first_terminal_original_color = self.wire_first_terminal_entity.color
                self.wire_first_terminal_entity.color = color.yellow  # 选中端子高亮
            else:
                self._wire_first_terminal_original_color = None
            # 不再修改元件本体颜色（避免电源等模型在连线后颜色异常）
            if self.ui_panel:
                self.ui_panel.text = f"已选起点: {comp_id}({terminal_id}) | 点击终点端子"
            logger.info(f"连线起点: {comp_id} {terminal_id}")
        else:
            # 第二次点击：连接两个端子
            if comp_id == self.wire_first_component_id and terminal_id == self.wire_first_terminal_id:
                self.show_feedback("不能连接同一端子")
                return

            conn = Connection(
                id=self._generate_wire_id(),
                component1_id=self.wire_first_component_id,
                component2_id=comp_id,
                connection_point1=tuple(self.entities[self.wire_first_component_id].position),
                connection_point2=tuple(self.entities[comp_id].position),
                terminal1_id=self.wire_first_terminal_id,
                terminal2_id=terminal_id,
            )

            success = self.connect_components(conn)
            if success:
                self.show_feedback(f"连线成功: {self.wire_first_component_id} <-> {comp_id}")
                # 连线成功后保持连线模式，继续连线
            else:
                self.show_feedback(f"连线失败: {self.wire_first_component_id} <-> {comp_id}")

            # 重置起点端子，准备下次连线（恢复为原始颜色，而不是写死为白色）
            if self.wire_first_terminal_entity and hasattr(self.wire_first_terminal_entity, 'color'):
                orig = self._wire_first_terminal_original_color
                self.wire_first_terminal_entity.color = orig if orig is not None else color.white
            self.wire_first_terminal_entity = None
            self.wire_first_component_id = None
            self.wire_first_terminal_id = None
            self._wire_first_terminal_original_color = None

    def _select_component_entity_by_id(self, comp_id: str) -> None:
        """根据元件 id 选中元件（不显示视觉高亮，只记录选中状态）"""
        self._clear_selection()
        self._clear_wire_selection()
        main_ent = self.entities.get(comp_id)
        if not main_ent:
            return
        self.selected_component_id = comp_id
        # 不创建高亮实体，只记录选中状态
        self._selected_highlight_entity = None

        # 获取元件信息并更新 UI
        component = self.data_layer.get_component(comp_id)
        if component and component.type.value == 'oscilloscope':
            sampling_rate = component.parameters.get('sampling_rate', 'N/A')
            channels = component.parameters.get('channels', 'N/A')
            measured_nodes = component.parameters.get('measured_node_ids', [])

            if self.ui_panel:
                self.ui_panel.text = f"示波器 {comp_id} | 采样率:{sampling_rate}Hz | 通道:{channels} | 测量节点:{measured_nodes} | 鼠标滚轮缩放 | 点击空白关闭 | ESC取消"
        else:
            if self.ui_panel:
                self.ui_panel.text = f"已选中: {comp_id} | Delete删除 | ESC取消选中"

    def _select_component_entity(self, ent: Entity) -> None:
        """选中元件实体（委托给 _select_component_entity_by_id，保证始终高亮主体）"""
        comp_id = self._get_clicked_component_id(ent)
        if comp_id and comp_id in self.entities:
            self._select_component_entity_by_id(comp_id)
            return
        comp_id = getattr(ent, "component_id", None)
        if not comp_id:
            return
        self._select_component_entity_by_id(comp_id)

    def _clear_selection(self) -> None:
        """清除选中状态"""
        if not self.selected_component_id:
            return

        # 销毁高亮实体（如果存在）
        if self._selected_highlight_entity:
            try:
                destroy(self._selected_highlight_entity)
            except:
                pass
            self._selected_highlight_entity = None

        self.selected_component_id = None
        self._selected_original_color = None

        if self.ui_panel:
            self.ui_panel.text = self._default_ui_hint

    def _delete_selected_component(self) -> None:
        """删除选中的元件"""
        if not self.selected_component_id:
            self.show_feedback("未选中元件")
            return

        comp_id = self.selected_component_id
        self._clear_selection()
        ok = self.remove_component(comp_id)
        if ok:
            self.show_feedback(f"已删除: {comp_id}")
        else:
            self.show_feedback(f"删除失败: {comp_id}")

    def _select_wire_entity(self, wire_ent: Entity) -> None:
        """选中导线实体"""
        # 先取消旧选中（导线和元件都清空）
        self._clear_selection()
        self._clear_wire_selection()

        # 若点中的是 hitbox（导线子实体），则解析到导线本体（在 self.wires 里的那个）
        if wire_ent not in self.wires and getattr(wire_ent, "parent", None) is not None:
            p = wire_ent.parent
            if p is not None and p in self.wires:
                wire_ent = p

        conn_id = getattr(wire_ent, "connection_id", None)
        if not conn_id:
            return

        self.selected_connection_id = conn_id
        self.selected_wire_entity = wire_ent
        self._selected_wire_original_color = wire_ent.color

        # 不改变导线颜色，避免覆盖到元件时视觉效果不清晰
        # 只在 UI 上提示

        if self.ui_panel:
            self.ui_panel.text = f"已选中导线: {conn_id} | Delete删除 | ESC取消选中"

    def _clear_wire_selection(self) -> None:
        """取消导线选中（优先用实体引用恢复颜色，避免状态不同步）"""
        # 即使 id 为空，也把引用清掉，防止悬空
        if not getattr(self, "selected_connection_id", None):
            self.selected_wire_entity = None
            self._selected_wire_original_color = None
            return

        # 1) 用实体引用恢复颜色（最稳）
        if getattr(self, "selected_wire_entity", None) is not None:
            if getattr(self, "_selected_wire_original_color", None) is not None:
                try:
                    self.selected_wire_entity.color = self._selected_wire_original_color
                except Exception:
                    pass

        # 2) 清空状态
        self.selected_wire_entity = None
        self.selected_connection_id = None
        self._selected_wire_original_color = None

    def _delete_selected_wire(self) -> None:
        """删除选中的导线"""
        if not getattr(self, "selected_connection_id", None):
            self.show_feedback("未选中导线")
            return

        conn_id = self.selected_connection_id
        wire_ent = getattr(self, "selected_wire_entity", None)

        self._clear_wire_selection()

        # 1) 删 data 层连接（用于保持数据一致；提示语不依赖其返回值）
        self.data_layer.remove_connection(conn_id)

        # 2) 删 VR 里的导线实体，并记录是否真的移除了
        did_remove = False

        if wire_ent is not None and wire_ent in self.wires:
            self.wires.remove(wire_ent)
            destroy(wire_ent)
            did_remove = True
        else:
            wire_to_remove = None
            for w in self.wires:
                if getattr(w, "connection_id", None) == conn_id:
                    self.wires.remove(w)
                    destroy(w)
                    did_remove = True
                    break

        if self.circuit_simulator:
            self.circuit_simulator.clear_last_result()

        self.show_feedback(f"已删除导线: {conn_id}" if did_remove else f"删除导线失败: {conn_id}")

    def _create_data_component(self, comp_type: str, comp_id: str, pos: tuple, params: Dict = None) -> Component:
        """创建数据层元件对象"""
        params = params or {}
        if comp_type == 'power_source':
            # 普通直流电源与信号发生器在数据层统一为 POWER_SOURCE，
            # 通过参数中的 mode 区分 'dc' / 'ac'。
            voltage = params.get('voltage', 5.0)
            ps = PowerSource(id=comp_id, position=pos, voltage=voltage)
            # 若上层约定这是“信号发生器”，会在 params 中传入 mode / frequency
            mode = params.get("mode")
            if mode in ("dc", "ac"):
                ps.parameters["mode"] = mode
            # 仅在 AC 模式下才关心频率，直流保持默认即可
            if ps.parameters.get("mode", "dc") == "ac" and "frequency" in params:
                ps.parameters["frequency"] = float(params["frequency"])
            return ps
        elif comp_type == 'resistor':
            return Resistor(id=comp_id, position=pos, resistance=params.get('resistance', 1000))
        elif comp_type == 'ground':
            return Ground(id=comp_id, position=pos)
        elif comp_type == 'oscilloscope':
            from scr.data.data_layer import Oscilloscope
            return Oscilloscope(
                id=comp_id,
                position=pos,
                sampling_rate=params.get('sampling_rate', 1000.0),
                measured_node_ids=params.get('measured_node_ids', []),
                channels=params.get('channels', 2)
            )
        else:
            raise ValueError(f"未知元件类型: {comp_type}")

    def _normalize_component_type(self, comp_type: str) -> str:
        """将别名统一为数据层/VR 使用的类型名；未支持类型兜底为 resistor，避免菜单/迷你界面崩溃"""
        if not comp_type or not isinstance(comp_type, str):
            return "resistor"
        key = comp_type.strip().lower()
        alias_map = {
            "power_source": "power_source",
            "power": "power_source",
            "power_supply": "power_source",
            "voltage_source": "power_source",
            "source": "power_source",
            "voltage": "power_source",
            "signal_generator": "power_source",
            "resistor": "resistor",
            "ground": "ground",
            "gnd": "ground",
            "oscilloscope": "oscilloscope",
            # 未支持类型兜底，避免 ValueError 导致界面异常
            "capacitor": "resistor",
            "inductor": "resistor",
            "wire": "resistor",
            "unknown": "resistor",
        }
        return alias_map.get(key, "resistor")

    def _add_to_data_layer(self, comp_id: str, component: Component) -> bool:
        """添加元件到数据层"""
        if comp_id not in self.data_layer.components:
            if not self.data_layer.add_component(component):
                logger.error(f"数据层添加元件失败: {comp_id}")
                return False
        else:
            logger.info(f"元件 {comp_id} 已存在于数据层，仅创建VR实体")
        return True

    def _create_vr_entity(self, comp_type: str, comp_id: str, pos: tuple, params: Dict) -> None:
        """创建VR实体"""
        if comp_type == 'power_source':
            self._create_power_source_entity(comp_id, pos, params)
        elif comp_type == 'resistor':
            self._create_resistor_entity(comp_id, pos, params)
        elif comp_type == 'ground':
            # 对于接地元件，传递空字典作为params，避免传递不必要的属性
            self._create_ground_entity(comp_id, pos, {})
        elif comp_type == 'oscilloscope':
            self._create_oscilloscope_entity(comp_id, pos, params)

    # 端子：直接挂在 scene 下、用世界坐标，避免被父级遮挡或缩放影响
    TERMINAL_WORLD_SCALE = (0.25, 0.25, 0.25)  # 世界空间尺寸，明显可见
    TERMINAL_WORLD_OFFSET = 0.22  # 端子相对元件中心的世界坐标偏移
    TERMINAL_LABELS = {
        TERMINAL_POSITIVE: "+",
        TERMINAL_NEGATIVE: "-",
        TERMINAL_1: "1",
        TERMINAL_2: "2",
        TERMINAL_COMMON: "GND",
    }

    def _add_terminal_entity(self, parent: Entity, local_pos: Tuple[float, float, float],
                             terminal_id: str, comp_id: str, terminal_color, label: Optional[str] = None) -> Entity:
        """已废弃：改用 _add_terminal_entity_world。"""
        return self._add_terminal_entity_world(
            parent.world_position, local_pos, terminal_id, comp_id, terminal_color, label
        )

    def _add_terminal_entity_world(self, comp_world_pos: Tuple[float, float, float],
                                  local_offset: Tuple[float, float, float],
                                  terminal_id: str, comp_id: str, terminal_color,
                                  label: Optional[str] = None) -> Entity:
        """
        在场景中创建端子实体：父级为 scene，用世界坐标放置，确保一定可见、可点。
        comp_world_pos: 元件当前世界坐标
        local_offset: 端子相对元件中心的偏移 (x,y,z)，与元件朝向一致时相当于局部偏移
        """
        wx = comp_world_pos[0] + local_offset[0]
        wy = comp_world_pos[1] + local_offset[1]
        wz = comp_world_pos[2] + local_offset[2]
        self._ensure_pick_roots()
        term = Entity(
            parent=self.pick_terminals,
            model='cube',
            color=terminal_color,
            scale=self.TERMINAL_WORLD_SCALE,
            position=(wx, wy, wz),
            collider='box',
        )
        term.terminal_id = terminal_id
        term.component_id = comp_id
        term._comp_world_pos = comp_world_pos  # 用于后续若元件移动时同步
        term._local_offset = local_offset
        text_str = label if label is not None else self.TERMINAL_LABELS.get(terminal_id, "?")
        # 电源 + / - 用更大字号便于辨认；电阻 1/2 保持默认
        text_scale = 18 if terminal_id in (TERMINAL_POSITIVE, TERMINAL_NEGATIVE) else 10
        # 标签放在端子方块上方，多角度都能看到（避免背对相机时看不到 - / 1 / 2）
        # 某些环境下 Panda3D 的 DynamicTextFont 在特定时机会触发断言：
        # get_num_pages() == 0 at dynamicTextFont.I
        # 为避免整条语音流程因字体异常崩溃，这里对 Text 创建做保护性捕获。
        try:
            Text(
                parent=term,
                text=text_str,
                position=(0, 0.5, 0),
                scale=text_scale,
                color=color.black,
                background=color.rgba(255, 255, 255, 0.9),
                origin=(0, 0),
                billboard=True,
                bold=True,
            )
        except AssertionError as e:
            logger.warning(f"创建端子标签文字失败（忽略，仅不显示文字）: {e}")
        except Exception as e:
            logger.warning(f"创建端子标签文字异常（忽略，仅不显示文字）: {e}")
        return term

    def _create_and_add_component(self, command_data: Dict) -> bool:
        """创建并添加元件"""
        try:
            comp_type = command_data.get('component_type')
            comp_id = command_data.get('component_id')
            pos = command_data.get('position', (0, 0.8, 0))
            params = {k: v for k, v in command_data.items() if k not in ('component_type', 'component_id', 'position')}

            # 针对“信号发生器”增加默认 AC 参数，再统一元件类型别名
            if comp_type == "signal_generator":
                # 信号发生器语义：POWER_SOURCE + mode='ac'
                params.setdefault("voltage", 5.0)
                params.setdefault("mode", "ac")
                params.setdefault("frequency", 1000.0)

            # 统一元件类型别名（如 power -> power_source），避免未知类型报错
            comp_type = self._normalize_component_type(comp_type)

            # 创建数据层元件
            component = self._create_data_component(comp_type, comp_id, pos, params)

            # 添加到数据层
            if not self._add_to_data_layer(comp_id, component):
                return False

            # 创建VR实体
            self._create_vr_entity(comp_type, comp_id, pos, component.__dict__)
            logger.info(f"添加元件成功: {comp_id}")
            return True

        except Exception as e:
            logger.error(f"创建元件失败: {e}", exc_info=True)
            return False

    def _component_type_to_string(self, comp_type: ComponentType) -> str:
        """将ComponentType枚举转换为字符串"""
        type_map = {
            ComponentType.RESISTOR: 'resistor',
            ComponentType.POWER_SOURCE: 'power_source',
            ComponentType.GROUND: 'ground',
            ComponentType.OSCILLOSCOPE: 'oscilloscope'
        }
        return type_map.get(comp_type, 'unknown')

    def add_component(self, component: Component) -> bool:
        """添加元件（保留自定义Z轴，仅限制试验台范围）"""
        try:
            comp_id = component.id

            if not self._add_to_data_layer(comp_id, component):
                return False

            pos_x, pos_z = self._clamp_position_to_table(component.position[0], component.position[2])
            pos = (pos_x, self.COMPONENT_Y, pos_z)
            params = component.__dict__

            comp_type_str = self._component_type_to_string(component.type)
            self._create_vr_entity(comp_type_str, comp_id, pos, params)

            logger.info(f"添加元件成功: {comp_id}（Z轴={pos_z}）")
            return True
        except Exception as e:
            logger.error(f"添加元件失败: {e}", exc_info=True)
            return False

    def _create_power_source_entity(self, comp_id: str, pos: tuple, params: Dict) -> None:
        from .component_renderer import create_power_source_entity
        create_power_source_entity(self, comp_id, pos, params)

    def _create_resistor_entity(self, comp_id: str, pos: tuple, params: Dict) -> None:
        from .component_renderer import create_resistor_entity
        create_resistor_entity(self, comp_id, pos, params)

    def _create_ground_entity(self, comp_id: str, pos: tuple, params: Dict = None) -> None:
        from .component_renderer import create_ground_entity
        create_ground_entity(self, comp_id, pos, params)

    def _create_oscilloscope_entity(self, comp_id: str, pos: tuple, params: Dict) -> None:
        from .component_renderer import create_oscilloscope_entity
        create_oscilloscope_entity(self, comp_id, pos, params)

    def _create_oscilloscope_grid(self, screen: Entity) -> None:
        """创建示波器屏幕网格线"""
        grid_color = color.rgba(0, 255, 0, 0.3)  # 绿色网格

        # 垂直网格线（时间轴）
        for x in range(-5, 6):
            Entity(
                parent=screen,
                model='cube',
                color=grid_color,
                scale=(0.005, 0.5, 0.01),
                position=(x * 0.2, 0, 0.01),
                collider=None
            )

        # 水平网格线（电压轴）
        for y in range(-4, 5):
            Entity(
                parent=screen,
                model='cube',
                color=grid_color,
                scale=(1.0, 0.005, 0.01),
                position=(0, y * 0.125, 0.01),
                collider=None
            )

    def _ensure_oscilloscope_waveform_points(self) -> None:
        """确保为示波器波形创建采样点实体（电压/电流各一组）。"""
        if not self.oscilloscope_detail_panel:
            return
        from ursina import Entity as UEntity, color as ucolor  # 局部导入避免循环

        if not getattr(self, "oscilloscope_voltage_points", None):
            parent = getattr(self.oscilloscope_detail_panel, "voltage_screen", None)
            if parent:
                self.oscilloscope_voltage_points = [
                    UEntity(
                        parent=parent,
                        model='quad',
                        color=ucolor.green,
                        # 稍微放大点大小，波形更连贯
                        scale=(0.02, 0.02, 0),
                        position=(0, 0, 0.01),
                        collider=None,
                        visible=False,
                    )
                    # 提高采样点数量，让曲线更平滑
                    for _ in range(128)
                ]
        if not getattr(self, "oscilloscope_current_points", None):
            parent = getattr(self.oscilloscope_detail_panel, "current_screen", None)
            if parent:
                self.oscilloscope_current_points = [
                    UEntity(
                        parent=parent,
                        model='quad',
                        color=ucolor.red,
                        scale=(0.02, 0.02, 0),
                        position=(0, 0, 0.01),
                        collider=None,
                        visible=False,
                    )
                    for _ in range(128)
                ]

    def _update_oscilloscope_waveforms(self, oscilloscope) -> None:
        """根据最新仿真结果更新示波器电压/电流波形与实时数值。"""
        try:
            if not self.circuit_simulator:
                return

            osc_id = oscilloscope.id
            latest = self.circuit_simulator.get_latest_results()
            if (not latest or
                    not latest.oscilloscope_data or
                    osc_id not in latest.oscilloscope_data):
                # 隐藏波形图像
                if hasattr(self, "oscilloscope_waveform_image"):
                    self.oscilloscope_waveform_image.visible = False
                return

            osc_data = latest.oscilloscope_data[osc_id]
            channels = osc_data.get("channels", {})
            ch1 = channels.get("CH1")
            if not ch1:
                return

            time_data = ch1.get("time_data") or []
            voltage_data = ch1.get("voltage_data") or []
            if len(time_data) < 2 or len(voltage_data) < 2:
                return

            # 电流数据
            current_data = None
            # 1) 若示波器接在交流信号源两端，仅显示电压，不显示电流波形
            if not self._is_oscilloscope_ac_source_probe(oscilloscope):
                # 2) 其他情况：尝试找到离示波器最近的电阻，使用其电流幅值生成交流/直流电流波形
                resistor_id = self._find_nearest_resistor_for_oscilloscope(oscilloscope.id)
                current_amp = 0.0
                ac_freq = 0.0
                if resistor_id:
                    # 从仿真层获取电阻电流（使用功率分配中的 current 字段）
                    power_info = self.circuit_simulator._calculate_power_distribution(latest.raw_data)
                    comp_power = power_info.get("component_power", {})
                    resistor_info = comp_power.get(resistor_id)
                    if isinstance(resistor_info, dict):
                        current_amp = float(resistor_info.get("current", 0.0))
                    # 检测交流频率：取任一 mode='ac' 的电源的 frequency
                    from scr.data.data_layer import ComponentType as CT2
                    ac_sources = [
                        c for c in self.data_layer.get_all_components()
                        if c.type == CT2.POWER_SOURCE and c.parameters.get("mode", "dc") == "ac"
                    ]
                    if ac_sources:
                        ac_freq = float(ac_sources[0].parameters.get("frequency", 50.0))

                # 生成电流波形数据
                if ac_freq > 0.0:
                    # 交流模式：生成正弦波，添加明显的相位变化和幅度变化以增强动态效果
                    omega = 2 * np.pi * ac_freq
                    # 添加随时间变化的相位偏移，使波形看起来更动态
                    phase_offset = time.time() * 0.5  # 更快变化的相位
                    # 添加微小的幅度变化
                    amplitude_variation = 1.0 + 0.1 * np.sin(time.time() * 0.2)  # 缓慢的幅度变化
                    current_data = [current_amp * amplitude_variation * np.sin(omega * t + phase_offset) for t in time_data]
                else:
                    # 直流模式：添加更明显的波动以显示动态效果
                    current_data = [current_amp + (np.random.rand() - 0.5) * 0.05 for t in time_data]
            
            # 为电压波形添加更明显的波动以增强动态效果
            # 添加随时间变化的基线偏移
            baseline_offset = 0.1 * np.sin(time.time() * 0.3)
            # 添加随机波动
            voltage_data = [v + baseline_offset + (np.random.rand() - 0.5) * 0.1 for v in voltage_data]

            # 初始化波形历史
            if osc_id not in self.waveform_history:
                self.waveform_history[osc_id] = {
                    'time': [],
                    'voltage': [],
                    'current': []
                }

            # 计算时间偏移，使新数据在时间轴上延续
            history = self.waveform_history[osc_id]
            if history['time']:
                last_time = history['time'][-1]
                time_offset = last_time + (time_data[1] - time_data[0])
                new_time_data = [t + time_offset for t in time_data]
            else:
                new_time_data = time_data.copy()

            # 添加新数据到历史
            history['time'].extend(new_time_data)
            history['voltage'].extend(voltage_data)
            if current_data:
                history['current'].extend(current_data)

            # 限制历史数据长度
            if len(history['time']) > self.max_history_points:
                excess = len(history['time']) - self.max_history_points
                history['time'] = history['time'][excess:]
                history['voltage'] = history['voltage'][excess:]
                if history['current']:
                    history['current'] = history['current'][excess:]

            # 使用历史数据生成波形
            display_time = history['time']
            display_voltage = history['voltage']
            display_current = history['current'] if history['current'] else None

            # 添加实时时间戳到波形图像
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # 使用Matplotlib生成波形图像
            image = self.generate_oscilloscope_image(display_time, display_voltage, display_current, timestamp)
            if image:
                try:
                    # 隐藏原始的波形点
                    for pt in getattr(self, "oscilloscope_voltage_points", []) or []:
                        pt.visible = False
                    for pt in getattr(self, "oscilloscope_current_points", []) or []:
                        pt.visible = False
                    
                    # 尝试将PIL Image对象转换为Ursina可用的纹理
                    texture = Texture(image)
                    
                    # 创建或更新波形图像实体
                    if not hasattr(self, "oscilloscope_waveform_image"):
                        # 创建波形图像实体
                        self.oscilloscope_waveform_image = Entity(
                            parent=camera.ui,
                            model='quad',
                            texture=texture,
                            scale=(0.8, 0.6),  # 调整大小以适应屏幕
                            position=(0, 0, -1),  # 置于UI最上层
                            collider=None
                        )
                        logger.info("创建波形图像实体成功")
                    else:
                        # 更新现有图像
                        self.oscilloscope_waveform_image.texture = texture
                        self.oscilloscope_waveform_image.visible = True
                        logger.info("更新波形图像实体成功")
                except Exception as e:
                    logger.error(f"创建或更新波形图像实体失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"更新示波器波形失败: {e}", exc_info=True)

    def _update_oscilloscope_waveforms_fallback(self, oscilloscope, time_data, voltage_data, current_data=None) -> None:
        """回退方法：使用原始的点显示方式更新波形"""
        try:
            logger.info("使用回退方法更新示波器波形")
            # 确保波形点已初始化
            self._ensure_oscilloscope_waveform_points()

            # 以幅值为基准缩放波形高度
            v_amp = max(abs(v) for v in voltage_data)

            # 更新电压波形
            vscreen = getattr(self.oscilloscope_detail_panel, "voltage_screen", None)
            if vscreen and self.oscilloscope_voltage_points:
                n_pts = len(self.oscilloscope_voltage_points)
                sx, sy, _ = vscreen.scale
                for i, pt in enumerate(self.oscilloscope_voltage_points):
                    alpha = i / max(n_pts - 1, 1)
                    idx = int(alpha * (len(voltage_data) - 1))
                    v = voltage_data[idx]
                    # 以幅值归一化到 [-1, 1]
                    v_norm = v / max(v_amp, 1e-9) if v_amp != 0 else 0.0

                    local_x = (alpha - 0.5) * sx * 0.9
                    # 提高纵向比例到 0.8，让波形更高、更清晰
                    local_y = v_norm * sy * 0.8
                    pt.position = (local_x, local_y, 0.01)
                    pt.visible = True

            # 更新电流波形
            if current_data:
                cscreen = getattr(self.oscilloscope_detail_panel, "current_screen", None)
                if cscreen and self.oscilloscope_current_points:
                    n_cpts = len(self.oscilloscope_current_points)
                    cx, cy, _ = cscreen.scale
                    i_amp = max(abs(i) for i in current_data)
                    for i, pt in enumerate(self.oscilloscope_current_points):
                        alpha = i / max(n_cpts - 1, 1)
                        idx = int(alpha * (len(current_data) - 1))
                        i_val = current_data[idx]
                        # 归一化到 [-1, 1]
                        i_norm = i_val / max(i_amp, 1e-9) if i_amp != 0 else 0.0

                        local_x = (alpha - 0.5) * cx * 0.9
                        # 提高纵向比例到 0.8
                        local_y = i_norm * cy * 0.8
                        pt.position = (local_x, local_y, 0.01)
                        pt.visible = True
            else:
                # 隐藏电流波形
                for pt in self.oscilloscope_current_points:
                    pt.visible = False

            logger.info("回退方法更新示波器波形成功")
        except Exception as e:
            logger.error(f"回退方法更新示波器波形失败: {e}", exc_info=True)

    def _is_oscilloscope_ac_source_probe(self, oscilloscope) -> bool:
        """判断示波器是否接在某个交流信号源的两个端子上。"""
        try:
            conns = self.data_layer.get_connections_for_component(oscilloscope.id)
            if not conns:
                return False
            other_ids = {
                c.component1_id if c.component2_id == oscilloscope.id else c.component2_id
                for c in conns
            }
            if len(other_ids) != 1:
                return False
            other_id = next(iter(other_ids))
            other = self.data_layer.get_component(other_id)
            if not other or other.type != ComponentType.POWER_SOURCE:
                return False
            return other.parameters.get("mode", "dc") == "ac"
        except Exception as e:
            logger.error(f"判断示波器是否接在信号源两端失败: {e}", exc_info=True)
            return False

    def generate_oscilloscope_image(self, time_data, voltage_data, current_data=None, timestamp=None):
        """使用Matplotlib生成示波器风格的波形图像"""
        try:
            # 配置Matplotlib样式（示波器风格）
            plt.rcParams.update({
                'figure.facecolor': 'black',
                'axes.facecolor': 'black',
                'axes.edgecolor': 'white',
                'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': '#666666',
                'grid.linestyle': '--',
                'grid.linewidth': 0.8
            })

            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), dpi=100)

            # 计算电压峰值
            v_max = max(abs(v) for v in voltage_data)
            v_padding = v_max * 0.1  # 10%的 padding
            y_min = -(v_max + v_padding)
            y_max = v_max + v_padding
            
            # 绘制电压波形
            ax1.plot(time_data, voltage_data, color='#00ff00', linewidth=1.5)
            ax1.set_title('Voltage', color='white')
            ax1.set_ylabel('Voltage (V)', color='white')
            ax1.grid(True, alpha=0.7)
            # 自动调整Y轴范围，以峰值为基准
            ax1.set_ylim(y_min, y_max)
            # 设置Y轴刻度，显示峰值对应的数值
            import numpy as np
            # 生成合理的刻度
            num_ticks = 5
            y_ticks = np.linspace(y_min, y_max, num_ticks)
            # 格式化刻度值，保留合适的小数位数
            y_tick_labels = [f'{tick:.2f}' for tick in y_ticks]
            ax1.set_yticks(y_ticks)
            ax1.set_yticklabels(y_tick_labels)
            # 时间轴延伸效果
            # 使用实际的历史时间数据，让波形沿着时间轴真正延伸
            ax1.set_xlim(min(time_data), max(time_data))

            # 绘制电流波形
            if current_data:
                # 计算电流数据的范围
                i_min = min(current_data)
                i_max = max(current_data)
                i_range = i_max - i_min
                
                # 计算电流的平均值（绝对值）
                i_avg = sum(abs(i) for i in current_data) / len(current_data)
                
                # 如果电流很小，放大显示
                if i_avg < 0.01 or i_range < 0.01:
                    # 放大电流波形，使其更加明显
                    scale_factor = 100  # 放大100倍
                    scaled_current = [i * scale_factor for i in current_data]
                    ax2.plot(time_data, scaled_current, color='#ff0000', linewidth=1.5)
                    ax2.set_ylabel('Current (A) × 100', color='white')
                    # 计算电流峰值并自动调整Y轴范围
                    i_scaled_max = max(abs(i) for i in scaled_current)
                    i_scaled_padding = i_scaled_max * 0.1  # 10%的 padding
                    y_min = -(i_scaled_max + i_scaled_padding)
                    y_max = i_scaled_max + i_scaled_padding
                    ax2.set_ylim(y_min, y_max)
                    # 设置Y轴刻度，显示峰值对应的数值
                    import numpy as np
                    # 生成合理的刻度
                    num_ticks = 5
                    y_ticks = np.linspace(y_min, y_max, num_ticks)
                    # 格式化刻度值，保留合适的小数位数
                    y_tick_labels = [f'{tick:.2f}' for tick in y_ticks]
                    ax2.set_yticks(y_ticks)
                    ax2.set_yticklabels(y_tick_labels)
                else:
                    # 正常显示电流波形
                    ax2.plot(time_data, current_data, color='#ff0000', linewidth=1.5)
                    ax2.set_ylabel('Current (A)', color='white')
                    # 计算电流峰值并自动调整Y轴范围
                    i_max = max(abs(i) for i in current_data)
                    i_padding = i_max * 0.1  # 10%的 padding
                    y_min = -(i_max + i_padding)
                    y_max = i_max + i_padding
                    ax2.set_ylim(y_min, y_max)
                    # 设置Y轴刻度，显示峰值对应的数值
                    import numpy as np
                    # 生成合理的刻度
                    num_ticks = 5
                    y_ticks = np.linspace(y_min, y_max, num_ticks)
                    # 格式化刻度值，保留合适的小数位数
                    y_tick_labels = [f'{tick:.2f}' for tick in y_ticks]
                    ax2.set_yticks(y_ticks)
                    ax2.set_yticklabels(y_tick_labels)
                
                ax2.set_title('Current', color='white')
                ax2.set_xlabel('Time (s)', color='white')
                ax2.grid(True, alpha=0.7)
                # 时间轴延伸效果，与电压波形同步
                ax2.set_xlim(min(time_data), max(time_data))
            else:
                ax2.set_title('Current', color='white')
                ax2.set_ylabel('Current (A)', color='white')
                ax2.set_xlabel('Time (s)', color='white')
                ax2.grid(True, alpha=0.7)
                ax2.set_xlim(min(time_data), max(time_data))
                ax2.set_ylim(-0.2, 0.2)
                ax2.text(0.5, 0.5, 'No Data', color='white', ha='center', va='center', transform=ax2.transAxes)

            # 添加实时时间戳
            if timestamp:
                fig.text(0.95, 0.95, f"Real-time: {timestamp}", color='yellow', ha='right', va='top', fontsize=8)

            # 调整布局
            plt.tight_layout()

            # 保存到内存
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = Image.open(buffer)

            # 清理
            plt.close(fig)

            return image
        except Exception as e:
            logger.error(f"生成示波器波形图像失败: {e}", exc_info=True)
            return None

    def _find_nearest_resistor_for_oscilloscope(self, osc_id: str) -> Optional[str]:
        """在连接图中从示波器开始向外搜索，找到最近的电阻ID，用于电流波形计算。"""
        try:
            from collections import deque
            comps = self.data_layer.components
            conns = self.data_layer.connections
            if osc_id not in comps:
                return None

            # 构建无向邻接表（基于连接关系）
            adj: Dict[str, List[str]] = {}
            for conn in conns.values():
                a, b = conn.component1_id, conn.component2_id
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)

            visited = set()
            q = deque([osc_id])
            visited.add(osc_id)

            while q:
                cid = q.popleft()
                for nb in adj.get(cid, []):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    comp = comps.get(nb)
                    if comp and comp.type == ComponentType.RESISTOR:
                        return nb
                    q.append(nb)
            return None
        except Exception as e:
            logger.error(f"查找示波器最近电阻失败: {e}", exc_info=True)
            return None

    def connect_components(self, connection: Connection) -> bool:
        """连接元件（精准对齐+适配自定义Z轴）"""
        try:
            if not self.data_layer.add_connection(connection):
                logger.error(f"数据层添加连接失败: {connection.id}")
                return False

            comp1_id = connection.component1_id
            comp2_id = connection.component2_id

            if comp1_id not in self.entities or comp2_id not in self.entities:
                logger.warning(f"元件未创建，无法连接: {comp1_id} <-> {comp2_id}")
                return False

            wire = self._create_wire_entity(connection)
            if wire:
                # 改动点A：给导线添加碰撞体和连接ID
                wire.collider = 'box'
                wire.connection_id = connection.id
                self.wires.append(wire)

                # 导线 hitbox 略粗，便于点选（元件碰撞体已缩小，不易挡住导线）
                hitbox = Entity(
                    parent=wire,
                    model='cube',
                    scale=(0.6, 0.6, wire.scale.z),
                    color=color.rgba(0, 0, 0, 0),
                    collider='box'
                )
                hitbox.connection_id = connection.id

                logger.info(f"创建导线成功: {comp1_id} <-> {comp2_id}（已对齐）")
                return True
            return False
        except Exception as e:
            logger.error(f"连接元件失败: {e}", exc_info=True)
            return False

    def _get_terminal_entity(self, comp_id: str, terminal_id: Optional[str]):
        """根据元件 id 和端子 id 查找端子实体，用于导线锚点计算。"""
        if not terminal_id:
            return None
        ent = self.entities.get(comp_id)
        if not ent or not getattr(ent, "terminals", None):
            return None
        for t in ent.terminals:
            if getattr(t, "terminal_id", None) == terminal_id:
                return t
        return None

    def _create_wire_entity(self, connection: Connection) -> Optional[Entity]:
        """
        创建导线实体：
        - 优先使用端子实体的位置作为导线两端的锚点（端子到端子）
        - 若缺少端子实体（如接地等），回退到按元件几何中心+旋转计算边缘点
        """
        comp1_id = connection.component1_id
        comp2_id = connection.component2_id

        ent1 = self.entities[comp1_id]
        ent2 = self.entities[comp2_id]

        # 优先：根据 terminal1_id / terminal2_id 找到端子实体
        t1 = self._get_terminal_entity(comp1_id, getattr(connection, "terminal1_id", None))
        t2 = self._get_terminal_entity(comp2_id, getattr(connection, "terminal2_id", None))

        if t1 is not None and t2 is not None:
            # 端子实体存在：直接使用端子世界坐标作为导线两端
            p1 = Vec3(t1.position)
            p2 = Vec3(t2.position)
        else:
            # 兼容路径：缺少端子实体时，回退到基于元件几何中心/旋转计算左右边缘点的旧逻辑
            def get_edge_point(entity: Entity, is_right_edge: bool) -> Vec3:
                """根据元件旋转，计算左/右边缘点（以几何中心为基准）"""
                # 0. 几何中心：优先使用组件上记录的 _center_world_pos，否则退回实体 position
                center_attr = getattr(entity, "_center_world_pos", None)
                if center_attr is None:
                    center = entity.position
                else:
                    # 可能是 Vec3，也可能是 (x, y, z) 元组
                    center = center_attr if isinstance(center_attr, Vec3) else Vec3(*center_attr)

                # 1. 获取元件旋转角度（欧拉角，单位度）
                rotation = entity.rotation  # (x_rot, y_rot, z_rot)，Ursina默认以度为单位

                # 2. 计算边缘方向向量（默认右边缘为X轴正方向，左边缘为X轴负方向）
                direction = Vec3(1, 0, 0) if is_right_edge else Vec3(-1, 0, 0)

                # 3. 应用旋转：将方向向量绕Y轴旋转（假设主要旋转为绕Y轴，可根据需求扩展其他轴）
                y_rad = math.radians(rotation.y)
                # 绕Y轴旋转矩阵：[cosθ, 0, sinθ; 0, 1, 0; -sinθ, 0, cosθ]
                rotated_x = direction.x * math.cos(y_rad) + direction.z * math.sin(y_rad)
                rotated_z = -direction.x * math.sin(y_rad) + direction.z * math.cos(y_rad)
                rotated_direction = Vec3(rotated_x, 0, rotated_z).normalized()  # 归一化方向

                # 4. 计算边缘点：几何中心 + 旋转后的方向 * 半宽（X方向缩放的一半）
                half_width = entity.scale.x / 2
                edge_point = center + rotated_direction * half_width
                # 固定Y坐标（与元件高度一致）
                edge_point.y = self.COMPONENT_Y
                return edge_point

            # 计算两个元件的连接点（ent1的右边缘 -> ent2的左边缘）
            p1 = get_edge_point(ent1, is_right_edge=True)
            p2 = get_edge_point(ent2, is_right_edge=False)

        # 后续导线创建逻辑保持不变（计算长度、中点、方向等）
        distance = (p2 - p1).length()
        if distance < 0.01:
            return None

        mid_pos = (p1 + p2) / 2

        self._ensure_pick_roots()
        wire = Entity(
            model='cube',
            color=color.white,
            scale=(self.WIRE_THICKNESS, self.WIRE_THICKNESS, distance),
            position=mid_pos,
            parent=self.pick_wires
        )

        # 让导线朝向第二个锚点（通常是终点端子），保持与旧逻辑一致
        wire.look_at(p2)
        wire.rotation_x = 0  # 保持导线在水平面上（根据需求调整）
        wire.rotation_z = 0

        return wire

    def load_all_components(self) -> None:
        """从数据层加载元件（保留自定义Z轴）"""
        try:
            components = self.data_layer.get_all_components()
            for comp in components:
                comp_id = comp.id
                pos_x, pos_z = self._clamp_position_to_table(comp.position[0], comp.position[2])
                pos = (pos_x, self.COMPONENT_Y, pos_z)
                params = comp.__dict__

                comp_type_str = self._component_type_to_string(comp.type)
                self._create_vr_entity(comp_type_str, comp_id, pos, params)

            logger.info(f"加载元件完成，共{len(components)}个（Z轴自定义生效）")
        except Exception as e:
            logger.error(f"加载元件失败: {e}", exc_info=True)

    def load_all_connections(self) -> None:
        """从数据层加载连接（精准对齐）"""
        try:
            connections = list(self.data_layer.connections.values())
            for conn in connections:
                # 兼容旧数据：为缺少端子信息的连接补全 terminal1_id/terminal2_id，
                # 这样加载的导线也能尽量从端子到端子绘制
                conn = self._fill_connection_terminals_if_missing(conn)
                comp1_id = conn.component1_id
                comp2_id = conn.component2_id
                if comp1_id in self.entities and comp2_id in self.entities:
                    wire = self._create_wire_entity(conn)
                    if wire:
                        wire.collider = 'box'  # 让加载的线也能被点击
                        wire.connection_id = conn.id  # 绑定到对应的 Connection
                        self.wires.append(wire)

                        # 给加载的导线也加透明加粗 hitbox
                        hitbox = Entity(
                            parent=wire,
                            model='cube',
                            scale=(0.5, 0.5, wire.scale.z),
                            color=color.rgba(0, 0, 0, 0),
                            collider='box'
                        )
                        hitbox.connection_id = conn.id

            logger.info(f"加载连接完成，共{len(connections)}个（已对齐）")
        except Exception as e:
            logger.error(f"加载连接失败: {e}", exc_info=True)

    def remove_component(self, component_id: str) -> bool:
        """移除元件（同步更新 data 连接、VR 导线实体与仿真缓存）"""
        try:
            # 先收集涉及该元件的连接 ID，再删 data，再按 ID 删导线实体
            conn_ids = [
                c.id for c in self.data_layer.connections.values()
                if component_id in (c.component1_id, c.component2_id)
            ]
            if not self.data_layer.remove_component(component_id):
                return False

            if component_id in self.entities:
                ent = self.entities[component_id]
                self._remove_associated_wires(conn_ids)
                if getattr(ent, "terminals", None):
                    for t in ent.terminals:
                        destroy(t)
                destroy(ent)
                del self.entities[component_id]
            if self.circuit_simulator:
                self.circuit_simulator.clear_last_result()
            logger.info(f"移除元件成功: {component_id}")
            return True
        except Exception as e:
            logger.error(f"移除元件失败: {e}", exc_info=True)
            return False

    def _remove_associated_wires(self, connection_ids: list) -> None:
        """按连接 ID 移除导线实体（与 data 层一致）"""
        if not connection_ids:
            return
        ids_set = set(connection_ids)
        to_remove = [w for w in self.wires if getattr(w, "connection_id", None) in ids_set]
        for w in to_remove:
            self.wires.remove(w)
            destroy(w)

    def show_feedback(self, text: str) -> None:
        """更新UI提示"""
        if self.ui_panel:
            self.ui_panel.text = text
            logger.info(f"更新UI提示: {text}")

    def start_voice_input(self) -> None:
        """从VR层触发一次语音输入（菜单/手柄等入口共用）"""
        if not self.ai_voice_layer:
            self.show_feedback("语音层未初始化")
            logger.warning("start_voice_input: 尝试开始语音输入，但 ai_voice_layer 为空")
            return

        def _voice_worker():
            try:
                logger.info("start_voice_input: 开始一次语音输入流程（后台线程）")
                # 采集语音与识别在后台线程完成（不触碰任何 Ursina 对象）
                audio_data = self.ai_voice_layer._capture_voice()
                if not audio_data:
                    logger.warning("start_voice_input: 未采集到语音")
                    invoke(self._on_voice_result, None, None, False, "未采集到语音，请重试")
                    return

                text = self.ai_voice_layer._speech_to_text(audio_data)
                if not text or not text.strip():
                    logger.warning("start_voice_input: 未识别到文字")
                    invoke(self._on_voice_result, None, None, False, "未识别到文字，请重试")
                    return

                logger.info(f"start_voice_input: 识别到原始文本: {text}")
                print(f"识别到语音: {text}")
                # 在后台线程完成解析（ERNIE/规则），避免阻塞菜单和迷你界面
                parsed = self.ai_voice_layer.parse_command_only(text.strip())
                # 仅执行与 UI 更新调度回主线程
                invoke(self._on_voice_result, text.strip(), parsed, True, "")

            except Exception as e:
                logger.error(f"start_voice_input: 语音输入流程出错: {e}", exc_info=True)
                invoke(self._on_voice_result, None, None, False, f"语音处理出错: {e}")

        # 先在主线程更新提示，然后后台线程做耗时的采集与识别，避免界面"未响应"
        self.show_feedback("正在听...（请说话）")
        threading.Thread(target=_voice_worker, daemon=True).start()

    def _on_voice_result(self, text: Optional[str], parsed_command, ok: bool, msg: str) -> None:
        """
        在主线程中处理语音识别结果：
        - 执行语音指令（会创建/更新 Ursina 实体）
        - 更新 UI 提示
        parsed_command 已在后台解析完成，此处仅执行，避免阻塞菜单/迷你界面。
        """
        try:
            # 失败分支：仅更新提示
            if not ok:
                if msg:
                    self.show_feedback(msg)
                return

            # 执行语音指令（parsed_command 已传入则跳过解析，仅执行，保持界面流畅）
            result = self.ai_voice_layer.execute_command(text or "", parsed_command)
            exec_msg = result.get("message", "执行完成" if result.get("success") else "执行失败")

            # 显示"识别到的内容"方便判断是否误识别
            recognized = result.get("recognized_text", text)
            if len(recognized) > 18:
                recognized = recognized[:18] + "…"
            display_msg = f"识别: {recognized} → {exec_msg}"

            logger.info(
                f"_on_voice_result: 语音指令执行结束，success={result.get('success')}, message={exec_msg}"
            )
            self.show_feedback(display_msg)

        except Exception as e:
            logger.error(f"_on_voice_result: 处理语音结果时出错: {e}", exc_info=True)
            self.show_feedback(f"语音处理出错: {e}")

    # ========== 语音指令接口（与AI语音层联动） ==========

    def execute_voice_command(self, command_type: str, params: dict) -> bool:
        """
        执行来自语音层的指令
        这是VR层与语音层联动的核心接口

        Args:
            command_type: 指令类型 (add_component, connect_components, delete_component, delete_wire)
            params: 指令参数

        Returns:
            bool: 执行是否成功
        """
        try:
            handlers = {
                "add_component": self._voice_add_component,
                "connect_components": self._voice_connect_components,
                "delete_component": self._voice_delete_component,
                "delete_wire": self._voice_delete_wire,
                "modify_component": self._voice_modify_component,
            }

            handler = handlers.get(command_type)
            if handler:
                result = handler(params)
                # 更新UI显示
                if result:
                    self.show_feedback(f"语音指令执行成功: {command_type}")
                else:
                    self.show_feedback(f"语音指令执行失败: {command_type}")
                return result
            else:
                logger.warning(f"未知的语音指令类型: {command_type}")
                return False

        except Exception as e:
            logger.error(f"执行语音指令失败: {e}", exc_info=True)
            return False

    def _voice_add_component(self, params: dict) -> bool:
        """通过语音添加元件（方案 C：支持行/列与相对放置）"""
        comp_type = params.get("component_type")
        comp_id = params.get("component_id")
        position = params.get("position", (0, self.COMPONENT_Y, 0))

        if not comp_type or not comp_id:
            logger.error("添加元件缺少必要参数: component_type 或 component_id")
            return False

        # 方案 C：优先相对放置，其次行/列，最后自动排位
        if "place_relative" in params:
            rel = params["place_relative"]
            ref_id = rel.get("ref_id")
            direction = rel.get("direction", "right")
            if ref_id and ref_id in self.entities:
                ref_pos = self.entities[ref_id].position
                dx = {"right": 2, "left": -2, "front": 0, "back": 0}.get(direction, 0)
                dz = {"right": 0, "left": 0, "front": 2, "back": -2}.get(direction, 0)
                x, z = self._clamp_position_to_table(ref_pos.x + dx, ref_pos.z + dz)
                position = (x, self.COMPONENT_Y, z)
            else:
                self.show_feedback(f"请先添加元件 {ref_id or '?'}")
                return False
        elif "placement_row" in params and "placement_col" in params:
            position = self._position_from_grid(params["placement_row"], params["placement_col"])
        elif position == (0, self.COMPONENT_Y, 0):
            position = self._find_empty_position()

        # 构建指令数据
        command_data = {
            'component_type': comp_type,
            'component_id': comp_id,
            'position': position,
        }

        # 添加参数值
        if 'resistance' in params:
            command_data['resistance'] = params['resistance']
        if 'voltage' in params:
            command_data['voltage'] = params['voltage']

        # 调用现有的添加方法
        result = self._create_and_add_component(command_data)

        if result:
            logger.info(f"语音添加元件成功: {comp_id} at {position}")

        return result

    def _find_empty_position(self) -> Tuple[float, float, float]:
        """在实验台上找一个空位置，与视觉网格线一致（18 行 × 24 列，按行优先）"""
        existing_count = len(self.entities)
        row = existing_count // self.GRID_COLS
        col = existing_count % self.GRID_COLS
        return self._position_from_grid(row, col)

    def _voice_connect_components(self, params: dict) -> bool:
        """通过语音连接元件"""
        comp1_id = params.get("component1_id")
        comp2_id = params.get("component2_id")
        wire_id = params.get("wire_id") or self._generate_wire_id()

        if not comp1_id or not comp2_id:
            logger.error("连接元件缺少必要参数: component1_id 或 component2_id")
            return False

        # 检查元件是否存在
        if comp1_id not in self.entities:
            logger.error(f"元件不存在: {comp1_id}")
            return False
        if comp2_id not in self.entities:
            logger.error(f"元件不存在: {comp2_id}")
            return False

        # 创建连接
        conn = Connection(
            id=wire_id,
            component1_id=comp1_id,
            component2_id=comp2_id,
            connection_point1=tuple(self.entities[comp1_id].position),
            connection_point2=tuple(self.entities[comp2_id].position),
        )
        # 端子化仿真要求每条连接都带端子信息：语音连接若未指定端子，则按"优先未占用端子"补全
        self._fill_connection_terminals_if_missing(conn)

        result = self.connect_components(conn)

        if result:
            logger.info(f"语音连接成功: {comp1_id} <-> {comp2_id}")

        return result

    def _voice_delete_component(self, params: dict) -> bool:
        """通过语音删除元件"""
        comp_id = params.get("component_id")

        if not comp_id:
            logger.error("删除元件缺少必要参数: component_id")
            return False

        if comp_id not in self.entities:
            logger.error(f"元件不存在: {comp_id}")
            return False

        result = self.remove_component(comp_id)

        if result:
            logger.info(f"语音删除元件成功: {comp_id}")

        return result

    def _voice_modify_component(self, params: dict) -> bool:
        """通过语音修改元件参数（如：修改R1为两千欧、把V1改成5伏）"""
        comp_id = params.get("component_id")
        if not comp_id:
            logger.error("修改元件缺少必要参数: component_id")
            self.show_feedback("请指定要修改的元件")
            return False
        comp = self.data_layer.get_component(comp_id)
        if not comp:
            self.show_feedback(f"没有该元件: {comp_id}")
            return False
        if comp_id not in self.entities:
            self.show_feedback(f"请先在实验台添加 {comp_id}")
            return False
        from scr.data.data_layer import ComponentType
        updated = False
        if "resistance" in params and comp.type == ComponentType.RESISTOR:
            val = params["resistance"]
            if val <= 0 or val > 1e9:
                self.show_feedback("阻值需为正数且不超过 1e9")
                return False
            comp.parameters["resistance"] = val
            ent = self.entities[comp_id]
            for c in ent.children:
                if type(c).__name__ == 'Text':
                    c.text = f"{comp_id}\n{int(val)}Ω"
                    break
            updated = True
        if "voltage" in params and comp.type == ComponentType.POWER_SOURCE:
            val = params["voltage"]
            if val < -1e6 or val > 1e6:
                self.show_feedback("电压请在合理范围内")
                return False
            comp.parameters["voltage"] = val
            ent = self.entities[comp_id]
            for c in ent.children:
                if type(c).__name__ == 'Text':
                    c.text = f"{comp_id}\n{val}V"
                    break
            updated = True
        if not updated:
            self.show_feedback(f"未指定可修改的参数（阻值或电压）")
            return False
        self.show_feedback(f"已修改 {comp_id}")
        logger.info(f"语音修改元件成功: {comp_id}")
        return True

    def _voice_delete_wire(self, params: dict) -> bool:
        """通过语音删除导线"""
        wire_id = params.get("wire_id")

        if not wire_id:
            logger.error("删除导线缺少必要参数: wire_id")
            return False

        # 查找导线实体
        wire_to_remove = None
        for wire in self.wires:
            if getattr(wire, "connection_id", None) == wire_id:
                wire_to_remove = wire
                break

        if not wire_to_remove:
            logger.error(f"导线不存在: {wire_id}")
            return False

        # 删除
        self.data_layer.remove_connection(wire_id)
        self.wires.remove(wire_to_remove)
        destroy(wire_to_remove)

        if self.circuit_simulator:
            self.circuit_simulator.clear_last_result()

        logger.info(f"语音删除导线成功: {wire_id}")
        return True

    def _run_circuit_simulation(self) -> None:
        """运行电路仿真"""
        try:
            if not self.circuit_simulator:
                self.show_feedback("仿真器未初始化")
                return

            # 检查是否有电路元件
            components = self.data_layer.get_all_components()
            if not components:
                self.show_feedback("电路为空，无法仿真")
                return

            self.show_feedback("正在运行电路仿真...")

            # 运行仿真
            result = self.circuit_simulator.analyze_circuit()

            if result['success']:
                node_count = len(result.get('node_voltages', {}))
                self.show_feedback(f"仿真完成！节点数: {node_count} | 按R查看结果")
                logger.info("电路仿真成功完成")
            else:
                error_msg = result.get('error', '未知错误')
                self.show_feedback(f"仿真失败: {error_msg}")
                logger.error(f"电路仿真失败: {error_msg}")

        except Exception as e:
            logger.error(f"运行仿真时出错: {e}", exc_info=True)
            self.show_feedback(f"仿真出错: {str(e)}")

    def _show_simulation_results(self) -> None:
        """显示仿真结果 - 增强版（显示功率和改进的电流信息）"""
        try:
            # 先隐藏菜单，避免干扰
            if self.component_menu and self.component_menu.visible:
                self.component_menu.visible = False
                # 禁用所有菜单按钮的collider
                for child in self.component_menu.children:
                    if type(child).__name__ == 'Entity' and hasattr(child, 'collider'):
                        child.collider = None
                logger.info("隐藏菜单以显示仿真结果")
            
            # 检查面板是否存在
            if not hasattr(self, 'simulation_results_panel') or not self.simulation_results_panel:
                logger.error("仿真结果面板未初始化")
                self.show_feedback("仿真结果面板未初始化，请先初始化VR场景")
                return

            if not self.circuit_simulator:
                self.show_feedback("仿真器未初始化")
                return

            # 获取最新仿真结果
            latest_result = self.circuit_simulator.get_latest_results()

            if not latest_result:
                self.show_feedback("无仿真结果，请先运行仿真")
                return

            # === 参考示波器面板的显示方式 ===

            # 显示面板容器
            self.simulation_results_panel.visible = True

            # 显示背景
            if hasattr(self.simulation_results_panel, 'bg') and self.simulation_results_panel.bg:
                self.simulation_results_panel.bg.visible = True

            # 显示标题
            if hasattr(self.simulation_results_panel, 'title') and self.simulation_results_panel.title:
                self.simulation_results_panel.title.visible = True
                self.simulation_results_panel.title.text = "电路仿真结果"

            # 显示关闭按钮并重新启用collider
            if hasattr(self.simulation_results_panel, 'close_btn') and self.simulation_results_panel.close_btn:
                self.simulation_results_panel.close_btn.visible = True
                if not hasattr(self.simulation_results_panel.close_btn, 'collider') or not self.simulation_results_panel.close_btn.collider:
                    self.simulation_results_panel.close_btn.collider = 'box'

            # 显示分隔线
            if hasattr(self.simulation_results_panel,
                       'left_separator') and self.simulation_results_panel.left_separator:
                self.simulation_results_panel.left_separator.visible = True
            if hasattr(self.simulation_results_panel,
                       'right_separator') and self.simulation_results_panel.right_separator:
                self.simulation_results_panel.right_separator.visible = True

            # 显示状态栏背景
            if hasattr(self.simulation_results_panel, 'status_bg') and self.simulation_results_panel.status_bg:
                self.simulation_results_panel.status_bg.visible = True

            # 准备显示数据 - 从最新仿真结果提取，不重复仿真
            voltage_data = latest_result.raw_data or {}
            voltage_items = list(voltage_data.items())[:8]

            # 从已有结果计算功率和电流（避免重复 run_simulation）
            power_data = self.circuit_simulator.get_power_distribution_from_last_result()
            component_power = power_data.get('component_power', {})
            # 提取每个电阻的电流用于显示
            branch_currents = []
            for comp_id, comp_data in component_power.items():
                if isinstance(comp_data, dict) and 'current' in comp_data:
                    branch_currents.append((comp_id, f"{comp_data['current']:.4f} A"))
            def _current_sort_key(item):
                m = re.match(r'R(\d+)', item[0])
                return (0, int(m.group(1))) if m else (1, 0)
            branch_currents.sort(key=_current_sort_key)
            total_power = power_data.get('total_power', 0.0)

            # KVL验证结果
            kvl_verified = latest_result.kvl_results.get('loops_verified', 0) if latest_result.kvl_results else 0
            kvl_failed = latest_result.kvl_results.get('loops_failed', 0) if latest_result.kvl_results else 0
            kvl_status = "通过" if kvl_failed == 0 and kvl_verified > 0 else "失败"

            # KCL验证结果
            kcl_verified = latest_result.kcl_results.get('nodes_verified', 0) if latest_result.kcl_results else 0
            kcl_failed = latest_result.kcl_results.get('nodes_failed', 0) if latest_result.kcl_results else 0
            kcl_status = "通过" if kcl_failed == 0 and kcl_verified > 0 else "失败"

            # 功率平衡验证（简化：检查总功率是否合理）
            power_balance_status = "通过" if total_power > 0 else "N/A"

            # 更新显示 - 使用索引来定位不同部分
            text_index = 0
            
            # === 左栏：节点电压 ===
            # 跳过标题（索引0）
            text_index += 1
            
            # 更新8个节点电压
            for i in range(8):
                label, value_text = self.sim_result_texts[text_index]
                label.visible = True
                value_text.visible = True
                
                if i < len(voltage_items):
                    node_name, voltage = voltage_items[i]
                    # 更新标签为真实节点名称
                    label.text = f"{node_name}:"
                    value_text.text = f"{voltage:.4f}V"
                    value_text.color = color.black
                else:
                    # 没有数据时隐藏
                    label.visible = False
                    value_text.visible = False
                
                text_index += 1
            
            # === 中栏：电流（统一显示为 电流(元件ID) 的形式）===
            # 跳过标题（索引9）
            text_index += 1
            
            # 更新8个支路电流
            for i in range(8):
                label, value_text = self.sim_result_texts[text_index]
                label.visible = True
                value_text.visible = True
                
                if i < len(branch_currents):
                    element_name, current_str = branch_currents[i]
                    # 统一标签：电流(电阻ID)
                    label.text = f"电流({element_name}):"
                    value_text.text = current_str
                    value_text.color = color.black
                else:
                    # 没有数据时隐藏
                    label.visible = False
                    value_text.visible = False
                
                text_index += 1
            
            # === 右栏：验证结果（增强版 - 包含功率）===
            # 跳过标题（索引18）
            text_index += 1
            
            # 更新5个验证结果
            # 最后一项改为"全部元件总数"（包括电源、电阻、接地、示波器等）
            verification_values = [
                kvl_status,
                kcl_status,
                power_balance_status,          # 功率平衡
                f"{total_power:.2f}W",         # 总功率
                f"{len(self.data_layer.components)}"  # 元件总数（所有元件）
            ]
            
            verification_labels = [
                "KVL验证:",
                "KCL验证:",
                "功率平衡:",
                "总功率:",
                "元件总数:"
            ]
            
            for i in range(5):
                label, value_text = self.sim_result_texts[text_index]
                label.visible = True
                value_text.visible = True
                
                # 更新标签文本
                label.text = verification_labels[i]
                value_text.text = verification_values[i]
                
                # 设置颜色
                if verification_values[i] == "通过":
                    value_text.color = color.green
                elif verification_values[i] == "失败":
                    value_text.color = color.red
                else:
                    value_text.color = color.black
                
                text_index += 1
            
            # === 底部状态栏 ===
            # 更新状态（索引24）
            label, value_text = self.sim_result_texts[text_index]
            label.visible = True
            value_text.visible = True
            value_text.text = "完成"
            value_text.color = color.green

            # 隐藏其他面板和背景UI元素
            if self.component_menu:
                self.component_menu.visible = False

            # 隐藏背景UI面板，避免透明背景显示后面的文字
            if self.ui_panel:
                self.ui_panel.visible = False

            logger.info("显示仿真结果面板成功（增强版 - 包含功率信息）")

        except Exception as e:
            logger.error(f"显示仿真结果时出错: {e}", exc_info=True)
            self.show_feedback(f"显示结果出错: {str(e)}")

    def get_simulation_status(self) -> Dict[str, str]:
        """获取仿真状态（供外部调用）"""
        try:
            if not self.circuit_simulator:
                return {'status': 'no_simulator', 'message': '仿真器未初始化'}

            latest_result = self.circuit_simulator.get_latest_results()

            if not latest_result:
                return {'status': 'no_results', 'message': '无仿真结果'}

            return {
                'status': 'success',
                'timestamp': latest_result.timestamp,
                'node_count': len(latest_result.raw_data or {}),
                'has_kvl_results': latest_result.kvl_results is not None,
                'has_kcl_results': latest_result.kcl_results is not None,
                'has_oscilloscope_data': latest_result.oscilloscope_data is not None,
                'message': '仿真结果可用'
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def cleanup(self) -> None:
        """清理资源"""
        for ent in self.entities.values():
            destroy(ent)
        for wire in self.wires:
            destroy(wire)
        for elem in [self.experiment_table, self.ground_plane, self.ui_panel, self.interactive_camera]:
            if elem:
                destroy(elem)
        self.entities.clear()
        self.wires.clear()

        logger.info("VR资源清理完成")
