# component_renderer.py - 元件渲染模块
# 使用单实体结构（主体即带 model+color 的 Entity），避免父子层级导致颜色/形状异常。
# 电阻优先使用 assets/resistor/resistor.glb，若不存在或加载失败则回退为 cube。

from pathlib import Path
from typing import Dict, Any, Tuple
from ursina import Entity, Text, color
from ursina.mesh_importer import load_model
from ursina.models.procedural.cylinder import Cylinder

from scr.data.data_layer import TERMINAL_POSITIVE, TERMINAL_NEGATIVE, TERMINAL_1, TERMINAL_2
from scr.config import PROJECT_ROOT

import logging

logger = logging.getLogger(__name__)


def _get_resistor_model_and_scale_and_halfx():
    """
    电阻模型与缩放：(model, scale, half_x_world)。
    - 若存在 assets/resistor/resistor.glb：按“目标长度”自动缩放到接近原本 cube 电阻的视觉尺寸
    - half_x_world 用于把端子放到模型最外侧（更贴近引脚）
    """
    glb_dir = Path(PROJECT_ROOT) / "assets" / "resistor"
    glb_path = glb_dir / "resistor.glb"
    if glb_path.exists():
        try:
            model = load_model(glb_path.name, path=glb_dir)
            if model:
                # 目标：和电压方块（cube scale=0.4）在“视觉体量”上接近
                # 对细长电阻而言，用“厚度/高度”（y/z）对齐更符合肉眼感受
                target_visual_size = 0.4
                try:
                    min_b, max_b = model.getTightBounds()
                    dx = float(max_b.x - min_b.x)
                    dy = float(max_b.y - min_b.y)
                    dz = float(max_b.z - min_b.z)
                    cross = max(dy, dz)
                    if cross > 1e-6:
                        s = target_visual_size / cross
                    else:
                        s = 0.4
                except Exception:
                    s = 0.4

                # half_x_world：按缩放后的模型包围盒推端子到最外侧，略微外扩一点点
                try:
                    min_b, max_b = model.getTightBounds()
                    half_x_world = float((max_b.x - min_b.x) * s * 0.5)
                except Exception:
                    half_x_world = 0.22

                return model, (s, s, s), half_x_world
        except Exception:
            logger.warning("电阻模型加载失败，回退为 cube", exc_info=True)
    else:
        logger.warning(f"未找到电阻模型文件，回退为 cube: {glb_path}")
    # cube：与之前一致
    return "cube", (0.4, 0.4, 0.4), 0.22


def _find_model_in_assets(keywords, target_visual_size: float, fallback_scale):
    """
    通用资产加载助手：在项目根的 assets 下递归搜索名称包含关键字的模型文件（glb/gltf/obj/bam），
    若找到则按目标视觉尺寸缩放；否则返回立方体及兜底缩放。
    """
    base_dir = Path(PROJECT_ROOT) / "assets"
    if not base_dir.exists():
        return "cube", fallback_scale

    exts = (".glb", ".gltf", ".obj", ".bam")
    keywords = tuple(k.lower() for k in keywords if k)

    candidates = []
    try:
        for p in base_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                stem = p.stem.lower()
                if any(k in stem for k in keywords):
                    candidates.append(p)
    except Exception:
        # 资产很多或权限问题时，直接退回默认
        return "cube", fallback_scale

    if not candidates:
        return "cube", fallback_scale

    # 选一个“路径最短”的候选，通常更靠近根目录
    candidates.sort(key=lambda p: len(str(p)))
    model_path = candidates[0]
    glb_dir = model_path.parent

    try:
        model = load_model(model_path.name, path=glb_dir)
        if model:
            try:
                min_b, max_b = model.getTightBounds()
                dx = float(max_b.x - min_b.x)
                dy = float(max_b.y - min_b.y)
                dz = float(max_b.z - min_b.z)
                max_dim = max(dx, dy, dz)
                if max_dim > 1e-6:
                    s = target_visual_size / max_dim
                else:
                    s = fallback_scale[0]
            except Exception:
                s = fallback_scale[0]
            return model, (s, s, s)
    except Exception:
        pass

    return "cube", fallback_scale


def _get_power_source_model_and_scale():
    """
    电源模型与缩放：(model, scale, half_x_world, is_default, center_offset_world)。
    - 优先使用 assets/power_source/power_source.glb（带内嵌贴图）
    - 其次使用 assets/power_source/power_source.obj
    - 其次使用 assets/Voltage/vv.obj / V.glb
    - 最后在 assets 下搜索名称包含 power/source/电源/psu 的模型文件
    找到则按目标体量缩放到略大于原本 cube 电源（更醒目），并将模型中心对齐到原点
    center_offset_world 用于把模型整体平移，使其包围盒中心位于期望位置（例如两个端子之间）。
    """
    # 通用放大因子（用于非 V.glb 的其他电源模型）
    SCALE_UP = 1.6

    # 1) 优先使用新的电源模型 assets/power_source/power_source.glb（带贴图）
    power_source_glb_path = Path(PROJECT_ROOT) / "assets" / "power_source" / "power_source.glb"
    if power_source_glb_path.exists():
        try:
            model = load_model(power_source_glb_path.name, path=power_source_glb_path.parent)
            if model:
                try:
                    min_b, max_b = model.getTightBounds()
                    dx = float(max_b.x - min_b.x)
                    center_x = float((max_b.x + min_b.x) * 0.5)
                    center_y = float((max_b.y + min_b.y) * 0.5)
                    center_z = float((max_b.z + min_b.z) * 0.5)
                    target_width = 1.0
                    if dx > 1e-6:
                        s = target_width / dx
                        half_x_world = target_width * 0.5
                    else:
                        s = 0.6
                        half_x_world = 0.3
                    center_offset_world = (center_x * s, center_y * s, center_z * s)
                except Exception:
                    s = 0.6
                    half_x_world = 0.3
                    center_offset_world = (0.0, 0.0, 0.0)
                scale = (s, s, s)
                return model, scale, half_x_world, False, center_offset_world
        except Exception:
            pass

    # 2) 其次使用 assets/power_source/power_source.obj
    power_source_path = Path(PROJECT_ROOT) / "assets" / "power_source" / "power_source.obj"
    if power_source_path.exists():
        try:
            model = load_model(power_source_path.name, path=power_source_path.parent)
            if model:
                try:
                    min_b, max_b = model.getTightBounds()
                    dx = float(max_b.x - min_b.x)
                    center_x = float((max_b.x + min_b.x) * 0.5)
                    center_y = float((max_b.y + min_b.y) * 0.5)
                    center_z = float((max_b.z + min_b.z) * 0.5)
                    target_width = 1.0
                    if dx > 1e-6:
                        s = target_width / dx
                        half_x_world = target_width * 0.5
                    else:
                        s = 0.6
                        half_x_world = 0.3
                    center_offset_world = (center_x * s, center_y * s, center_z * s)
                except Exception:
                    s = 0.6
                    half_x_world = 0.3
                    center_offset_world = (0.0, 0.0, 0.0)
                scale = (s, s, s)
                return model, scale, half_x_world, False, center_offset_world
        except Exception:
            pass

    # 2) 其次使用 assets\Voltage\vv.obj / V.glb
    specific_candidates = [
        Path(PROJECT_ROOT) / "assets" / "Voltage" / "vv.obj",
        Path(PROJECT_ROOT) / "assets" / "Voltage" / "V.glb",
    ]
    for specific_path in specific_candidates:
        if specific_path.exists():
            try:
                model = load_model(specific_path.name, path=specific_path.parent)
                if model:
                    # 根据包围盒宽度(dx)缩放到指定目标宽度，保持各向同性缩放，
                    # 并记录包围盒中心用于后续居中。
                    try:
                        min_b, max_b = model.getTightBounds()
                        dx = float(max_b.x - min_b.x)
                        center_x = float((max_b.x + min_b.x) * 0.5)
                        center_y = float((max_b.y + min_b.y) * 0.5)
                        center_z = float((max_b.z + min_b.z) * 0.5)
                        target_width = 1.0  # 目标世界宽度，不会压满实验台
                        if dx > 1e-6:
                            s = target_width / dx
                            half_x_world = target_width * 0.5
                        else:
                            s = 0.6
                            half_x_world = 0.3
                        center_offset_world = (center_x * s, center_y * s, center_z * s)
                    except Exception:
                        s = 0.6
                        half_x_world = 0.3
                        center_offset_world = (0.0, 0.0, 0.0)
                    scale = (s, s, s)
                    return model, scale, half_x_world, False, center_offset_world
            except Exception:
                pass

    # 2) 通用关键字搜索
    model, scale = _find_model_in_assets(
        keywords=("power", "source", "电源", "psu", "vsource"),
        target_visual_size=0.4,
        fallback_scale=(0.4, 0.4, 0.4),
    )
    is_default = model == "cube"

    # 若找到了模型，再次读取包围盒计算半宽并放大整体，并根据中心偏移进行居中
    half_x_world = 0.22
    center_offset_world = (0.0, 0.0, 0.0)
    if not is_default:
        try:
            min_b, max_b = model.getTightBounds()
            dx = float(max_b.x - min_b.x)
            center_x = float((max_b.x + min_b.x) * 0.5)
            center_y = float((max_b.y + min_b.y) * 0.5)
            center_z = float((max_b.z + min_b.z) * 0.5)
            # 先计算最终缩放，再换算半宽与中心偏移
            final_scale = tuple(v * SCALE_UP for v in scale)
            if dx > 1e-6:
                half_x_world = float(dx * final_scale[0] * 0.5)
            scale = final_scale
            center_offset_world = (
                center_x * scale[0],
                center_y * scale[1],
                center_z * scale[2],
            )
        except Exception:
            scale = tuple(v * SCALE_UP for v in scale)
    else:
        # cube 情况：直接在原本大小基础上放大
        scale = tuple(v * SCALE_UP for v in scale)
        half_x_world = scale[0] * 0.5
        center_offset_world = (0.0, 0.0, 0.0)

    return model, scale, half_x_world, is_default, center_offset_world


def _get_oscilloscope_model_and_scale():
    """
    示波器模型与缩放：(model, scale, is_default)。
    - 在 assets 下搜索名称包含 osc/oscope/示波器/scope 的模型文件
    - 找到则按目标宽度缩放到接近原本 cube 示波器
    """
    model, scale = _find_model_in_assets(
        keywords=("osc", "oscope", "示波器", "scope"),
        target_visual_size=1.5,
        fallback_scale=(1.5, 1.0, 0.8),
    )
    return model, scale, (model == "cube")


def create_power_source_entity(vr, comp_id: str, pos: tuple, params: Dict[str, Any]) -> None:
    """创建电源实体：优先使用 assets 中的模型；自定义模型保留原始贴图颜色"""
    voltage = params.get('voltage', 5)
    vr._ensure_pick_roots()
    model_obj, scale, half_x_world, is_default, center_offset = _get_power_source_model_and_scale()
    px, py, pz = pos
    ox, oy, oz = center_offset

    # 默认 cube：用红色；自定义模型：使用模型自带贴图/颜色
    power_kwargs = dict(
        parent=vr.pick_components,
        model=model_obj,
        scale=scale,
        # 使用 center_offset 把模型的几何中心对齐到数据层位置
        position=(px - ox, py - oy, pz - oz),
        collider='box',
    )
    if is_default:
        power_kwargs["color"] = color.red
    power_source = Entity(**power_kwargs)
    power_source.component_id = comp_id
    # 记录几何中心世界坐标，供导线计算等逻辑使用
    power_source._center_world_pos = pos
    
    # 计算碰撞盒大小：根据实际模型缩放，稍微放大一点便于点击
    pick_scale = (
        scale[0] * 1.3,  # X轴稍大
        scale[1] * 1.3,  # Y轴稍大
        scale[2] * 1.3,  # Z轴稍大
    )
    # 不改变电源外观，只用一个稍大的隐形方块作为选中/点击 hitbox，便于用鼠标选中电源本体
    pick = Entity(parent=power_source, model='cube', scale=pick_scale, collider='box', visible=False)
    pick.component_id = comp_id
    pick._is_component_pick_collider = True

    # 电源标签：只显示元件编号（如 V1），放在机箱上方
    label_y = scale[1] * 1.2 if isinstance(scale, tuple) and len(scale) > 1 else 0.8
    Text(
        text=f"{comp_id}",
        parent=power_source,
        position=(0, label_y, 0),
        scale=10,
        color=color.yellow,
        background=color.rgba(0, 0, 0, 0.95),
        origin=(0, 0),
        billboard=True,
        bold=True
    )
    # 端子位置：以几何中心 pos 为基准，放在机箱右侧，距离与电阻类似
    center_pos = pos
    terminal_offset_x = max(float(getattr(vr, "TERMINAL_WORLD_OFFSET", 0.22)), half_x_world + 0.02)
    z_gap = 0.3
    y_offset = (scale[1] * 0.2) if isinstance(scale, tuple) and len(scale) > 1 else 0.2
    power_source.terminals = [
        # 右前：正端（+）- 红色
        vr._add_terminal_entity_world(
            center_pos, (terminal_offset_x, y_offset, z_gap), TERMINAL_POSITIVE, comp_id, color.red
        ),
        # 右后：负端（-）- 黑色
        vr._add_terminal_entity_world(
            center_pos, (terminal_offset_x, y_offset, -z_gap), TERMINAL_NEGATIVE, comp_id, color.black
        ),
    ]

    # 端子外观：正极红色，负极黑色
    for t in power_source.terminals:
        t.model = Cylinder(resolution=12, radius=0.5, height=1)
        # 保持端子的原始颜色（红色或黑色），不覆盖
        t.scale = (0.06, 0.06, 0.22)
        t.rotation = (0, 0, 90)

    vr.entities[comp_id] = power_source


def create_resistor_entity(vr, comp_id: str, pos: tuple, params: Dict[str, Any]) -> None:
    """创建电阻实体：优先使用 assets/resistor/resistor.glb（放大后与端子协调），否则为黄色立方体"""
    resistance = params.get('resistance', 1000)
    vr._ensure_pick_roots()
    model_obj, scale, half_x_world = _get_resistor_model_and_scale_and_halfx()
    resistor_kwargs = dict(
        parent=vr.pick_components,
        model=model_obj,
        scale=scale,
        position=pos,
        collider='box',
    )
    # 默认 cube：用黄色；自定义模型：保留模型自带贴图/颜色，避免被纯色覆盖导致“看起来像默认方块”
    if model_obj == "cube":
        resistor_kwargs["color"] = color.yellow
    resistor = Entity(**resistor_kwargs)
    resistor.component_id = comp_id
    
    # 计算碰撞盒大小：根据实际模型缩放
    pick_scale = (
        scale[0] * 1.1,  # X轴稍大
        scale[1] * 1.1,  # Y轴稍大
        scale[2] * 1.1,  # Z轴稍大
    )
    pick = Entity(parent=resistor, model='cube', scale=pick_scale, collider='box', visible=False)
    pick.component_id = comp_id
    pick._is_component_pick_collider = True

    Text(
        text=f"{comp_id}",  # 只显示元件标签（如 R1），不显示阻值
        parent=resistor,
        position=(0, 0.45, -0.25),
        scale=5,  # 再缩小一些，避免占据过多视野
        color=color.black,
        background=color.rgba(255, 255, 255, 0.9),
        origin=(0, 0),
        billboard=True,
        bold=True
    )
    # 端子位置：根据模型实际长度自适应（更贴近引脚/最外侧），并做一个最小偏移兜底
    terminal_offset_x = max(float(getattr(vr, "TERMINAL_WORLD_OFFSET", 0.22)), half_x_world + 0.02)
    resistor.terminals = [
        vr._add_terminal_entity_world(pos, (-terminal_offset_x, 0, 0), TERMINAL_1, comp_id, color.white),
        vr._add_terminal_entity_world(pos, (terminal_offset_x, 0, 0), TERMINAL_2, comp_id, color.white),
    ]

    # 将端子外观做成“引脚”样式：细金属圆柱，沿 X 轴朝外（仍保留 collider 用于连线）
    for t in resistor.terminals:
        t.model = Cylinder(resolution=12, radius=0.5, height=1)
        t.color = color.rgb(220, 220, 220)  # 金属灰
        t.scale = (0.06, 0.06, 0.22)
        t.rotation = (0, 0, 90)
    vr.entities[comp_id] = resistor


def create_ground_entity(vr, comp_id: str, pos: tuple, params: Dict[str, Any] = None) -> None:
    """创建接地实体（单立方体，黑色）"""
    vr._ensure_pick_roots()
    
    # 确保pos是一个有效的3D坐标
    if not pos or len(pos) < 3:
        pos = (0, 0, 0)
    
    ground = Entity(
        parent=vr.pick_components,
        model='cube',
        color=color.black,
        scale=(0.4, 0.4, 0.4),
        position=pos,
        collider='box',
    )
    ground.component_id = comp_id
    # 碰撞盒稍大一点便于点击
    pick = Entity(parent=ground, model='cube', scale=(1.2, 1.2, 1.2), collider='box', visible=False)
    pick.component_id = comp_id
    pick._is_component_pick_collider = True

    Text(
        text=f"{comp_id}\nGND",
        parent=ground,
        position=(0, 0.6, -0.3),
        scale=10,
        color=color.white,
        background=color.rgba(80, 80, 80, 0.95),
        origin=(0, 0),
        billboard=True,
        bold=True
    )
    ground.terminals = []
    vr.entities[comp_id] = ground


def create_oscilloscope_entity(vr, comp_id: str, pos: tuple, params: Dict[str, Any]) -> None:
    """创建示波器实体：优先使用 assets 中的模型；自定义模型保留原始贴图颜色"""
    vr._ensure_pick_roots()
    model_obj, scale, is_default = _get_oscilloscope_model_and_scale()
    # 自定义模型：使用白色，保留模型本身颜色；默认 cube：用蓝色区分示波器
    body_color = color.blue if is_default else color.white
    oscilloscope = Entity(
        parent=vr.pick_components,
        model=model_obj,
        color=body_color,
        scale=scale,
        position=pos,
        collider='box',
    )
    oscilloscope.component_id = comp_id
    # 不再为示波器创建文字标签，只保留模型本身
    # 为示波器创建两个测量端子（探头），都放在“正面”，避免一个被机箱遮挡
    # 参考电源端子的大小，但不影响电源本身的实现。
    terminal_offset_x = float(getattr(vr, "TERMINAL_WORLD_OFFSET", 0.22))
    # 对当前相机视角（从 z 负方向看向原点），"正面" 是朝向 -z 方向
    z_front = -0.25  # 朝向相机方向略微凸出，避免被机箱遮挡
    oscilloscope.terminals = [
        # 左前探头（黄色）
        vr._add_terminal_entity_world(
            pos, (-terminal_offset_x, 0.0, z_front), TERMINAL_1, comp_id, color.yellow
        ),
        # 右前探头（灰色）
        vr._add_terminal_entity_world(
            pos, (terminal_offset_x, 0.0, z_front), TERMINAL_2, comp_id, color.gray
        ),
    ]
    from ursina.models.procedural.cylinder import Cylinder as _Cyl
    for t in oscilloscope.terminals:
        t.model = _Cyl(resolution=12, radius=0.5, height=1)
        t.scale = (0.06, 0.06, 0.22)
        t.rotation = (0, 0, 90)
    vr.entities[comp_id] = oscilloscope
