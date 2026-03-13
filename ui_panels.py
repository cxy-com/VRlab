# -*- coding: utf-8 -*-
"""
UI 面板构建模块：
- 左上提示面板
- 元件选择菜单
- 示波器详情面板
- 修改参数面板
- 仿真结果面板

与 VRInteractionLayer 解耦，便于单独维护 UI 逻辑。
"""

import logging
from ursina import Entity, Text, InputField, camera, color

logger = logging.getLogger(__name__)


def create_ui_panel(vr) -> None:
    """创建左上角 UI 提示面板（含行/列数提示）。"""
    vr.ui_panel = Text(
        text=vr._default_ui_hint,
        position=(-0.30, 0.45),
        scale=1.0,
        color=color.black,
        background=color.rgba(255, 255, 255, 0.95),
        origin=(0, 0),
        billboard=False,
    )


def _create_menu_button(vr, label: str, action_type: str, comp_color, x_pos: float, y_pos: float) -> None:
    """创建单个菜单按钮（背景 + 文本）。"""
    # 创建半透明的按钮背景
    btn_bg = Entity(
        parent=vr.component_menu,
        model='quad',
        color=color.gray,  # 原始灰色
        scale=(0.35, 0.1, 0),
        position=(x_pos, y_pos, 0),
        collider=None,  # 初始化时不设置 collider，显示菜单时再启用
    )
    btn_bg.action_type = action_type
    logger.info(f"  按钮背景: {label} at ({x_pos}, {y_pos})")

    # 不显式指定 font 参数，使用 Text.default_font（与 run_vr.py 一致）
    # 文字 Z 轴设为 -0.1，确保在背景后面不被其他元素遮挡
    btn_text = Text(
        parent=vr.component_menu,
        text=label,
        position=(x_pos, y_pos, -0.1),
        scale=2,
        color=color.black,
        origin=(0, 0),
        background=False,
    )
    btn_text.action_type = action_type
    logger.info(f"  按钮文字: {label}")


def create_component_menu(vr) -> None:
    """创建元件选择菜单 - 两列布局。"""
    try:
        logger.info("=== 开始创建菜单（两列布局） ===")

        # 菜单主容器（挂载到 camera.ui）
        vr.component_menu = Entity(
            parent=camera.ui,
            visible=False,
            position=(0, 0),
            scale=1,
        )

        logger.info("正在创建菜单标题...")
        # 菜单标题（带背景）
        Text(
            parent=vr.component_menu,
            text="菜单",
            position=(0, 0.2, 10),
            scale=2,
            color=color.green,  # 文字颜色
            origin=(0, 0),
            background=True,  # 启用背景
            background_color=color.rgba(30, 30, 80, 0.95),  # 背景颜色
            z=10,
        )
        logger.info("  标题创建成功: z=10")

        logger.info("正在创建菜单项（两列布局）...")
        # 菜单项 - 两列布局
        # 左列：元件类型
        left_column = [
            ("电源", "power_source", color.gray, 0.12),
            ("信号发生器", "signal_generator", color.gray, -0.01),
            ("电阻", "resistor", color.gray, -0.14),
            ("接地", "ground", color.gray, -0.27),
            ("示波器", "oscilloscope", color.gray, -0.40),
        ]
        # 右列：操作功能（保留修改参数）
        right_column = [
            ("连线模式", "wire_mode", color.gray, 0.12),
            ("运行仿真", "run_simulation", color.gray, -0.01),
            ("查看仿真", "view_simulation", color.gray, -0.14),
            ("语音指令", "voice_input", color.gray, -0.27),
            ("修改参数", "modify_params", color.gray, -0.40),
        ]

        # 创建左列按钮
        for label, action_type, comp_color, y_pos in left_column:
            _create_menu_button(vr, label, action_type, comp_color, -0.2, y_pos)
        # 创建右列按钮
        for label, action_type, comp_color, y_pos in right_column:
            _create_menu_button(vr, label, action_type, comp_color, 0.2, y_pos)

    except Exception as e:
        logger.error(f"创建元件菜单失败: {e}", exc_info=True)


def create_oscilloscope_detail_panel(vr) -> None:
    """创建示波器详情面板（仅用于管理Matplotlib波形）。"""
    try:
        logger.info("正在创建示波器详情面板...")

        # 主容器（仅用于管理，不显示任何UI元素）
        vr.oscilloscope_detail_panel = Entity(
            parent=camera.ui,
            visible=False,
            position=(0, 0, 0),
            scale=1,
        )

        # 预留：由 VRInteractionLayer 在运行时填充的波形点列表
        vr.oscilloscope_voltage_points = []
        vr.oscilloscope_current_points = []

        logger.info("示波器详情面板创建成功（仅用于管理Matplotlib波形）")

    except Exception as e:
        logger.error(f"创建示波器详情面板失败: {e}", exc_info=True)


def create_modify_params_panel(vr) -> None:
    """创建修改参数迷你面板。"""
    try:
        panel_x, panel_y = 0.0, -0.2
        # 整体面板略放大一点，便于点击
        panel_scale = 0.7
        z_bg = 1000
        z_front = -1000
        bg_color = color.rgba(235, 238, 245, 0.98)
        text_color = color.rgba(25, 28, 35, 1)

        vr.modify_params_panel = Entity(
            parent=camera.ui,
            visible=False,
            position=(panel_x, panel_y, 0),
            scale=panel_scale,
        )
        vr.modify_params_panel.bg = Entity(
            parent=camera.ui,
            model='quad',
            # 外层白色大框再小一点、再透明一点
            color=color.rgba(235, 238, 245, 0.8),
            scale=(0.60 * panel_scale, 0.40 * panel_scale, 0),
            position=(panel_x, panel_y, z_bg),
            collider=None,
            visible=False,
        )
        vr.modify_params_panel.title = Text(
            parent=camera.ui,
            text="修改参数",
            position=(panel_x, panel_y + 0.18 * panel_scale, z_front),
            scale=1.4 * panel_scale,
            color=text_color,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr.modify_params_panel.param_label = Text(
            parent=camera.ui,
            text="阻值(Ω):",
            # 主参数标签整体上移，避免贴得太近按钮
            position=(panel_x - 0.2 * panel_scale, panel_y + 0.08 * panel_scale, z_front),
            scale=1.0 * panel_scale,
            color=text_color,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr.modify_params_panel.input_field = InputField(
            parent=camera.ui,
            default_value="1000",
            # 主输入框与标签一起上移
            position=(panel_x, panel_y + 0.08 * panel_scale, 0),
            # 输入框尺寸保持不变
            scale=(0.22 * panel_scale, 0.06 * panel_scale),
            visible=False,
            character_limit=12,
        )
        # 仅保留文字和光标，去掉 InputField 自带的深色背景
        if hasattr(vr.modify_params_panel.input_field, 'color'):
            vr.modify_params_panel.input_field.color = color.rgba(0, 0, 0, 0)
        if hasattr(vr.modify_params_panel.input_field, 'highlight_color'):
            vr.modify_params_panel.input_field.highlight_color = color.rgba(0, 0, 0, 0)
        if hasattr(vr.modify_params_panel.input_field, 'pressed_color'):
            vr.modify_params_panel.input_field.pressed_color = color.rgba(0, 0, 0, 0)
        # 参数数字改为白色
        if hasattr(vr.modify_params_panel.input_field, 'text_color'):
            vr.modify_params_panel.input_field.text_color = color.white
        vr.modify_params_panel.input_field_bg = Entity(
            parent=camera.ui,
            model='quad',
            # 输入框底色改为黑色，配合白色数字
            color=color.black,
            # 底框比输入区域略小一点
            scale=(0.20 * panel_scale, 0.055 * panel_scale, 0),
            # 跟随主输入框上移
            position=(panel_x, panel_y + 0.08 * panel_scale, 100),
            collider=None,
            visible=False,
        )
        # 频率输入（仅在交流信号源时显示）
        vr.modify_params_panel.freq_label = Text(
            parent=camera.ui,
            text="频率(Hz):",
            # 跟随主参数上移一档，保持垂直间距
            position=(panel_x - 0.2 * panel_scale, panel_y + 0.00 * panel_scale, z_front),
            scale=1.0 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr.modify_params_panel.freq_input_field = InputField(
            parent=camera.ui,
            default_value="1000",  # 默认频率改为1000Hz
            # 与频率标签同一高度
            position=(panel_x, panel_y + 0.00 * panel_scale, 0),
            scale=(0.22 * panel_scale, 0.06 * panel_scale),
            visible=False,
            character_limit=12,
        )
        if hasattr(vr.modify_params_panel.freq_input_field, 'color'):
            vr.modify_params_panel.freq_input_field.color = color.rgba(0, 0, 0, 0)
        if hasattr(vr.modify_params_panel.freq_input_field, 'highlight_color'):
            vr.modify_params_panel.freq_input_field.highlight_color = color.rgba(0, 0, 0, 0)
        if hasattr(vr.modify_params_panel.freq_input_field, 'pressed_color'):
            vr.modify_params_panel.freq_input_field.pressed_color = color.rgba(0, 0, 0, 0)
        if hasattr(vr.modify_params_panel.freq_input_field, 'text_color'):
            vr.modify_params_panel.freq_input_field.text_color = color.white
        vr.modify_params_panel.freq_input_field_bg = Entity(
            parent=camera.ui,
            model='quad',
            color=color.black,
            scale=(0.20 * panel_scale, 0.055 * panel_scale, 0),
            # 与频率输入框一起上移
            position=(panel_x, panel_y + 0.00 * panel_scale, 100),
            collider=None,
            visible=False,
        )
        vr.modify_params_panel.btn_ok = Entity(
            parent=camera.ui,
            model='quad',
            color=color.green,
            # 确定按钮加宽加高（保持原先高度）
            scale=(0.18 * panel_scale, 0.08 * panel_scale, 0),
            position=(panel_x - 0.11 * panel_scale, panel_y - 0.14 * panel_scale, z_front),
            collider=None,
            visible=False,
        )
        vr.modify_params_panel.btn_ok.is_modify_ok_button = True
        vr.modify_params_panel.btn_ok_label = Text(
            parent=camera.ui,
            text="确定",
            position=(panel_x - 0.11 * panel_scale, panel_y - 0.14 * panel_scale, z_front),
            scale=1.1 * panel_scale,
            color=text_color,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr.modify_params_panel.btn_cancel = Entity(
            parent=camera.ui,
            model='quad',
            color=color.orange,
            # 取消按钮同样放大，保持与确定按钮同一高度
            scale=(0.18 * panel_scale, 0.08 * panel_scale, 0),
            position=(panel_x + 0.11 * panel_scale, panel_y - 0.14 * panel_scale, z_front),
            collider=None,
            visible=False,
        )
        vr.modify_params_panel.btn_cancel.is_modify_cancel_button = True
        vr.modify_params_panel.btn_cancel_label = Text(
            parent=camera.ui,
            text="取消",
            position=(panel_x + 0.11 * panel_scale, panel_y - 0.14 * panel_scale, z_front),
            scale=1.1 * panel_scale,
            color=text_color,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr._modify_params_comp_id = None
        logger.info("修改参数面板创建成功")

    except Exception as e:
        logger.error(f"创建修改参数面板失败: {e}", exc_info=True)


def create_simulation_results_panel(vr) -> None:
    """创建仿真结果面板（三栏式布局）。"""
    try:
        logger.info("正在创建仿真结果面板（三栏式布局）...")

        panel_x, panel_y = 0.0, -0.15
        panel_scale = 0.70

        vr.simulation_results_panel = Entity(
            parent=camera.ui,
            visible=False,
            position=(panel_x, panel_y, 0),
            scale=panel_scale,
        )

        vr.simulation_results_panel.bg = Entity(
            parent=camera.ui,
            model='quad',
            color=color.rgba(180, 200, 230, 0.17),
            scale=(1.5 * panel_scale, 1.3 * panel_scale, 0),
            position=(panel_x, panel_y, -1000),
            collider=None,
            visible=False,
        )

        vr.simulation_results_panel.title = Text(
            parent=camera.ui,
            text="电路仿真结果",
            position=(panel_x, panel_y + 0.60 * panel_scale, 1000),
            scale=1.5 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )

        vr.simulation_results_panel.close_btn = Text(
            parent=camera.ui,
            text="×",
            position=(panel_x + 0.68 * panel_scale, panel_y + 0.56 * panel_scale, 1000),
            scale=3.0 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            visible=False,
        )
        vr.simulation_results_panel.close_btn.is_sim_close_button = True
        vr.simulation_results_panel.close_btn.collider = None

        # 分隔线
        left_separator = Entity(
            parent=camera.ui,
            model='quad',
            color=color.white,
            scale=(0.003, 0.9 * panel_scale, 0),
            position=(panel_x - 0.25 * panel_scale, panel_y, 1000),
            collider=None,
            visible=False,
        )
        right_separator = Entity(
            parent=camera.ui,
            model='quad',
            color=color.white,
            scale=(0.003, 0.9 * panel_scale, 0),
            position=(panel_x + 0.25 * panel_scale, panel_y, 1000),
            collider=None,
            visible=False,
        )
        vr.simulation_results_panel.left_separator = left_separator
        vr.simulation_results_panel.right_separator = right_separator

        vr.sim_result_texts = []

        # 左栏：节点电压
        left_title = Text(
            parent=camera.ui,
            text="节点电压",
            position=(panel_x - 0.50 * panel_scale, panel_y + 0.55 * panel_scale, 1000),
            scale=1.1 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        left_value = Text(parent=camera.ui, text="", visible=False)
        vr.sim_result_texts.append((left_title, left_value))

        for i in range(8):
            node_label = Text(
                parent=camera.ui,
                text=f"节点{i + 1}电压:",
                position=(panel_x - 0.68 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            node_value = Text(
                parent=camera.ui,
                text="None",
                position=(panel_x - 0.32 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            vr.sim_result_texts.append((node_label, node_value))

        # 中栏：电流（每个元件）
        middle_title = Text(
            parent=camera.ui,
            text="电流",
            position=(panel_x, panel_y + 0.55 * panel_scale, 1000),
            scale=1.1 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        middle_value = Text(parent=camera.ui, text="", visible=False)
        vr.sim_result_texts.append((middle_title, middle_value))

        for i in range(8):
            branch_label = Text(
                parent=camera.ui,
                text=f"支路{i + 1}电流:",
                position=(panel_x - 0.18 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            branch_value = Text(
                parent=camera.ui,
                text="None",
                position=(panel_x + 0.18 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            vr.sim_result_texts.append((branch_label, branch_value))

        # 右栏：验证结果
        right_title = Text(
            parent=camera.ui,
            text="验证结果",
            position=(panel_x + 0.50 * panel_scale, panel_y + 0.55 * panel_scale, 1000),
            scale=1.1 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        right_value = Text(parent=camera.ui, text="", visible=False)
        vr.sim_result_texts.append((right_title, right_value))

        verification_items = [
            "KVL验证:",
            "KCL验证:",
            "功率平衡:",
            "电流守恒:",
            "能量守恒:",
        ]
        for i, item_text in enumerate(verification_items):
            verify_label = Text(
                parent=camera.ui,
                text=item_text,
                position=(panel_x + 0.32 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            verify_value = Text(
                parent=camera.ui,
                text="None",
                position=(panel_x + 0.68 * panel_scale, panel_y + (0.40 - i * 0.09) * panel_scale, 1000),
                scale=0.85 * panel_scale,
                color=color.black,
                origin=(0, 0),
                background=False,
                visible=False,
            )
            vr.sim_result_texts.append((verify_label, verify_value))

        # 底部状态栏
        status_bg = Entity(
            parent=camera.ui,
            model='quad',
            color=color.rgba(180, 200, 230, 0.17),
            scale=(2.0 * panel_scale, 0.15 * panel_scale, 0),
            position=(panel_x, panel_y - 0.75 * panel_scale, -1000),
            collider=None,
            visible=False,
        )
        vr.simulation_results_panel.status_bg = status_bg

        status_label = Text(
            parent=camera.ui,
            text="仿真状态:",
            position=(panel_x - 0.85 * panel_scale, panel_y - 0.75 * panel_scale, 1000),
            scale=1.0 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        status_value = Text(
            parent=camera.ui,
            text="就绪",
            position=(panel_x - 0.45 * panel_scale, panel_y - 0.75 * panel_scale, 1000),
            scale=1.0 * panel_scale,
            color=color.black,
            origin=(0, 0),
            background=False,
            bold=True,
            visible=False,
        )
        vr.sim_result_texts.append((status_label, status_value))

        logger.info("仿真结果面板创建成功（三栏式布局）")

    except Exception as e:
        logger.error(f"创建仿真结果面板失败: {e}", exc_info=True)
