import sys
import os
sys.path.append(os.path.abspath("."))

from ursina import Ursina
from scr.vr_interaction.vr_interaction_layer import VRInteractionLayer
from scr.data.data_layer import Resistor, PowerSource, Ground, Connection

try:
    # 初始化Ursina应用
    app = Ursina()

    # 实例化VR层并初始化场景
    vr_layer = VRInteractionLayer()
    vr_layer.initialize_scene()

    # 运行应用
    # ===== 关键：将 VR 层的 input 方法绑定到全局输入 =====
    def input(key):
        """全局输入处理，委托给 VR 层"""
        vr_layer.input(key)

    # ===== 关键：将 VR 层的 update 方法绑定到全局帧更新 =====
    def update():
        """全局帧更新处理，委托给 VR 层"""
        vr_layer.update()

    app.run()

except Exception as e:
    import traceback
    traceback.print_exc()
    input("按回车键退出...")
