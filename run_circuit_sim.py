"""
电路仿真层启动文件
类似于 run_vr.py，提供电路仿真的启动和测试功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 导入仿真层
from scr.circuit_sim.circuit_simulation_layer import (
    get_circuit_simulation_layer, SimulationConfig, SimulationEngine
)

# 导入数据层
from scr.data.data_layer import (
    get_data_layer, Resistor, PowerSource, Ground, Connection
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_circuit():
    """创建测试电路"""
    logger.info("创建测试电路...")
    
    data_layer = get_data_layer()
    data_layer.clear_all_data()
    
    # 创建简单电路：电源 -> 电阻1 -> 电阻2 -> 接地
    power_source = PowerSource(id="V1", position=(0, 0, 0), voltage=12.0)
    resistor1 = Resistor(id="R1", position=(2, 0, 0), resistance=1000.0)
    resistor2 = Resistor(id="R2", position=(4, 0, 0), resistance=2000.0)
    ground = Ground(id="GND1", position=(6, 0, 0))
    
    # 添加元件
    components = [power_source, resistor1, resistor2, ground]
    for comp in components:
        data_layer.add_component(comp)
    
    # 添加连接
    connections = [
        Connection(id="W1", component1_id="V1", component2_id="R1",
                  connection_point1=(0, 0, 0), connection_point2=(2, 0, 0)),
        Connection(id="W2", component1_id="R1", component2_id="R2",
                  connection_point1=(2, 0, 0), connection_point2=(4, 0, 0)),
        Connection(id="W3", component1_id="R2", component2_id="GND1",
                  connection_point1=(4, 0, 0), connection_point2=(6, 0, 0))
    ]
    
    for conn in connections:
        data_layer.add_connection(conn)
    
    logger.info(f"测试电路创建完成：{len(components)}个元件，{len(connections)}个连接")
    return data_layer


def run_basic_simulation():
    """运行基本仿真测试"""
    logger.info("=== 基本仿真测试 ===")
    
    # 创建测试电路
    data_layer = create_test_circuit()
    
    # 创建仿真器
    simulator = get_circuit_simulation_layer()
    
    # 运行仿真
    try:
        result = simulator.analyze_circuit()
        
        if result['success']:
            logger.info("✓ 仿真成功")
            
            # 显示节点电压
            node_voltages = result.get('node_voltages', {})
            logger.info("节点电压:")
            for node, voltage in node_voltages.items():
                logger.info(f"  {node}: {voltage:.3f}V")
            
            # 显示KVL验证
            kvl_result = result.get('kvl_verification', {})
            logger.info(f"KVL验证: {kvl_result.get('verified', False)}")
            
            # 显示KCL验证
            kcl_result = result.get('kcl_verification', {})
            logger.info(f"KCL验证: {kcl_result.get('verified', False)}")
            
            return True
        else:
            logger.error(f"✗ 仿真失败: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"仿真过程出错: {e}", exc_info=True)
        return False


def run_oscilloscope_test():
    """运行示波器测试"""
    logger.info("=== 示波器测试 ===")
    
    try:
        # 导入示波器类
        from scr.data.data_layer import Oscilloscope
        
        data_layer = get_data_layer()
        data_layer.clear_all_data()
        
        # 创建带示波器的电路
        power_source = PowerSource(id="V1", position=(0, 0, 0), voltage=5.0)
        resistor = Resistor(id="R1", position=(2, 0, 0), resistance=1000.0)
        ground = Ground(id="GND1", position=(4, 0, 0))
        
        # 添加示波器，测量节点1和2
        oscilloscope = Oscilloscope(
            id="OSC1",
            position=(0, 2, 0),
            sampling_rate=1000.0,
            measured_node_ids=[1, 2],
            channels=2
        )
        
        # 添加所有元件
        for comp in [power_source, resistor, ground, oscilloscope]:
            data_layer.add_component(comp)
        
        # 添加连接
        connections = [
            Connection(id="W1", component1_id="V1", component2_id="R1",
                      connection_point1=(0, 0, 0), connection_point2=(2, 0, 0)),
            Connection(id="W2", component1_id="R1", component2_id="GND1",
                      connection_point1=(2, 0, 0), connection_point2=(4, 0, 0))
        ]
        
        for conn in connections:
            data_layer.add_connection(conn)
        
        # 运行仿真
        simulator = get_circuit_simulation_layer()
        result = simulator.analyze_circuit()
        
        if result['success']:
            logger.info("✓ 示波器仿真成功")
            
            # 检查示波器数据
            osc_data = result.get('oscilloscope_data', {})
            if osc_data:
                logger.info("示波器数据:")
                for osc_id, data in osc_data.items():
                    channels = data.get('channels', {})
                    logger.info(f"  {osc_id}: {len(channels)}个通道")
                    for ch_name, ch_data in channels.items():
                        measurements = ch_data.get('measurements', {})
                        dc_offset = measurements.get('dc_offset', 0)
                        logger.info(f"    {ch_name}: DC偏移 = {dc_offset:.3f}V")
            
            return True
        else:
            logger.error(f"✗ 示波器仿真失败: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"示波器测试出错: {e}", exc_info=True)
        return False


def run_kvl_kcl_verification():
    """运行KVL/KCL验证测试"""
    logger.info("=== KVL/KCL验证测试 ===")
    
    try:
        # 创建复杂一点的电路进行验证
        data_layer = get_data_layer()
        data_layer.clear_all_data()
        
        # 电路：电源 + 两个并联电阻
        power_source = PowerSource(id="V1", position=(0, 0, 0), voltage=10.0)
        resistor1 = Resistor(id="R1", position=(2, 0, 0), resistance=1000.0)
        resistor2 = Resistor(id="R2", position=(2, 2, 0), resistance=2000.0)
        ground = Ground(id="GND1", position=(4, 0, 0))
        
        # 添加元件
        for comp in [power_source, resistor1, resistor2, ground]:
            data_layer.add_component(comp)
        
        # 添加连接（并联结构）
        connections = [
            Connection(id="W1", component1_id="V1", component2_id="R1",
                      connection_point1=(0, 0, 0), connection_point2=(2, 0, 0)),
            Connection(id="W2", component1_id="V1", component2_id="R2",
                      connection_point1=(0, 0, 0), connection_point2=(2, 2, 0)),
            Connection(id="W3", component1_id="R1", component2_id="GND1",
                      connection_point1=(2, 0, 0), connection_point2=(4, 0, 0)),
            Connection(id="W4", component1_id="R2", component2_id="GND1",
                      connection_point1=(2, 2, 0), connection_point2=(4, 0, 0))
        ]
        
        for conn in connections:
            data_layer.add_connection(conn)
        
        # 运行仿真和验证
        simulator = get_circuit_simulation_layer()
        laws_result = simulator.verify_circuit_laws()
        
        if laws_result.get('kvl_satisfied') and laws_result.get('kcl_satisfied'):
            logger.info("✓ KVL和KCL验证通过")
            
            # 显示详细信息
            kvl_details = laws_result.get('kvl_details', [])
            logger.info(f"KVL详情: {len(kvl_details)}个元件验证")
            
            kcl_details = laws_result.get('kcl_details', [])
            logger.info(f"KCL详情: {len(kcl_details)}个节点验证")
            
            return True
        else:
            logger.warning("✗ KVL或KCL验证失败")
            logger.info(f"KVL满足: {laws_result.get('kvl_satisfied', False)}")
            logger.info(f"KCL满足: {laws_result.get('kcl_satisfied', False)}")
            return False
            
    except Exception as e:
        logger.error(f"KVL/KCL验证测试出错: {e}", exc_info=True)
        return False


def run_performance_test():
    """运行性能测试"""
    logger.info("=== 性能测试 ===")
    
    try:
        import time
        
        # 创建较大的电路
        data_layer = get_data_layer()
        data_layer.clear_all_data()
        
        # 创建多个电阻串联
        power_source = PowerSource(id="V1", position=(0, 0, 0), voltage=24.0)
        data_layer.add_component(power_source)
        
        resistors = []
        for i in range(10):  # 10个电阻
            resistor = Resistor(id=f"R{i+1}", position=(i*2, 0, 0), resistance=100.0*(i+1))
            resistors.append(resistor)
            data_layer.add_component(resistor)
        
        ground = Ground(id="GND1", position=(20, 0, 0))
        data_layer.add_component(ground)
        
        # 添加串联连接
        prev_comp = "V1"
        for i, resistor in enumerate(resistors):
            conn = Connection(
                id=f"W{i+1}",
                component1_id=prev_comp,
                component2_id=resistor.id,
                connection_point1=(i*2-2, 0, 0),
                connection_point2=(i*2, 0, 0)
            )
            data_layer.add_connection(conn)
            prev_comp = resistor.id
        
        # 最后连接到地
        final_conn = Connection(
            id="W_final",
            component1_id=prev_comp,
            component2_id="GND1",
            connection_point1=(18, 0, 0),
            connection_point2=(20, 0, 0)
        )
        data_layer.add_connection(final_conn)
        
        # 测量仿真时间
        start_time = time.time()
        
        simulator = get_circuit_simulation_layer()
        result = simulator.analyze_circuit()
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        if result['success']:
            logger.info(f"✓ 性能测试完成")
            logger.info(f"电路规模: {len(data_layer.get_all_components())}个元件")
            logger.info(f"仿真时间: {simulation_time:.3f}秒")
            logger.info(f"节点数量: {len(result.get('node_voltages', {}))}")
            
            return True
        else:
            logger.error(f"✗ 性能测试失败: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"性能测试出错: {e}", exc_info=True)
        return False


def main():
    """主函数 - 运行所有测试"""
    logger.info("电路仿真层启动测试开始...")
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("基本仿真", run_basic_simulation),
        ("示波器功能", run_oscilloscope_test),
        ("KVL/KCL验证", run_kvl_kcl_verification),
        ("性能测试", run_performance_test)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n开始 {test_name} 测试...")
        try:
            result = test_func()
            test_results.append((test_name, result))
            logger.info(f"{test_name} 测试 {'通过' if result else '失败'}")
        except Exception as e:
            logger.error(f"{test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    logger.info("\n=== 测试总结 ===")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓" if result else "✗"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！电路仿真层工作正常")
    else:
        logger.warning("⚠️ 部分测试失败，请检查配置")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)