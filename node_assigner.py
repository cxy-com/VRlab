"""
节点分配器 - 统一管理电路节点分配逻辑
负责为电路中的所有元件分配SPICE节点编号
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from scr.data.data_layer import (
    Component, Connection, ComponentType,
    TERMINAL_POSITIVE, TERMINAL_NEGATIVE, TERMINAL_1, TERMINAL_2, TERMINAL_COMMON,
)

logger = logging.getLogger(__name__)


class NodeAssigner:
    """节点分配器 - 基于电路拓扑分配节点编号"""
    
    def __init__(self, use_node_based: bool = False, use_terminal_based: bool = False):
        """
        Args:
            use_node_based: 若为 True，使用“以节点为本”的分配策略（旧版，根据 connections 合并节点）
            use_terminal_based: 若为 True，使用端子化分配（需 Connection 带 terminal1_id/terminal2_id）
        """
        self.logger = logger
        self.connection_graph: Dict[str, List[str]] = {}
        self.node_assignments: Dict[str, int] = {}
        self.use_node_based = use_node_based
        self.use_terminal_based = use_terminal_based
        # 以节点为本/端子模式：存储每个元件的 (node1, node2)
        self.component_node_pairs: Dict[str, Tuple[int, int]] = {}
        
        # 缓存机制
        self._cached_connection_graph: Optional[Dict[str, List[str]]] = None
        self._cached_components_hash: Optional[int] = None
        self._cached_connections_hash: Optional[int] = None
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cached_connection_graph = None
        self._cached_components_hash = None
        self._cached_connections_hash = None
        self.logger.debug("缓存已清除")
    
    def assign_nodes(self, components: Dict[str, Component], 
                    connections: Dict[str, Connection]) -> Dict[str, int]:
        """
        为所有元件分配节点编号
        
        参数:
            components: 元件字典 {component_id: Component}
            connections: 连接字典 {connection_id: Connection}
        
        返回:
            节点分配映射 {component_id: node_id}
        """
        try:
            self.connection_graph = self._build_connection_graph(components, connections)
            self.logger.debug(f"连接图: {self.connection_graph}")
            
            # 端子化模式：当所有连接都带端子信息时，使用 union-find 分配节点
            if self.use_terminal_based:
                all_have_terminals = all(
                    getattr(c, 'terminal1_id', None) and getattr(c, 'terminal2_id', None)
                    for c in connections.values()
                )
                if all_have_terminals and connections:
                    return self._assign_nodes_terminal_based(components, connections)
                self.logger.debug("端子化模式：部分连接无端子信息，回退到旧逻辑")
            
            if self.use_node_based:
                return self._assign_nodes_node_based(components, connections)
            
            # 原有逻辑
            self.node_assignments = {}
            self.component_node_pairs = {}
            self._assign_ground_nodes(components)
            groups = self._find_connected_groups(components)
            self.logger.debug(f"连通组: {groups}")
            
            next_node_id = 1
            for group in groups:
                next_node_id = self._assign_group_nodes(group, components, next_node_id)
            
            self.logger.info(f"节点分配完成: {len(self.node_assignments)}个元件")
            return self.node_assignments
            
        except Exception as e:
            self.logger.error(f"节点分配失败: {e}", exc_info=True)
            return {}
    
    def _assign_nodes_node_based(self, components: Dict[str, Component],
                                 connections: Dict[str, Connection]) -> Dict[str, int]:
        """
        以节点为本：根据 connections 合并端点得到 SPICE 节点，再为每个元件分配 (node1, node2)
        """
        self.component_node_pairs = {}
        self.node_assignments = {}
        
        conn_list = [(c.component1_id, c.component2_id) for c in connections.values()]
        if not conn_list:
            self.logger.warning("无连接，无法进行节点分配")
            return {}
        
        # 1. 每个 connection (A,B) 定义一个初始 junction，用 frozenset 表示
        #    合并规则：J(A,B) 与 J(C,B) 当 A 与 C 存在不经过 B 的路径时合并
        graph = self.connection_graph
        
        def path_exists(a: str, b: str, exclude: str) -> bool:
            """是否存在从 a 到 b 的路径（不经过 exclude）"""
            if a == b:
                return True
            visited = {exclude}
            stack = [a]
            while stack:
                cur = stack.pop()
                if cur == b:
                    return True
                if cur in visited:
                    continue
                visited.add(cur)
                for n in graph.get(cur, []):
                    if n not in visited:
                        stack.append(n)
            return False
        
        # 并查集：junction 用 (min(a,b), max(a,b)) 作为初始代表
        parent: Dict[Tuple[str, str], Tuple[str, str]] = {}
        
        def make_junction(a: str, b: str) -> Tuple[str, str]:
            return (min(a, b), max(a, b))
        
        def find(j: Tuple[str, str]) -> Tuple[str, str]:
            if j not in parent:
                parent[j] = j
            if parent[j] != j:
                parent[j] = find(parent[j])
            return parent[j]
        
        def union(j1: Tuple[str, str], j2: Tuple[str, str]) -> None:
            p1, p2 = find(j1), find(j2)
            if p1 != p2:
                parent[p2] = p1
        
        # 2. 对每条 connection 创建 junction，并按规则合并
        ground_comp = next((cid for cid, c in components.items() if c.type == ComponentType.GROUND), None)
        for a, b in conn_list:
            j = make_junction(a, b)
            find(j)
        for a, b in conn_list:
            j_ab = make_junction(a, b)
            for c, d in conn_list:
                if (a, b) == (c, d):
                    continue
                j_cd = make_junction(c, d)
                # 合并规则：J(A,B) 与 J(C,B) 当 A、C 存在不经过 B 的路径，且 A、C 均非 GND
                # （避免把中间节点误合并到地）
                def should_merge(aa: str, cc: str, bb: str) -> bool:
                    if aa == cc or ground_comp in (aa, cc):
                        return False
                    return path_exists(aa, cc, bb)
                # 只合并 J(A,B) 与 J(C,B)：共享 B，A/C 连通。不合并 J(A,B) 与 J(A,C)（同一元件两端）
                if b == d and a != c and should_merge(a, c, b):
                    union(j_ab, j_cd)
                elif b == c and a != d and should_merge(a, d, b):
                    union(j_ab, j_cd)
                # 所有含 GND 的 junction 合并为同一节点
                elif ground_comp and ground_comp in (a, b) and ground_comp in (c, d):
                    union(j_ab, j_cd)
        
        # 3. 收集所有 junction 的根，分配节点编号（含 GND 的 junction 固定为 0）
        roots: Dict[Tuple[str, str], int] = {}
        ground_comp = next((cid for cid, c in components.items() if c.type == ComponentType.GROUND), None)
        next_id = 1
        for a, b in conn_list:
            j = make_junction(a, b)
            r = find(j)
            if r not in roots:
                if ground_comp and ground_comp in (a, b):
                    roots[r] = 0
                else:
                    roots[r] = next_id
                    next_id += 1
        
        if ground_comp and 0 not in roots.values():
            roots[tuple(sorted([ground_comp, ground_comp]))] = 0
        
        # 4. 为每个元件确定两个节点
        comp_junctions: Dict[str, Set[Tuple[str, str]]] = {}
        for a, b in conn_list:
            j = find(make_junction(a, b))
            for cid in (a, b):
                if cid not in comp_junctions:
                    comp_junctions[cid] = set()
                comp_junctions[cid].add(j)
        
        for comp_id, component in components.items():
            if component.type == ComponentType.GROUND:
                self.node_assignments[comp_id] = 0
                self.component_node_pairs[comp_id] = (0, 0)
                continue
            juncs = comp_junctions.get(comp_id, set())
            if len(juncs) < 2:
                juncs = list(juncs)
                n1 = roots.get(juncs[0], 0) if juncs else 0
                self.node_assignments[comp_id] = n1
                self.component_node_pairs[comp_id] = (n1, 0)
            else:
                juncs = list(juncs)
                n1, n2 = roots.get(juncs[0], 0), roots.get(juncs[1], 0)
                self.node_assignments[comp_id] = n1
                self.component_node_pairs[comp_id] = (n1, n2)
        
        self.logger.info(f"以节点为本分配完成: {len(self.node_assignments)}个元件")
        return self.node_assignments
    
    def _assign_nodes_terminal_based(self, components: Dict[str, Component],
                                     connections: Dict[str, Connection]) -> Dict[str, int]:
        """
        端子化节点分配：每条连接 (A, t1) <-> (B, t2) 表示两个端子同属一个 SPICE 节点。
        使用 union-find 合并所有通过导线相连的端子，再为每个元件分配 (node1, node2)。
        """
        self.component_node_pairs = {}
        self.node_assignments = {}
        
        # (comp_id, terminal_id) -> 并查集父节点
        parent: Dict[Tuple[str, str], Tuple[str, str]] = {}
        
        def find(pt: Tuple[str, str]) -> Tuple[str, str]:
            if pt not in parent:
                parent[pt] = pt
            if parent[pt] != pt:
                parent[pt] = find(parent[pt])
            return parent[pt]
        
        def union(pt1: Tuple[str, str], pt2: Tuple[str, str]) -> None:
            p1, p2 = find(pt1), find(pt2)
            if p1 != p2:
                parent[p2] = p1
        
        ground_comp = next((cid for cid, c in components.items() if c.type == ComponentType.GROUND), None)
        
        # 1. 对每条连接，将两端端子 union 在一起
        for conn in connections.values():
            t1 = getattr(conn, 'terminal1_id', None) or ""
            t2 = getattr(conn, 'terminal2_id', None) or ""
            if not t1 or not t2:
                continue
            pt1 = (conn.component1_id, t1)
            pt2 = (conn.component2_id, t2)
            find(pt1)
            find(pt2)
            union(pt1, pt2)
        
        # 2. 确保所有元件的端子都在并查集中（未连接的端子单独成类）
        for comp_id, component in components.items():
            if component.type == ComponentType.POWER_SOURCE:
                for t in [TERMINAL_POSITIVE, TERMINAL_NEGATIVE]:
                    pt = (comp_id, t)
                    find(pt)
            elif component.type == ComponentType.RESISTOR:
                for t in [TERMINAL_1, TERMINAL_2]:
                    pt = (comp_id, t)
                    find(pt)
            elif component.type == ComponentType.GROUND:
                pt = (comp_id, TERMINAL_COMMON)
                find(pt)
        
        # 3. 为每个等价类分配节点编号：含 GND 的为 0，其余为 1, 2, ...
        root_to_node: Dict[Tuple[str, str], int] = {}
        next_id = 1
        for pt in parent:
            r = find(pt)
            if r not in root_to_node:
                comp_id = r[0]
                if ground_comp and comp_id == ground_comp:
                    root_to_node[r] = 0
                else:
                    root_to_node[r] = next_id
                    next_id += 1
        
        if ground_comp and 0 not in root_to_node.values():
            gnd_pt = (ground_comp, TERMINAL_COMMON)
            if gnd_pt in parent:
                root_to_node[find(gnd_pt)] = 0
        
        # 3. 为每个元件确定 (node1, node2)
        for comp_id, component in components.items():
            if component.type == ComponentType.GROUND:
                self.node_assignments[comp_id] = 0
                self.component_node_pairs[comp_id] = (0, 0)
                continue
            
            nodes: List[int] = []
            if component.type == ComponentType.POWER_SOURCE:
                for t in [TERMINAL_POSITIVE, TERMINAL_NEGATIVE]:
                    pt = (comp_id, t)
                    if pt in parent:
                        nodes.append(root_to_node.get(find(pt), 0))
            elif component.type == ComponentType.RESISTOR:
                for t in [TERMINAL_1, TERMINAL_2]:
                    pt = (comp_id, t)
                    if pt in parent:
                        nodes.append(root_to_node.get(find(pt), 0))
            else:
                continue
            
            if len(nodes) < 2:
                n1 = nodes[0] if nodes else 0
                self.node_assignments[comp_id] = n1
                self.component_node_pairs[comp_id] = (n1, 0)
            else:
                n1, n2 = nodes[0], nodes[1]
                self.node_assignments[comp_id] = n1
                # 电压源负极内部逻辑接地：SPICE 中负极固定为节点 0，不在 VR 中要求连线到 GND
                if component.type == ComponentType.POWER_SOURCE:
                    self.component_node_pairs[comp_id] = (n1, 0)
                else:
                    self.component_node_pairs[comp_id] = (n1, n2)
        
        self.logger.info(f"端子化节点分配完成: {len(self.node_assignments)}个元件")
        return self.node_assignments
    
    def _build_connection_graph(self, components: Dict[str, Component],
                                connections: Dict[str, Connection]) -> Dict[str, List[str]]:
        """构建元件连接图（带缓存）"""
        # 计算当前数据的哈希值
        components_hash = hash(tuple(sorted(components.keys())))
        connections_hash = hash(tuple(sorted(connections.keys())))
        
        # 检查缓存是否有效
        if (self._cached_connection_graph is not None and
            self._cached_components_hash == components_hash and
            self._cached_connections_hash == connections_hash):
            self.logger.debug("使用缓存的连接图")
            return self._cached_connection_graph
        
        # 构建新的连接图
        self.logger.debug("构建新的连接图")
        graph = {comp_id: [] for comp_id in components.keys()}
        
        for connection in connections.values():
            comp1 = connection.component1_id
            comp2 = connection.component2_id
            
            if comp1 in graph and comp2 in graph:
                graph[comp1].append(comp2)
                graph[comp2].append(comp1)
        
        # 更新缓存
        self._cached_connection_graph = graph
        self._cached_components_hash = components_hash
        self._cached_connections_hash = connections_hash
        
        return graph
    
    def _assign_ground_nodes(self, components: Dict[str, Component]) -> None:
        """为接地元件分配节点0"""
        for comp_id, component in components.items():
            if component.type == ComponentType.GROUND:
                self.node_assignments[comp_id] = 0
                self.logger.debug(f"接地元件 {comp_id} -> 节点0")
    
    def _find_connected_groups(self, components: Dict[str, Component]) -> List[List[str]]:
        """找到所有连通的元件组（使用DFS）"""
        visited: Set[str] = set()
        groups: List[List[str]] = []
        
        for comp_id in components.keys():
            if comp_id not in visited:
                group = []
                self._dfs_visit(comp_id, visited, group)
                if group:
                    groups.append(group)
        
        return groups
    
    def _dfs_visit(self, comp_id: str, visited: Set[str], group: List[str]) -> None:
        """深度优先搜索访问"""
        visited.add(comp_id)
        group.append(comp_id)
        
        for neighbor in self.connection_graph.get(comp_id, []):
            if neighbor not in visited:
                self._dfs_visit(neighbor, visited, group)
    
    def _assign_group_nodes(self, group: List[str], 
                           components: Dict[str, Component],
                           start_node_id: int) -> int:
        """
        为一个连通组分配节点
        
        策略:
        1. 检测电路拓扑类型（串联/并联/混联）
        2. 根据类型选择不同的节点分配策略
        3. 为每个元件分配节点
        
        返回: 下一个可用的节点ID
        """
        # 检测拓扑类型
        topology_type = self._detect_topology_type(group, components)
        self.logger.debug(f"检测到拓扑类型: {topology_type}")
        
        # 找到电源元件作为起点
        power_source = None
        ground = None
        for comp_id in group:
            component = components.get(comp_id)
            if component:
                if component.type == ComponentType.POWER_SOURCE:
                    power_source = comp_id
                elif component.type == ComponentType.GROUND:
                    ground = comp_id
        
        if not power_source:
            # 如果没有电源，使用第一个非接地元件
            for comp_id in group:
                component = components.get(comp_id)
                if component and component.type != ComponentType.GROUND:
                    power_source = comp_id
                    break
        
        if not power_source:
            return start_node_id
        
        # 根据拓扑类型分配节点
        if topology_type == 'parallel':
            return self._assign_parallel_nodes(group, components, power_source, start_node_id)
        elif topology_type == 'mixed':
            # 检查是否是"并联的串联支路"（每条支路内部有串联）
            if ground:
                paths = self._find_all_paths(power_source, ground, group)
                # 如果有多条路径且至少有一条路径长度>3，这是并联的串联支路
                if len(paths) > 1 and any(len(p) > 3 for p in paths):
                    self.logger.debug(f"检测到并联的串联支路，路径数={len(paths)}")
                    return self._assign_parallel_series_nodes(paths, components, power_source, start_node_id)
            # 否则使用默认串联分配
            return self._assign_series_nodes(group, components, power_source, start_node_id)
        else:
            # 串联电路
            return self._assign_series_nodes(group, components, power_source, start_node_id)
    
    def _assign_series_nodes(self, group: List[str], 
                            components: Dict[str, Component],
                            start_comp_id: str,
                            start_node_id: int) -> int:
        """
        为串联电路分配节点
        
        策略:
        - 电源: 正极=start_node_id, 负极=0(接地)
        - 电阻: 输入节点=前一个元件的输出节点
        - 每个电阻的node_id表示其输入节点
        - 最后一个元件: 输出节点=0(接地)
        
        例如：V0 -> R1 -> R2 -> GND
        - V0: node_1
        - R1: node_1 (输入=V0的正极)
        - R2: node_2 (输入=R1的输出)
        - 输出节点通过get_component_nodes()查找连接确定
        """
        # 按拓扑顺序排列元件
        ordered = self._order_components_from_start(group, start_comp_id)
        self.logger.debug(f"拓扑顺序: {ordered}")
        
        current_input_node = start_node_id
        next_node_id = start_node_id + 1
        
        for i, comp_id in enumerate(ordered):
            # 跳过已分配的元件（如接地）
            if comp_id in self.node_assignments:
                continue
            
            component = components.get(comp_id)
            if not component:
                continue
            
            if component.type == ComponentType.POWER_SOURCE:
                # 电源: 正极=current_input_node
                self.node_assignments[comp_id] = current_input_node
                self.logger.debug(f"电源 {comp_id} -> 节点{current_input_node}")
                # 电源的输出节点就是它的正极节点，下一个元件从这里开始
                # current_input_node保持不变
                
            elif component.type == ComponentType.RESISTOR:
                # 电阻: 输入节点=current_input_node
                self.node_assignments[comp_id] = current_input_node
                self.logger.debug(f"电阻 {comp_id} -> 节点{current_input_node} (输入)")
                
                # 检查是否是最后一个电阻（下一个是GND）
                is_last = False
                neighbors = self.connection_graph.get(comp_id, [])
                for neighbor_id in neighbors:
                    neighbor = components.get(neighbor_id)
                    if neighbor and neighbor.type == ComponentType.GROUND:
                        is_last = True
                        break
                
                if not is_last:
                    # 不是最后一个，创建中间节点
                    current_input_node = next_node_id
                    next_node_id += 1
                # 如果是最后一个，输出到GND (node_0)，不需要创建新节点
        
        return next_node_id
    
    def _order_components_from_start(self, group: List[str], start_comp_id: str) -> List[str]:
        """从起始元件开始，按拓扑顺序排列元件"""
        ordered = []
        visited = set()
        current = start_comp_id
        
        while current and current not in visited:
            visited.add(current)
            ordered.append(current)
            
            # 找到下一个未访问的邻居
            next_comp = None
            for neighbor in self.connection_graph.get(current, []):
                if neighbor in group and neighbor not in visited:
                    next_comp = neighbor
                    break
            
            current = next_comp
        
        # 添加剩余未访问的元件
        for comp_id in group:
            if comp_id not in visited:
                ordered.append(comp_id)
        
        return ordered
    
    def get_component_nodes(self, comp_id: str, 
                           components: Dict[str, Component]) -> Optional[Tuple[int, int]]:
        """
        获取元件的两个节点（输入节点，输出节点）
        
        返回: (input_node, output_node) 或 None
        """
        if self.component_node_pairs and comp_id in self.component_node_pairs:
            return self.component_node_pairs[comp_id]
        if comp_id not in self.node_assignments:
            return None
        
        component = components.get(comp_id)
        if not component:
            return None
        
        comp_node = self.node_assignments[comp_id]
        
        if component.type == ComponentType.POWER_SOURCE:
            # 电源: 正极=comp_node, 负极=0
            return (comp_node, 0)
        
        elif component.type == ComponentType.RESISTOR:
            # 电阻: 输入节点 = comp_node
            input_node = comp_node
            
            # 查找输出节点：检查所有邻居
            output_node = None
            neighbors = self.connection_graph.get(comp_id, [])
            
            for neighbor_id in neighbors:
                neighbor_component = components.get(neighbor_id)
                if not neighbor_component:
                    continue
                
                neighbor_node = self.node_assignments.get(neighbor_id)
                if neighbor_node is None:
                    continue
                
                # 如果邻居是接地，输出到 node_0
                if neighbor_component.type == ComponentType.GROUND:
                    output_node = 0
                    break
                
                # 如果邻居是电源，跳过（那是输入侧）
                if neighbor_component.type == ComponentType.POWER_SOURCE:
                    continue
                
                # 如果邻居是另一个电阻
                if neighbor_component.type == ComponentType.RESISTOR:
                    # 如果邻居的节点编号大于当前节点，那是输出侧
                    if neighbor_node > comp_node:
                        output_node = neighbor_node
                        break
                    # 如果邻居的节点编号等于当前节点，检查是否连接到GND
                    elif neighbor_node == comp_node:
                        # 这种情况下，两个电阻并联，都输出到GND
                        # 继续查找是否有GND连接
                        continue
            
            # 如果没有找到输出节点，默认输出到 GND (node_0)
            if output_node is None:
                output_node = 0
            
            return (input_node, output_node)
        
        elif component.type == ComponentType.GROUND:
            # 接地: 两端都是0
            return (0, 0)
        
        return None
    
    def _find_input_node(self, comp_id: str, components: Dict[str, Component]) -> int:
        """找到元件的输入节点（前一个元件的输出节点）"""
        # 找到所有连接到此元件的邻居
        neighbors = self.connection_graph.get(comp_id, [])
        
        # 找到节点编号小于当前元件的邻居（上游元件）
        comp_node = self.node_assignments.get(comp_id, 999)
        upstream_nodes = []
        
        for neighbor_id in neighbors:
            neighbor_node = self.node_assignments.get(neighbor_id)
            if neighbor_node is not None and neighbor_node < comp_node:
                neighbor_component = components.get(neighbor_id)
                if neighbor_component and neighbor_component.type != ComponentType.GROUND:
                    upstream_nodes.append(neighbor_node)
        
        if upstream_nodes:
            # 返回最大的上游节点（最接近的上游）
            return max(upstream_nodes)
        
        # 如果没有上游节点，可能是第一个元件，连接到电源
        # 找到电源节点
        for neighbor_id in neighbors:
            neighbor_component = components.get(neighbor_id)
            if neighbor_component and neighbor_component.type == ComponentType.POWER_SOURCE:
                return self.node_assignments.get(neighbor_id, 1)
        
        # 默认返回节点1
        return 1
    
    def _detect_topology_type(self, group: List[str], components: Dict[str, Component]) -> str:
        """
        检测电路拓扑类型
        
        返回: 'series' (串联), 'parallel' (并联), 'mixed' (混合)
        
        检测策略:
        1. 找到电源和接地
        2. 计算从电源到接地的路径数
        3. 1条路径=串联，多条路径=并联/混合
        """
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
            return 'series'  # 默认
        
        # 计算从电源到接地的所有路径
        paths = self._find_all_paths(power_source, ground, group)
        self.logger.debug(f"从{power_source}到{ground}的路径数: {len(paths)}")
        self.logger.debug(f"路径: {paths}")
        
        if len(paths) == 0:
            return 'series'
        elif len(paths) == 1:
            return 'series'
        else:
            # 多条路径，检查是否为纯并联
            if self._is_pure_parallel_by_paths(paths, power_source, ground):
                return 'parallel'
            else:
                return 'mixed'
    
    def _find_all_paths(self, start: str, end: str, group: List[str]) -> List[List[str]]:
        """
        找到从start到end的所有路径
        
        使用DFS查找所有不重复节点的路径
        """
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                all_paths.append(path.copy())
                return
            
            visited.add(current)
            
            for neighbor in self.connection_graph.get(current, []):
                if neighbor in group and neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
            
            visited.remove(current)
        
        dfs(start, end, [start], set())
        return all_paths
    
    def _is_pure_parallel_by_paths(self, paths: List[List[str]], 
                                   start: str, end: str) -> bool:
        """
        通过路径判断是否为纯并联
        
        纯并联特征:
        - 所有路径长度相同（都是3：电源->电阻->接地）
        - 路径之间除了起点和终点外没有公共节点
        """
        if len(paths) < 2:
            return False
        
        # 检查所有路径长度是否相同
        path_lengths = [len(p) for p in paths]
        if len(set(path_lengths)) != 1:
            return False
        
        # 检查路径长度是否为3（电源->元件->接地）
        if path_lengths[0] != 3:
            return False
        
        # 检查路径之间是否没有公共中间节点
        for i, path1 in enumerate(paths):
            for path2 in paths[i+1:]:
                # 获取中间节点（排除起点和终点）
                middle1 = set(path1[1:-1])
                middle2 = set(path2[1:-1])
                if middle1 & middle2:  # 有交集
                    return False
        
        return True
    
    def _is_pure_parallel(self, group: List[str], degrees: Dict[str, int], 
                         components: Dict[str, Component]) -> bool:
        """
        检查是否为纯并联电路
        
        纯并联特征:
        1. 有且仅有2个分支点（度数>2的节点）
        2. 所有其他元件都连接在这两个分支点之间
        """
        # 找到所有分支点（度数>2）
        branch_points = [comp_id for comp_id, degree in degrees.items() if degree > 2]
        
        # 纯并联应该有1-2个分支点
        # 1个分支点：电源作为分支点，另一端接地
        # 2个分支点：两个分支点之间有多条路径
        if len(branch_points) == 0:
            return False
        
        if len(branch_points) > 2:
            # 超过2个分支点，可能是复杂混合电路
            return False
        
        # 检查所有非分支点元件是否都是度数为2（连接两个节点）
        for comp_id in group:
            if comp_id in branch_points:
                continue
            
            component = components.get(comp_id)
            if component and component.type == ComponentType.GROUND:
                continue
            
            # 非分支点元件应该度数为2（连接在两个节点之间）
            if degrees[comp_id] != 2:
                return False
        
        return True
    
    def _assign_parallel_nodes(self, group: List[str], 
                              components: Dict[str, Component],
                              power_source_id: str,
                              start_node_id: int) -> int:
        """
        为并联电路分配节点
        
        策略:
        - 所有并联元件共享相同的两个节点
        - 输入节点: start_node_id (电源正极)
        - 输出节点: 0 (接地)
        
        示例:
            V1 ─┬─ R1 ─┬─ GND
                ├─ R2 ─┤
                └─ R3 ─┘
            
            节点分配:
            V1: 节点1
            R1, R2, R3: 都是 (1, 0)
        """
        self.logger.info(f"检测到并联电路，共{len(group)}个元件")
        
        # 为电源分配节点
        if power_source_id not in self.node_assignments:
            self.node_assignments[power_source_id] = start_node_id
            self.logger.debug(f"电源 {power_source_id} -> 节点{start_node_id}")
        
        # 为所有并联元件分配相同的节点
        # 输入节点 = 电源节点，输出节点 = 0（接地）
        input_node = start_node_id
        output_node = 0
        
        for comp_id in group:
            if comp_id in self.node_assignments:
                # 已分配（电源或接地）
                continue
            
            component = components.get(comp_id)
            if not component:
                continue
            
            if component.type == ComponentType.RESISTOR:
                # 并联电阻：都连接在输入节点和输出节点之间
                # 注意：这里我们只分配一个节点ID，实际的节点对在NetlistGenerator中处理
                self.node_assignments[comp_id] = input_node
                self.logger.debug(f"并联电阻 {comp_id} -> 节点{input_node} (输出到节点{output_node})")
        
        # 返回下一个可用节点ID
        return start_node_id + 1
    
    def _assign_parallel_series_nodes(self, paths: List[List[str]],
                                     components: Dict[str, Component],
                                     power_source_id: str,
                                     start_node_id: int) -> int:
        """
        为并联的串联支路分配节点
        
        支持两种情况：
        1. 电源直接连接到并联部分：V → [R1 || R2] → GND
        2. 电源先经过串联电阻再到并联部分：V → R0 → [R1 || R2] → GND
        
        策略：
        1. 找出所有路径共享的前缀（串联部分）
        2. 为串联部分分配节点
        3. 为并联部分分配节点
        """
        self.logger.info(f"为并联的串联支路分配节点，共{len(paths)}条路径")
        
        # 电源节点
        self.node_assignments[power_source_id] = start_node_id
        self.logger.debug(f"电源 {power_source_id} -> 节点{start_node_id}")
        current_node = start_node_id + 1
        
        # 1. 找出所有路径共享的前缀（串联部分）
        common_prefix = self._find_common_prefix(paths, components, power_source_id)
        self.logger.debug(f"共享前缀（串联部分）: {common_prefix}")
        
        # 2. 为共享的串联部分分配节点
        parallel_input_node = start_node_id  # 并联部分的输入节点
        
        if common_prefix:
            # 有共享的串联部分
            for resistor_id in common_prefix:
                self.node_assignments[resistor_id] = parallel_input_node
                self.logger.debug(f"串联电阻 {resistor_id} -> 节点{parallel_input_node}")
                # 下一个节点
                parallel_input_node = current_node
                current_node += 1
        
        # 3. 为每条并联路径分配节点
        for path_idx, path in enumerate(paths, 1):
            self.logger.debug(f"处理路径{path_idx}: {path}")
            
            # 提取路径中的电阻（排除电源、接地和共享前缀）
            resistors_in_path = []
            for comp_id in path:
                if comp_id == power_source_id:
                    continue
                if comp_id in common_prefix:
                    continue  # 跳过已处理的共享部分
                    
                component = components.get(comp_id)
                if component and component.type == ComponentType.GROUND:
                    # 接地固定为0
                    if comp_id not in self.node_assignments:
                        self.node_assignments[comp_id] = 0
                        self.logger.debug(f"接地 {comp_id} -> 节点0")
                elif component and component.type == ComponentType.RESISTOR:
                    resistors_in_path.append(comp_id)
            
            # 为路径中的电阻分配节点
            # 第一个电阻的输入节点是并联输入节点
            path_input_node = parallel_input_node
            
            for i, resistor_id in enumerate(resistors_in_path):
                if resistor_id in self.node_assignments:
                    # 已分配（可能在其他路径中）
                    self.logger.debug(f"电阻 {resistor_id} 已分配，跳过")
                    continue
                
                # 分配输入节点
                self.node_assignments[resistor_id] = path_input_node
                self.logger.debug(f"并联支路电阻 {resistor_id} -> 节点{path_input_node}")
                
                # 如果不是最后一个电阻，创建中间节点
                if i < len(resistors_in_path) - 1:
                    path_input_node = current_node
                    current_node += 1
                else:
                    # 最后一个电阻输出到GND，下一条路径重新从并联输入节点开始
                    path_input_node = parallel_input_node
        
        return current_node
    
    def _find_common_prefix(self, paths: List[List[str]], 
                           components: Dict[str, Component],
                           power_source_id: str) -> List[str]:
        """
        找出所有路径共享的前缀（串联部分）
        
        返回：共享的电阻ID列表（按顺序）
        """
        if not paths or len(paths) < 2:
            return []
        
        # 提取每条路径的电阻序列（排除电源和接地）
        resistor_sequences = []
        for path in paths:
            resistors = []
            for comp_id in path:
                if comp_id == power_source_id:
                    continue
                component = components.get(comp_id)
                if component and component.type == ComponentType.RESISTOR:
                    resistors.append(comp_id)
            resistor_sequences.append(resistors)
        
        # 找出共同前缀
        common_prefix = []
        min_length = min(len(seq) for seq in resistor_sequences)
        
        for i in range(min_length):
            # 检查所有序列在位置i的元素是否相同
            first_element = resistor_sequences[0][i]
            if all(seq[i] == first_element for seq in resistor_sequences):
                common_prefix.append(first_element)
            else:
                # 一旦发现不同，停止
                break
        
        return common_prefix
    
    def validate_assignments(self, components: Dict[str, Component]) -> Dict[str, Any]:
        """
        验证节点分配的有效性
        
        返回: {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """
        errors = []
        warnings = []
        
        # 检查是否所有“参与仿真”的元件都分配了节点
        # 示波器等仅用于测量/可视化的元件不强制要求节点，以免影响仿真与日志。
        for comp_id, component in components.items():
            if component.type == ComponentType.OSCILLOSCOPE:
                continue
            if comp_id not in self.node_assignments:
                errors.append(f"元件 {comp_id} 未分配节点")
        
        # 检查接地节点
        ground_count = sum(1 for node_id in self.node_assignments.values() if node_id == 0)
        if ground_count == 0:
            errors.append("没有接地节点")
        
        # 检查节点编号连续性（仅作为调试信息，不影响仿真）
        node_ids = set(self.node_assignments.values())
        max_node = max(node_ids) if node_ids else 0
        expected_nodes = set(range(max_node + 1))
        missing_nodes = expected_nodes - node_ids
        
        if missing_nodes and missing_nodes != {0}:
            # 节点编号不连续是正常的，特别是在混联电路中
            # 只记录为DEBUG级别，不作为警告
            self.logger.debug(f"节点编号不连续（正常现象）: 缺少 {missing_nodes}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
