import pybullet as p
import numpy as np
import re


class Oracle:
    def __init__(self, scene_manager):
        self.scene_manager = scene_manager

    @staticmethod
    def clean_object_name(name):
        """
        Clean object name by removing numeric suffixes like '_1', '_2', etc.
        This is used for matching instructions to object registry.
        
        Args:
            name: Object name (e.g., 'blue_cup_1')
            
        Returns:
            str: Cleaned name (e.g., 'blue cup')
        """
        # Remove trailing _N pattern
        clean = re.sub(r'_\d+$', '', name)
        # Replace underscores with spaces
        clean = clean.replace('_', ' ')
        return clean
    
    @staticmethod
    def generate_instruction(object_name):
        """
        Generate a templated instruction for touching an object.
        
        Args:
            object_name: Raw object name from registry
            
        Returns:
            str: Instruction text
        """
        clean_name = Oracle.clean_object_name(object_name)
        return f"touch the {clean_name}"

    def find_best_target(self, text_query):
        """
        解析文本指令，在场景中查找对应的物体 ID，并计算其物理抓取点。
        """
        text_query = text_query.lower()

        # 获取场景管理器中注册的物体字典 {obj_id: obj_name}
        # 假设 scene_manager 维护了 object_registry
        if hasattr(self.scene_manager, "object_registry"):
            registry = self.scene_manager.object_registry
        else:
            # 如果没有注册表，尝试通过 pybullet user data 或 fallback 方式获取
            # 这里为了稳健，假设如果不存在则是一个空字典
            registry = {}

        best_match_id = -1
        best_match_name = None
        best_match_count = 0

        # 简单的关键词匹配策略
        # 指令如 "pick up the blue cup"，物体名如 "blue_cup_1"
        # registry is {name: id}
        for name, obj_id in registry.items():
            # 清理物体名：去除尾部数字和下划线 "blue_cup_1" -> "blue cup"
            clean_name = name.split('_') # 过滤掉 '1', '2' 等 ID 后缀
            keywords = [k for k in clean_name if not k.isdigit()]  # 过滤掉 '1', '2' 等 ID 后缀

            # 检查物体名中的关键特征词是否出现在指令中
            # 例如：如果 "blue" 和 "cup" 都在 query 中，则匹配
            match_count = 0
            for k in keywords:
                if k in text_query:
                    match_count += 1

            # 只要匹配到了至少一个核心词（如 'cup'），就视为候选
            # 改进：优先选择匹配词更多的物体（更精确）
            if match_count > best_match_count:
                best_match_id = obj_id
                best_match_name = name
                best_match_count = match_count

        if best_match_id == -1:
            return None

        # --- 获取物理真值 ---
        pos, orn = p.getBasePositionAndOrientation(best_match_id)

        # --- 计算几何中心 (AABB) ---
        # 针对书架、盒子等不规则物体，AABB 的顶部中心通常是较好的吸盘抓取点
        aabb_min, aabb_max = p.getAABB(best_match_id)
        center_x = (aabb_min[0] + aabb_max[0]) / 2.0
        center_y = (aabb_min[1] + aabb_max[1]) / 2.0
        # Z轴取最高点，方便吸盘从上方接触
        top_z = aabb_max[2]

        gras_pos = [center_x, center_y, top_z]

        return {
            "name": best_match_name,
            "id": best_match_id,
            "pos": pos,  # 物体原点 (Base Position)
            "orn": orn,
            "gras_pos": gras_pos  # 计算出的顶部表面中心
        }
    
    def get_random_target(self):
        """
        Select a random target object from the registry.
        
        Returns:
            dict: Target info with id, name, pos, orn, gras_pos, or None if no objects
        """
        if not hasattr(self.scene_manager, "object_registry"):
            return None
        
        registry = self.scene_manager.object_registry
        if not registry:
            return None
        
        # Select random object
        name = np.random.choice(list(registry.keys()))
        obj_id = registry[name]
        
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        
        # Calculate grasp position (AABB top center)
        aabb_min, aabb_max = p.getAABB(obj_id)
        center_x = (aabb_min[0] + aabb_max[0]) / 2.0
        center_y = (aabb_min[1] + aabb_max[1]) / 2.0
        top_z = aabb_max[2]
        
        gras_pos = [center_x, center_y, top_z]
        
        return {
            "name": name,
            "id": obj_id,
            "pos": pos,
            "orn": orn,
            "gras_pos": gras_pos
        }
