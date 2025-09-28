#selection_strategy.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
import random

# 定义池中元素的类型别名，方便维护
# 格式: (model_name, api_key, metadata_dict)
PoolItem = Tuple[str, str, Dict]
ItemIdentifier = Tuple[str, str]

class SelectionStrategy(ABC):
    """
    选择策略的抽象基类。
    """
    @abstractmethod
    def select(self, pool: List[PoolItem], failed_items: Set[ItemIdentifier]) -> Optional[PoolItem]:
        """
        从池中选择下一个要尝试的 (模型, 密钥) 对。
        :param pool: 完整的 (模型, 密钥, 元数据) 池。
        :param failed_items: 一个包含本次请求中已失败的项的集合。
        :return: 下一个要尝试的 PoolItem，如果无可用项则返回 None。
        """
        pass

class SequentialStrategy(SelectionStrategy):
    """
    顺序策略：按照池中定义的顺序依次选择。
    """
    def select(self, pool: List[PoolItem], failed_items: Set[ItemIdentifier]) -> Optional[PoolItem]:
        for item in pool:
            if (item[0], item[1]) not in failed_items:
                return item
        return None

class RandomStrategy(SelectionStrategy):
    """
    随机策略：从可用的选项中随机选择一个。
    """
    def select(self, pool: List[PoolItem], failed_items: Set[ItemIdentifier]) -> Optional[PoolItem]:
        # 筛选出尚未失败的可用项
        available_items = [
            item for item in pool 
            if (item[0], item[1]) not in failed_items
        ]
        if not available_items:
            return None
        return random.choice(available_items)