# working_memory.py
"""
🧠 工作记忆模块 (Prefrontal Cortex Simulation)

模拟人类工作记忆的核心特性：
1. 容量限制 (Miller's 7±2)
2. 时间衰减 (Temporal Decay)
3. 注意力聚焦 (Attention Spotlight)
4. 上下文绑定 (Context Binding)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import torch


@dataclass
class MemorySlot:
    """单个记忆槽位"""
    concept_idx: int  # 词汇索引
    concept_word: str  # 词汇文本
    activation: float  # 激活强度 (0-1)
    timestamp: float  # 进入时间
    source: str = "input"  # 来源: input/inference/self
    relations: Dict = field(default_factory=dict)  # 关联的其他概念

    def decay(self, current_time: float, half_life: float = 30.0) -> float:
        """计算时间衰减后的激活值"""
        age = current_time - self.timestamp
        decay_factor = 0.5 ** (age / half_life)
        return self.activation * decay_factor


class WorkingMemory:
    """
    工作记忆系统

    特性:
    - 固定容量 (默认7个槽位)
    - 自动衰减
    - 重要性排序
    - 上下文绑定
    """

    def __init__(self, capacity: int = 7, decay_half_life: float = 60.0):
        """
        Args:
            capacity: 最大槽位数 (Miller's 7±2)
            decay_half_life: 半衰期(秒)，超过此时间激活值减半
        """
        self.capacity = capacity
        self.decay_half_life = decay_half_life
        self.slots: deque = deque(maxlen=capacity * 2)  # 预留空间，后续会筛选

        # 对话上下文
        self.conversation_id: Optional[str] = None
        self.speaker_id: Optional[str] = None

        # 注意力焦点 (当前最关注的概念)
        self.focus_idx: Optional[int] = None

    def attend(self, concept_idx: int, concept_word: str,
               activation: float = 1.0, source: str = "input",
               relations: Dict = None) -> None:
        """
        将一个概念纳入工作记忆

        Args:
            concept_idx: 词汇索引
            concept_word: 词汇文本
            activation: 初始激活强度
            source: 来源类型
            relations: 与其他概念的关系
        """
        current_time = time.time()

        # 检查是否已存在，如果存在则刷新
        for slot in self.slots:
            if slot.concept_idx == concept_idx:
                # 刷新激活值和时间戳
                slot.activation = min(1.0, slot.activation + activation * 0.5)
                slot.timestamp = current_time
                if relations:
                    slot.relations.update(relations)
                self.focus_idx = concept_idx
                return

        # 创建新槽位
        new_slot = MemorySlot(
            concept_idx=concept_idx,
            concept_word=concept_word,
            activation=activation,
            timestamp=current_time,
            source=source,
            relations=relations or {}
        )

        self.slots.append(new_slot)
        self.focus_idx = concept_idx

        # 如果超过容量，执行遗忘
        self._enforce_capacity()

    def attend_batch(self, indices: List[int], words: List[str],
                     activations: List[float] = None) -> None:
        """批量添加概念"""
        if activations is None:
            # 默认：越靠后的词激活越高（recency effect）
            activations = [0.5 + 0.5 * (i / len(indices)) for i in range(len(indices))]

        for idx, word, act in zip(indices, words, activations):
            if idx > 1:  # 跳过 PAD 和 UNK
                self.attend(idx, word, act)

    def get_active_concepts(self, threshold: float = 0.1) -> List[MemorySlot]:
        """获取当前活跃的概念（按激活强度排序）"""
        current_time = time.time()

        active = []
        for slot in self.slots:
            decayed_activation = slot.decay(current_time, self.decay_half_life)
            if decayed_activation >= threshold:
                # 返回一个带有衰减激活值的副本
                active.append(MemorySlot(
                    concept_idx=slot.concept_idx,
                    concept_word=slot.concept_word,
                    activation=decayed_activation,
                    timestamp=slot.timestamp,
                    source=slot.source,
                    relations=slot.relations
                ))

        # 按激活强度排序
        active.sort(key=lambda x: x.activation, reverse=True)
        return active[:self.capacity]

    def get_context_indices(self) -> List[int]:
        """获取当前上下文的词汇索引列表（用于注入到模型）"""
        active = self.get_active_concepts()
        return [slot.concept_idx for slot in active]

    def get_context_weights(self) -> torch.Tensor:
        """获取当前上下文的权重向量"""
        active = self.get_active_concepts()
        if not active:
            return None

        indices = [slot.concept_idx for slot in active]
        weights = [slot.activation for slot in active]

        return indices, weights

    def get_focus(self) -> Optional[MemorySlot]:
        """获取当前注意力焦点"""
        if self.focus_idx is None:
            return None

        for slot in self.slots:
            if slot.concept_idx == self.focus_idx:
                return slot
        return None

    def bind_context(self, conversation_id: str, speaker_id: str = None) -> None:
        """绑定对话上下文"""
        # 如果对话变了，清空记忆
        if self.conversation_id != conversation_id:
            self.clear()

        self.conversation_id = conversation_id
        self.speaker_id = speaker_id

    def clear(self) -> None:
        """清空工作记忆"""
        self.slots.clear()
        self.focus_idx = None

    def _enforce_capacity(self) -> None:
        """强制执行容量限制（遗忘最不活跃的）"""
        if len(self.slots) <= self.capacity:
            return

        current_time = time.time()

        # 计算所有槽位的当前激活值
        slot_activations = []
        for slot in self.slots:
            decayed = slot.decay(current_time, self.decay_half_life)
            slot_activations.append((slot, decayed))

        # 按激活值排序，保留最强的
        slot_activations.sort(key=lambda x: x[1], reverse=True)

        # 只保留 capacity 个
        survivors = [sa[0] for sa in slot_activations[:self.capacity]]

        self.slots.clear()
        for slot in survivors:
            self.slots.append(slot)

    def get_status(self) -> Dict:
        """获取工作记忆状态（用于调试/显示）"""
        active = self.get_active_concepts()
        return {
            "capacity": self.capacity,
            "used": len(active),
            "focus": self.focus_idx,
            "concepts": [
                {
                    "word": s.concept_word,
                    "activation": round(s.activation, 2),
                    "source": s.source
                }
                for s in active
            ],
            "conversation_id": self.conversation_id
        }

    def __repr__(self) -> str:
        active = self.get_active_concepts()
        concepts = [f"{s.concept_word}({s.activation:.1f})" for s in active[:5]]
        return f"WorkingMemory[{len(active)}/{self.capacity}]: {', '.join(concepts)}"


class EpisodicBuffer:
    """
    情景缓冲区 - 存储最近的对话片段

    用于：
    1. 代词消解 (它 -> 猫)
    2. 话题追踪
    3. 多轮推理
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.episodes: deque = deque(maxlen=max_turns)

    def add_turn(self, speaker: str, text: str, indices: List[int],
                 words: List[str], timestamp: float = None) -> None:
        """添加一轮对话"""
        self.episodes.append({
            "speaker": speaker,
            "text": text,
            "indices": indices,
            "words": words,
            "timestamp": timestamp or time.time()
        })

    def get_recent_concepts(self, n_turns: int = 3) -> List[int]:
        """获取最近n轮对话中出现的概念"""
        concepts = []
        for episode in list(self.episodes)[-n_turns:]:
            concepts.extend(episode["indices"])
        return concepts

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """
        简单代词消解
        找最近提到的名词作为代词指代
        """
        # 常见代词
        pronouns = {"它", "他", "她", "这", "那", "它们", "他们", "她们", "这些", "那些"}

        if pronoun not in pronouns:
            return None

        # 回溯查找最近的名词（简化版）
        for episode in reversed(list(self.episodes)):
            for word in reversed(episode["words"]):
                # 跳过代词和虚词
                if word not in pronouns and len(word) > 1:
                    return word

        return None

    def get_topic(self) -> Optional[str]:
        """获取当前话题（最常出现的概念）"""
        from collections import Counter

        all_words = []
        for episode in self.episodes:
            all_words.extend(episode["words"])

        if not all_words:
            return None

        # 统计词频，排除停用词
        stopwords = {"的", "是", "了", "在", "我", "你", "有", "和", "就", "都", "也", "很", "不"}
        filtered = [w for w in all_words if w not in stopwords and len(w) > 1]

        if not filtered:
            return None

        counter = Counter(filtered)
        return counter.most_common(1)[0][0]

    def clear(self) -> None:
        """清空情景缓冲区"""
        self.episodes.clear()
