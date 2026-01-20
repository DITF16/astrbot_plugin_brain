"""
🔗 因果推理引擎 (Causal Reasoning Engine)

实现真正的因果推理能力：
1. 因果链搜索 (Causal Chain Search) — A导致B导致C
2. 逆向推理 (Backward Reasoning) — 为什么会发生X
3. 假设推理 (Hypothetical Reasoning) — 如果A会怎样
4. 解决方案搜索 (Solution Search) — 如何达成X

核心算法：
- BFS/A* 在因果图上搜索路径
- 路径置信度计算
- 多路径综合
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from collections import deque
import heapq

from .cognitive_graph_model import CHANNEL_CAUSES, CHANNEL_IS_A, CHANNEL_HAS_PROP, CHANNEL_ASSOCIATED


class ReasoningType(Enum):
    """推理类型枚举"""
    WHY = "why"  # 为什么 X 会发生？
    HOW = "how"  # 如何达成 X？
    WHAT_IF = "what_if"  # 如果 X 会怎样？
    PREDICT = "predict"  # X 会导致什么？
    EXPLAIN = "explain"  # 解释 X 和 Y 的关系
    NONE = "none"  # 非因果问题


@dataclass
class CausalLink:
    """因果链中的一个环节"""
    source: int  # 源概念索引
    target: int  # 目标概念索引
    source_word: str  # 源概念词
    target_word: str  # 目标概念词
    strength: float  # 因果强度 (0-1)
    channel: int  # 来源通道

    def __repr__(self):
        return f"{self.source_word} --({self.strength:.2f})--> {self.target_word}"


@dataclass
class CausalPath:
    """一条完整的因果路径"""
    links: List[CausalLink] = field(default_factory=list)
    total_confidence: float = 0.0

    @property
    def length(self) -> int:
        return len(self.links)

    @property
    def start_word(self) -> Optional[str]:
        if self.links:
            return self.links[0].source_word
        return None

    @property
    def end_word(self) -> Optional[str]:
        if self.links:
            return self.links[-1].target_word
        return None

    def get_words(self) -> List[str]:
        """获取路径上所有词"""
        if not self.links:
            return []
        words = [self.links[0].source_word]
        for link in self.links:
            words.append(link.target_word)
        return words

    def to_arrow_string(self) -> str:
        """转为箭头字符串: A → B → C"""
        words = self.get_words()
        return " → ".join(words)

    def __repr__(self):
        return f"CausalPath({self.to_arrow_string()}, conf={self.total_confidence:.2f})"


@dataclass
class ReasoningResult:
    """推理结果"""
    success: bool  # 是否成功推理
    reasoning_type: ReasoningType  # 推理类型
    query_concept: str  # 查询概念
    target_concept: Optional[str] = None  # 目标概念(如果有)

    # 推理结果
    primary_path: Optional[CausalPath] = None  # 主要因果路径
    alternative_paths: List[CausalPath] = field(default_factory=list)  # 备选路径
    related_concepts: List[Tuple[str, float]] = field(default_factory=list)  # 相关概念

    # 自然语言输出
    explanation: str = ""  # 解释文本
    keywords: List[str] = field(default_factory=list)  # 关键词列表(给表达中枢用)
    confidence: float = 0.0  # 整体置信度

    def __repr__(self):
        return f"ReasoningResult(type={self.reasoning_type.value}, success={self.success}, conf={self.confidence:.2f})"


class CausalReasoningEngine:
    """
    🧠 因果推理引擎

    核心职责：
    1. 检测用户问题的类型 (WHY/HOW/WHAT_IF/PREDICT)
    2. 在因果图上搜索相关路径
    3. 综合多条路径生成解释
    4. 输出结构化的推理结果
    """

    # 问题类型关键词
    WHY_KEYWORDS = {"为什么", "为啥", "怎么会", "何以", "缘何", "原因", "为何"}
    HOW_KEYWORDS = {"怎么", "如何", "怎样", "怎么样", "怎么才能", "怎样才能", "方法", "办法"}
    WHAT_IF_KEYWORDS = {"如果", "假如", "要是", "倘若", "万一", "假设"}
    PREDICT_KEYWORDS = {"会怎样", "会怎么", "会导致", "会引起", "会造成", "后果", "结果"}

    def __init__(self, model, idx2word: dict, word2idx: dict):
        """
        Args:
            model: CognitiveGraphModel 实例
            idx2word: 索引到词的映射
            word2idx: 词到索引的映射
        """
        self.model = model
        self.idx2word = idx2word
        self.word2idx = word2idx

        # 推理参数
        self.max_search_depth = 6  # 最大搜索深度
        self.min_causal_strength = 0.1  # 最小因果强度阈值
        self.max_paths = 5  # 最多返回的路径数
        self.beam_width = 10  # 束搜索宽度

    def detect_question_type(self, text: str) -> Tuple[ReasoningType, List[str]]:
        """
        检测问题类型并提取关键概念

        Returns:
            (问题类型, 关键概念列表)
        """
        text_lower = text.lower()

        # 检测问题类型
        if any(kw in text for kw in self.WHY_KEYWORDS):
            q_type = ReasoningType.WHY
        elif any(kw in text for kw in self.WHAT_IF_KEYWORDS):
            q_type = ReasoningType.WHAT_IF
        elif any(kw in text for kw in self.PREDICT_KEYWORDS):
            q_type = ReasoningType.PREDICT
        elif any(kw in text for kw in self.HOW_KEYWORDS):
            q_type = ReasoningType.HOW
        else:
            q_type = ReasoningType.NONE

        # 提取关键概念 (简化版: 移除问题词后的剩余词)
        # 实际应该用分词，这里调用者会传入已分词的indices
        concepts = []
        for kw_set in [self.WHY_KEYWORDS, self.HOW_KEYWORDS,
                       self.WHAT_IF_KEYWORDS, self.PREDICT_KEYWORDS]:
            for kw in kw_set:
                text = text.replace(kw, "")

        # 清理后返回
        text = text.strip()
        if text:
            concepts.append(text)

        return q_type, concepts

    def reason(self,
               query_indices: List[int],
               query_text: str = "",
               reasoning_type: ReasoningType = None) -> ReasoningResult:
        """
        主推理入口

        Args:
            query_indices: 查询概念的索引列表
            query_text: 原始查询文本 (用于问题类型检测)
            reasoning_type: 强制指定推理类型 (可选)

        Returns:
            ReasoningResult
        """
        # 1. 检测问题类型
        if reasoning_type is None:
            reasoning_type, _ = self.detect_question_type(query_text)

        # 过滤有效索引
        valid_indices = [idx for idx in query_indices if idx > 1]
        if not valid_indices:
            return ReasoningResult(
                success=False,
                reasoning_type=reasoning_type,
                query_concept="",
                explanation="无法识别问题中的概念"
            )

        # 取最后一个概念作为主要查询对象 (通常是核心概念)
        main_idx = valid_indices[-1]
        main_word = self.idx2word.get(main_idx, "")

        # 2. 根据类型执行不同推理
        if reasoning_type == ReasoningType.WHY:
            return self._reason_why(main_idx, main_word, valid_indices)
        elif reasoning_type == ReasoningType.HOW:
            return self._reason_how(main_idx, main_word, valid_indices)
        elif reasoning_type == ReasoningType.WHAT_IF:
            return self._reason_what_if(main_idx, main_word, valid_indices)
        elif reasoning_type == ReasoningType.PREDICT:
            return self._reason_predict(main_idx, main_word, valid_indices)
        else:
            # 非因果问题，返回相关联想
            return self._reason_associate(main_idx, main_word, valid_indices)

    def _reason_why(self, concept_idx: int, concept_word: str,
                    context_indices: List[int]) -> ReasoningResult:
        """
        回答"为什么"类问题
        搜索导致该概念的原因链
        """
        # 逆向搜索: 找导致 concept 的原因
        antecedents = self.get_causal_antecedents(concept_idx, max_depth=self.max_search_depth)

        if not antecedents:
            # 尝试用联想通道
            related = self.get_associated_concepts(concept_idx, top_k=5)
            return ReasoningResult(
                success=False,
                reasoning_type=ReasoningType.WHY,
                query_concept=concept_word,
                related_concepts=related,
                explanation=f"我还不知道为什么会{concept_word}",
                keywords=[concept_word] + [w for w, _ in related[:3]],
                confidence=0.2
            )

        # 构建解释
        paths = antecedents[:self.max_paths]
        primary_path = paths[0] if paths else None

        # 生成自然语言解释
        explanation = self._generate_why_explanation(concept_word, paths)

        # 提取关键词
        keywords = self._extract_path_keywords(paths)

        # 计算置信度
        confidence = primary_path.total_confidence if primary_path else 0.0

        return ReasoningResult(
            success=True,
            reasoning_type=ReasoningType.WHY,
            query_concept=concept_word,
            primary_path=primary_path,
            alternative_paths=paths[1:],
            explanation=explanation,
            keywords=keywords,
            confidence=confidence
        )

    def _reason_how(self, goal_idx: int, goal_word: str,
                    context_indices: List[int]) -> ReasoningResult:
        """
        回答"如何/怎样"类问题
        逆向搜索达成目标的方法
        """
        # 逆向搜索: 什么能导致 goal
        antecedents = self.get_causal_antecedents(goal_idx, max_depth=self.max_search_depth)

        if not antecedents:
            related = self.get_associated_concepts(goal_idx, top_k=5)
            return ReasoningResult(
                success=False,
                reasoning_type=ReasoningType.HOW,
                query_concept=goal_word,
                related_concepts=related,
                explanation=f"我还不知道如何{goal_word}",
                keywords=[goal_word] + [w for w, _ in related[:3]],
                confidence=0.2
            )

        paths = antecedents[:self.max_paths]
        primary_path = paths[0] if paths else None

        explanation = self._generate_how_explanation(goal_word, paths)
        keywords = self._extract_path_keywords(paths)
        confidence = primary_path.total_confidence if primary_path else 0.0

        return ReasoningResult(
            success=True,
            reasoning_type=ReasoningType.HOW,
            query_concept=goal_word,
            primary_path=primary_path,
            alternative_paths=paths[1:],
            explanation=explanation,
            keywords=keywords,
            confidence=confidence
        )

    def _reason_what_if(self, condition_idx: int, condition_word: str,
                        context_indices: List[int]) -> ReasoningResult:
        """
        回答"如果...会怎样"类问题
        正向搜索条件的后果
        """
        # 正向搜索: condition 会导致什么
        effects = self.get_causal_effects(condition_idx, max_depth=self.max_search_depth)

        if not effects:
            related = self.get_associated_concepts(condition_idx, top_k=5)
            return ReasoningResult(
                success=False,
                reasoning_type=ReasoningType.WHAT_IF,
                query_concept=condition_word,
                related_concepts=related,
                explanation=f"我还不确定{condition_word}会导致什么",
                keywords=[condition_word] + [w for w, _ in related[:3]],
                confidence=0.2
            )

        paths = effects[:self.max_paths]
        primary_path = paths[0] if paths else None

        explanation = self._generate_what_if_explanation(condition_word, paths)
        keywords = self._extract_path_keywords(paths)
        confidence = primary_path.total_confidence if primary_path else 0.0

        return ReasoningResult(
            success=True,
            reasoning_type=ReasoningType.WHAT_IF,
            query_concept=condition_word,
            primary_path=primary_path,
            alternative_paths=paths[1:],
            explanation=explanation,
            keywords=keywords,
            confidence=confidence
        )

    def _reason_predict(self, action_idx: int, action_word: str,
                        context_indices: List[int]) -> ReasoningResult:
        """
        预测行为的后果
        """
        # 与 what_if 类似，但更强调最终结果
        return self._reason_what_if(action_idx, action_word, context_indices)

    def _reason_associate(self, concept_idx: int, concept_word: str,
                          context_indices: List[int]) -> ReasoningResult:
        """
        非因果问题，返回联想结果
        """
        related = self.get_associated_concepts(concept_idx, top_k=10)

        return ReasoningResult(
            success=True,
            reasoning_type=ReasoningType.NONE,
            query_concept=concept_word,
            related_concepts=related,
            explanation="",
            keywords=[concept_word] + [w for w, _ in related[:5]],
            confidence=0.5
        )

    # =========================================================
    # 🔍 图搜索算法
    # =========================================================

    def get_causal_effects(self, start_idx: int, max_depth: int = 5) -> List[CausalPath]:
        """
        获取概念的因果后果 (正向搜索)
        使用 BFS + 束搜索

        Args:
            start_idx: 起始概念索引
            max_depth: 最大搜索深度

        Returns:
            按置信度排序的因果路径列表
        """
        if start_idx <= 1:
            return []

        cause_matrix = self.model.synapse_tensor[CHANNEL_CAUSES]
        device = cause_matrix.device

        # 使用优先队列进行束搜索 (置信度越高优先级越高)
        # (负置信度, 路径)
        start_word = self.idx2word.get(start_idx, "")
        initial_path = CausalPath(links=[], total_confidence=1.0)

        # 优先队列: (-confidence, path_id, current_idx, path)
        # path_id 用于打破置信度相同时的顺序
        pq = [(-1.0, 0, start_idx, initial_path)]
        path_counter = 1

        visited_states = set()  # (current_idx, frozenset(path_indices))
        result_paths = []

        while pq and len(result_paths) < self.max_paths * 2:
            neg_conf, _, current_idx, current_path = heapq.heappop(pq)
            current_conf = -neg_conf

            # 获取路径上已访问的节点
            path_indices = frozenset(link.target for link in current_path.links)
            state = (current_idx, path_indices)

            if state in visited_states:
                continue
            visited_states.add(state)

            # 获取当前节点的因果后果
            effects = cause_matrix[current_idx]

            # 找到强度超过阈值的后果
            strong_effects = (effects > self.min_causal_strength).nonzero(as_tuple=True)[0]

            if len(strong_effects) == 0 and current_path.length > 0:
                # 到达终点，保存路径
                result_paths.append(current_path)
                continue

            # 扩展路径
            for effect_idx in strong_effects.tolist():
                if effect_idx <= 1:  # 跳过 PAD, UNK
                    continue
                if effect_idx in path_indices:  # 避免环
                    continue
                if effect_idx == start_idx:  # 避免回到起点
                    continue

                effect_word = self.idx2word.get(effect_idx, "")
                strength = effects[effect_idx].item()

                # 创建新链接
                source_word = self.idx2word.get(current_idx, "")
                new_link = CausalLink(
                    source=current_idx,
                    target=effect_idx,
                    source_word=source_word,
                    target_word=effect_word,
                    strength=strength,
                    channel=CHANNEL_CAUSES
                )

                # 创建新路径
                new_path = CausalPath(
                    links=current_path.links + [new_link],
                    total_confidence=current_conf * min(strength, 1.0)
                )

                # 如果达到深度限制，保存路径
                if new_path.length >= max_depth:
                    result_paths.append(new_path)
                else:
                    # 加入优先队列
                    heapq.heappush(pq, (-new_path.total_confidence, path_counter, effect_idx, new_path))
                    path_counter += 1

        # 按置信度排序
        result_paths.sort(key=lambda p: p.total_confidence, reverse=True)

        return result_paths[:self.max_paths]

    def get_causal_antecedents(self, end_idx: int, max_depth: int = 5) -> List[CausalPath]:
        """
        获取概念的因果前因 (逆向搜索)
        使用转置的因果矩阵进行搜索

        Args:
            end_idx: 目标概念索引
            max_depth: 最大搜索深度

        Returns:
            按置信度排序的因果路径列表 (方向: 原因 → 结果)
        """
        if end_idx <= 1:
            return []

        # 转置因果矩阵: 从"A导致B"变成"B被A导致"
        cause_matrix = self.model.synapse_tensor[CHANNEL_CAUSES]
        reverse_matrix = cause_matrix.T  # 转置

        end_word = self.idx2word.get(end_idx, "")
        initial_path = CausalPath(links=[], total_confidence=1.0)

        pq = [(-1.0, 0, end_idx, initial_path)]
        path_counter = 1

        visited_states = set()
        result_paths = []

        while pq and len(result_paths) < self.max_paths * 2:
            neg_conf, _, current_idx, current_path = heapq.heappop(pq)
            current_conf = -neg_conf

            path_indices = frozenset(link.source for link in current_path.links)
            state = (current_idx, path_indices)

            if state in visited_states:
                continue
            visited_states.add(state)

            # 获取导致当前节点的原因
            antecedents = reverse_matrix[current_idx]
            strong_antecedents = (antecedents > self.min_causal_strength).nonzero(as_tuple=True)[0]

            if len(strong_antecedents) == 0 and current_path.length > 0:
                # 到达起点，保存路径 (需要反转)
                reversed_path = self._reverse_path(current_path)
                result_paths.append(reversed_path)
                continue

            for ante_idx in strong_antecedents.tolist():
                if ante_idx <= 1:
                    continue
                if ante_idx in path_indices:
                    continue
                if ante_idx == end_idx:
                    continue

                ante_word = self.idx2word.get(ante_idx, "")
                strength = antecedents[ante_idx].item()

                current_word = self.idx2word.get(current_idx, "")
                new_link = CausalLink(
                    source=ante_idx,
                    target=current_idx,
                    source_word=ante_word,
                    target_word=current_word,
                    strength=strength,
                    channel=CHANNEL_CAUSES
                )

                new_path = CausalPath(
                    links=[new_link] + current_path.links,
                    total_confidence=current_conf * min(strength, 1.0)
                )

                if new_path.length >= max_depth:
                    reversed_path = self._reverse_path(new_path)
                    result_paths.append(reversed_path)
                else:
                    heapq.heappush(pq, (-new_path.total_confidence, path_counter, ante_idx, new_path))
                    path_counter += 1

        result_paths.sort(key=lambda p: p.total_confidence, reverse=True)

        return result_paths[:self.max_paths]

    def search_causal_path(self, start_idx: int, end_idx: int,
                           max_depth: int = 6) -> Optional[CausalPath]:
        """
        搜索两个概念之间的因果路径 (A* 搜索)

        Args:
            start_idx: 起始概念索引
            end_idx: 目标概念索引
            max_depth: 最大搜索深度

        Returns:
            找到的因果路径，或 None
        """
        if start_idx <= 1 or end_idx <= 1:
            return None
        if start_idx == end_idx:
            return None

        cause_matrix = self.model.synapse_tensor[CHANNEL_CAUSES]

        # A* 搜索: f(n) = g(n) + h(n)
        # g(n) = 负对数置信度 (累积代价)
        # h(n) = 启发式 (这里简化为0，退化为Dijkstra)

        start_word = self.idx2word.get(start_idx, "")
        end_word = self.idx2word.get(end_idx, "")

        # (cost, counter, current_idx, path)
        pq = [(0.0, 0, start_idx, [])]
        path_counter = 1
        visited = set()

        while pq:
            cost, _, current_idx, path = heapq.heappop(pq)

            if current_idx == end_idx:
                # 找到了！
                return CausalPath(
                    links=path,
                    total_confidence=self._cost_to_confidence(cost)
                )

            if current_idx in visited:
                continue
            visited.add(current_idx)

            if len(path) >= max_depth:
                continue

            effects = cause_matrix[current_idx]
            strong_effects = (effects > self.min_causal_strength).nonzero(as_tuple=True)[0]

            for effect_idx in strong_effects.tolist():
                if effect_idx <= 1 or effect_idx in visited:
                    continue

                strength = effects[effect_idx].item()
                edge_cost = -torch.log(torch.tensor(min(strength, 0.999))).item()

                current_word = self.idx2word.get(current_idx, "")
                effect_word = self.idx2word.get(effect_idx, "")

                new_link = CausalLink(
                    source=current_idx,
                    target=effect_idx,
                    source_word=current_word,
                    target_word=effect_word,
                    strength=strength,
                    channel=CHANNEL_CAUSES
                )

                new_cost = cost + edge_cost
                new_path = path + [new_link]

                heapq.heappush(pq, (new_cost, path_counter, effect_idx, new_path))
                path_counter += 1

        return None

    def get_associated_concepts(self, concept_idx: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        获取关联概念 (从 ASSOCIATED 通道)

        Returns:
            [(word, strength), ...]
        """
        if concept_idx <= 1:
            return []

        assoc_matrix = self.model.synapse_tensor[CHANNEL_ASSOCIATED]
        weights = assoc_matrix[concept_idx]

        # 获取 top-k
        values, indices = torch.topk(weights, min(top_k + 2, len(weights)))

        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            if idx <= 1:  # 跳过 PAD, UNK
                continue
            if idx == concept_idx:  # 跳过自己
                continue
            word = self.idx2word.get(idx, "")
            if word:
                results.append((word, val))

        return results[:top_k]

    # =========================================================
    # 📝 自然语言生成
    # =========================================================

    def _generate_why_explanation(self, concept: str, paths: List[CausalPath]) -> str:
        """生成"为什么"的解释"""
        if not paths:
            return f"我还不知道为什么会{concept}"

        explanations = []

        for i, path in enumerate(paths[:3]):
            if path.length == 0:
                continue

            words = path.get_words()
            if len(words) >= 2:
                cause_chain = " → ".join(words)

                if i == 0:
                    explanations.append(f"因为 {words[0]}，所以导致了 {concept}")
                    if len(words) > 2:
                        explanations.append(f"（完整因果链：{cause_chain}）")
                else:
                    explanations.append(f"另外，{words[0]} 也可能导致 {concept}")

        return "。".join(explanations) if explanations else f"我还不太清楚为什么会{concept}"

    def _generate_how_explanation(self, goal: str, paths: List[CausalPath]) -> str:
        """生成"如何"的解释"""
        if not paths:
            return f"我还不知道如何{goal}"

        methods = []

        for path in paths[:3]:
            if path.length == 0:
                continue

            words = path.get_words()
            if words:
                # 取因果链的起点作为方法
                method = words[0]
                methods.append(method)

        if not methods:
            return f"我还不知道如何{goal}"

        if len(methods) == 1:
            return f"要{goal}的话，可以试试{methods[0]}"
        else:
            method_str = "、".join(methods[:-1]) + f"或者{methods[-1]}"
            return f"要{goal}的话，可以试试{method_str}"

    def _generate_what_if_explanation(self, condition: str, paths: List[CausalPath]) -> str:
        """生成"如果"的解释"""
        if not paths:
            return f"我还不确定{condition}会导致什么"

        effects = []

        for path in paths[:3]:
            if path.length == 0:
                continue

            words = path.get_words()
            if len(words) >= 2:
                # 取因果链的终点作为结果
                effect = words[-1]
                effects.append(effect)

        if not effects:
            return f"我还不确定{condition}会导致什么"

        if len(effects) == 1:
            return f"如果{condition}的话，可能会导致{effects[0]}"
        else:
            effect_str = "、".join(effects[:-1]) + f"甚至{effects[-1]}"
            return f"如果{condition}的话，可能会导致{effect_str}"

    def _extract_path_keywords(self, paths: List[CausalPath]) -> List[str]:
        """从路径中提取关键词"""
        keywords = []
        seen = set()

        for path in paths:
            for word in path.get_words():
                if word and word not in seen:
                    keywords.append(word)
                    seen.add(word)

        return keywords[:10]

    def _reverse_path(self, path: CausalPath) -> CausalPath:
        """反转路径方向"""
        # 对于逆向搜索，需要反转链接的方向
        return CausalPath(
            links=path.links,  # 链接在搜索时已经是正确方向
            total_confidence=path.total_confidence
        )

    def _cost_to_confidence(self, cost: float) -> float:
        """将代价转换为置信度"""
        import math
        return math.exp(-cost)

    # =========================================================
    # 🔧 辅助方法
    # =========================================================

    def get_causal_strength(self, source_idx: int, target_idx: int) -> float:
        """获取两个概念之间的直接因果强度"""
        if source_idx <= 1 or target_idx <= 1:
            return 0.0

        return self.model.synapse_tensor[CHANNEL_CAUSES, source_idx, target_idx].item()

    def get_causal_stats(self) -> Dict:
        """获取因果图统计信息"""
        cause_matrix = self.model.synapse_tensor[CHANNEL_CAUSES]

        # 非零连接数
        nonzero = (cause_matrix.abs() > self.min_causal_strength).sum().item()
        total = cause_matrix.numel()

        # 平均强度
        mask = cause_matrix.abs() > self.min_causal_strength
        if mask.any():
            avg_strength = cause_matrix[mask].mean().item()
        else:
            avg_strength = 0.0

        # 最强连接
        max_strength = cause_matrix.max().item()
        max_idx = cause_matrix.argmax().item()
        max_source = max_idx // cause_matrix.shape[1]
        max_target = max_idx % cause_matrix.shape[1]

        return {
            "total_connections": nonzero,
            "density": nonzero / total,
            "avg_strength": avg_strength,
            "max_strength": max_strength,
            "strongest_link": (
                self.idx2word.get(max_source, "?"),
                self.idx2word.get(max_target, "?")
            )
        }
