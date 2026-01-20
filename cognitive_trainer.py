import torch
from .cognitive_graph_model import CHANNEL_IS_A, CHANNEL_HAS_PROP, CHANNEL_CAUSES, CHANNEL_ASSOCIATED, CHANNEL_REPRESENTS

class HebbianTrainer:
    """
    双模态赫布训练器 (Dual-Mode Hebbian Trainer)
    
    支持两种学习模式：
    1. 联想学习 (Associative Learning): 基于共现统计，模拟潜意识/直觉。
    2. 逻辑植入 (Logical Imprinting): 基于三元组，进行精准的外科手术式连接。
    """

    def __init__(self, model, learning_rate=0.1):
        self.model = model
        self.lr = learning_rate
        self.model.train()

    def train_step_associative(self, sentence_indices):
        """
        [模式1: 联想学习]
        用于处理非结构化的闲聊文本。只更新 CHANNEL_ASSOCIATED (直觉通道)。
        保留原有的"距离衰减"和"能量归一化"逻辑。
        """
        seq_len = len(sentence_indices)
        if seq_len < 2:
            self.model.learn_from_input(torch.tensor(sentence_indices, device=self.model.synapse_tensor.device))
            return 0.0

        # 获取直觉通道的切片 (引用)
        W = self.model.synapse_tensor[CHANNEL_ASSOCIATED]
        
        # 获取惊讶度权重
        with torch.no_grad():
            indices_tensor = torch.tensor(sentence_indices, device=W.device)
            saliency_weights = self.model.get_attention_weights(indices_tensor)

        # 赫布更新循环
        with torch.no_grad():
            for i in range(seq_len):
                idx_i = sentence_indices[i]
                w_i = saliency_weights[i].item()

                for j in range(seq_len):
                    if i == j: continue

                    idx_j = sentence_indices[j]
                    w_j = saliency_weights[j].item()

                    distance = abs(i - j)
                    decay = 1.0 / float(distance) # 距离越远关系越弱

                    # 只有惊讶度高的词之间才会建立强连接
                    delta = self.lr * w_i * w_j * decay
                    
                    W[idx_i, idx_j] += delta
                    # 注意：联想通常是双向的，但也可能是不对称的，这里保持对称简化
                    W[idx_j, idx_i] += delta

            # 归一化 (仅针对直觉通道)
            self._normalize_synapse(W)

        # 别忘了更新词频统计
        self.model.learn_from_input(indices_tensor)
        return W.mean().item()

    def train_step_logical(self, triplets):
        """
        [模式2: 逻辑植入]
        用于处理结构化的三元组列表。
        triplets: list of (head_idx, relation_type, tail_idx)
        
        relation_type 映射:
        'is_a' -> CHANNEL_IS_A
        'has' -> CHANNEL_HAS_PROP
        'cause' -> CHANNEL_CAUSES
        """
        if not triplets: return 0.0
        
        cnt = 0
        with torch.no_grad():
            for head, rel, tail in triplets:
                # 映射关系类型到通道ID
                channel_id = -1
                if rel == 'is_a': channel_id = CHANNEL_IS_A
                elif rel == 'has': channel_id = CHANNEL_HAS_PROP
                elif rel == 'cause': channel_id = CHANNEL_CAUSES
                elif rel == 'represent': channel_id = CHANNEL_REPRESENTS
                
                if channel_id == -1: continue # 未知关系忽略
                
                # 获取对应通道的矩阵
                W = self.model.synapse_tensor[channel_id]
                
                # 逻辑注入是强力的、定向的
                # 不需要像联想那样根据词频打折，因为这是逻辑公理
                delta = 1.0  # 强连接
                
                W[head, tail] += delta
                # 注意：逻辑通常是有向的！(A是B != B是A)，所以不更新反向
                
                # 也可以顺便微弱更新一下联想通道，毕竟逻辑也是一种联想
                self.model.synapse_tensor[CHANNEL_ASSOCIATED, head, tail] += 0.1
                
                cnt += 1
            
            # 对所有受影响的通道进行归一化，防止数值爆炸
            for c in range(self.model.num_channels):
                self._normalize_synapse(self.model.synapse_tensor[c])
                
        return float(cnt)

    def _normalize_synapse(self, W):
        """
        突触软限制 - 只压制过强的连接，不破坏弱连接
        """
        with torch.no_grad():
            # 只钳位最大值，不做全局归一化
            W.data.clamp_(-10.0, 10.0)

            # 可选：对每行能量过大的进行软缩放
            row_sums = W.data.abs().sum(dim=1, keepdim=True)
            max_energy = 50.0  # 每行最大允许能量

            # 只缩放超标的行
            scale = torch.where(
                row_sums > max_energy,
                max_energy / row_sums,
                torch.ones_like(row_sums)
            )
            W.data.mul_(scale)

