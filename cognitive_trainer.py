import torch


class HebbianTrainer:
    """
    赫布学习训练器 (Hebbian Trainer)

    核心哲学：
    1. Fire together, wire together (共现即连接)。
    2. Synaptic Normalization (能量守恒)：一个神经元的连接越多，单条连接越弱。
       这自然解决了"常用词(如'是')权重过大"的问题。
    """

    def __init__(self, model, learning_rate=0.1):
        self.model = model
        self.lr = learning_rate

        # 确保模型处于训练模式 (开启 Dropout 等，如果有的话)
        self.model.train()

    def train_step(self, sentence_indices):
        """
        单步训练：输入一句话的索引列表，更新突触矩阵。
        input: sentence_indices (list or tensor) e.g., [2, 5, 8]
        """
        # 0. 基础检查
        seq_len = len(sentence_indices)
        if seq_len < 2:
            return 0.0  # 只有一个词无法建立连接

        # 1. 准备数据
        # 获取当前突触矩阵 (引用，直接修改它)
        W = self.model.synapse_matrix

        # 获取这句话中每个词的"动态惊讶度" (Dynamic Saliency)
        # 罕见词 (High Saliency) 应该拥有更强的塑造突触的能力
        with torch.no_grad():
            indices_tensor = torch.tensor(sentence_indices, device=W.device)
            # 获取权重 [Seq_Len]
            saliency_weights = self.model.get_attention_weights(indices_tensor)

        # 2. 赫布更新 (Hebbian Update Loop)
        # 遍历句子中的每一对词 (i, j)
        # 注意：这里使用 no_grad，因为我们是手动修改权重，不走反向传播
        with torch.no_grad():
            for i in range(seq_len):
                idx_i = sentence_indices[i]
                w_i = saliency_weights[i].item()  # 词 i 的重要性

                for j in range(seq_len):
                    if i == j:
                        continue  # 不处理自连接

                    idx_j = sentence_indices[j]
                    w_j = saliency_weights[j].item()  # 词 j 的重要性

                    # [距离衰减]
                    # 离得越远，关系越弱。相邻词(dist=1)最强。
                    distance = abs(i - j)
                    decay = 1.0 / float(distance)

                    # [更新公式]
                    # ΔW = 学习率 * 强度i * 强度j * 距离衰减
                    # 两个重要的词靠在一起 -> 连接大幅增强
                    # 两个不重要的词离得远 -> 连接微弱增强
                    delta = self.lr * w_i * w_j * decay

                    # 双向强化 (无向图逻辑，或者理解为联想是双向的)
                    W[idx_i, idx_j] += delta
                    W[idx_j, idx_i] += delta

            # 3. 突触竞争归一化 (Synaptic Normalization) - 核心修正
            # 这里是为了防止"常用词" (Hubs) 垄断能量。

            # A. 钳位 (Clamping): 防止单个权重爆炸
            W.data = torch.clamp(W.data, max=10.0)

            # B. 归一化 (Normalization):
            # 资源的稀缺性。每个词发出的总能量被限制。
            # "是"连了100个词 -> 每个连接分到 1/100
            # "女王"连了3个词 -> 每个连接分到 1/3 (胜出!)
            row_sums = W.data.sum(dim=1, keepdim=True) + 1e-6
            W.data = W.data / row_sums

            # C. 能量回升 (Rescaling):
            # 归一化后数值太小 (0.00x)，需要乘一个系数让它适合 Softmax
            W.data *= 5.0

        # 4. 更新经验统计 (Experience Update)
        # 这一步必须做，否则下一次计算 get_attention_weights 时就没有依据
        self.model.learn_from_input(indices_tensor)

        return W.mean().item()
