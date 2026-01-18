import torch
import torch.nn as nn
import math


class CognitiveGraphModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # 1. 静态基因：词向量 (L0 感知层)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # 2. 动态突触：图连接矩阵 (可学习参数)
        # 初始化为单位矩阵 + 微弱噪音
        # Trainer 会负责对这个矩阵进行"赫布更新"和"能量归一化"
        self.synapse_matrix = nn.Parameter(
            torch.eye(vocab_size) + torch.randn(vocab_size, vocab_size) * 0.01
        )

        # 3. 动态经验：词频统计 (记忆 Buffer)
        # 使用 register_buffer 确保这些统计数据随模型保存，但不需要梯度下降
        self.register_buffer("word_counts", torch.ones(vocab_size))
        self.register_buffer("total_experience", torch.tensor(float(vocab_size)))

    def get_attention_weights(self, input_indices):
        """
        计算"惊讶度"权重 (Dynamic Saliency)。
        基于韦伯-费希纳定律：越罕见的东西，刺激强度越大。
        """
        counts = self.word_counts[input_indices]
        total = self.total_experience

        # IDF 变体公式
        weights = torch.log(total + 1) / (torch.log(counts + 1) + 1e-6)

        # 归一化限制 (0.1 ~ 3.0)
        weights = torch.clamp(weights * 0.5, min=0.1, max=3.0)

        return weights

    def learn_from_input(self, input_indices):
        """
        [在线学习] 更新经验统计
        """
        # 展平并更新计数
        flat_indices = input_indices.view(-1)
        for idx in flat_indices:
            self.word_counts[idx] += 1
            self.total_experience += 1

    def forward(self, input_indices, steps=3):
        """
        前向传播：输入词索引 -> 激活思维图谱 -> 能量扩散
        """
        batch_size, seq_len = input_indices.shape

        # 1. 计算动态权重
        attn_weights = self.get_attention_weights(input_indices)

        # 2. 注入能量 (Injection)
        # 将输入的词在全词表空间点亮
        current_thought = torch.zeros(
            batch_size, self.vocab_size, device=input_indices.device
        )

        # 把权重值作为能量注入
        src = attn_weights
        current_thought.scatter_add_(1, input_indices, src)

        # 3. 思维扩散 (Diffusion)
        # 让能量沿着突触矩阵游走
        for _ in range(steps):
            current_thought = torch.matmul(current_thought, self.synapse_matrix)
            current_thought = torch.relu(current_thought)  # 激活阈值

        # 4. 顺便学习 (训练模式下自动更新统计)
        if self.training:
            self.learn_from_input(input_indices)

        return current_thought

    def generate_reply(self, input_indices, max_len=20):
        """
        [生成模块] 概率能量采样 + 返回抑制 (Inhibition of Return)
        """
        self.eval()  # 确保生成时不更新统计

        # 1. 产生意念 (Thought Map)
        with torch.no_grad():
            thought_energy = self(input_indices, steps=3)

        # 2. 初始状态设置
        # [核心补丁] 返回抑制初始化：
        # 将输入问题里的词直接加入"已访问"，强迫模型向外延展，而不是复读问题
        visited = set(input_indices[0].tolist())

        # 3. 选取起点 (Seed Selection)
        # 我们要避开已经问过的词
        start_energy = thought_energy[0].clone()
        for v in visited:
            start_energy[v] = -float("inf")  # 屏蔽输入词

        # 如果屏蔽后没词了(极罕见)，就随便选一个
        if torch.max(start_energy) == -float("inf"):
            probs = torch.ones_like(start_energy)
        else:
            probs = torch.softmax(start_energy * 2.0, dim=0)

        current_idx = torch.multinomial(probs, 1).item()

        reply_indices = [current_idx]
        visited.add(current_idx)  # 标记起点已访问

        print(f"🗣️ [生成启动] 避开原词，新想法 ID: {current_idx}")

        # 4. 路径游走 (Path Walking)
        for _ in range(max_len):
            # 获取当前节点连向其他节点的权重
            next_step_weights = self.synapse_matrix[current_idx].clone()

            # A. 施加返回抑制 (Inhibition of Return)
            # 走过的路即使连接再强，也暂时封死，逼迫寻找新路
            for v in visited:
                next_step_weights[v] = -float("inf")

            # B. 施加意念场引导 (Context Guidance)
            # 混合 "局部连接" 和 "全局语境"
            # 0.5 是引导系数：既要顺着路走，又要不忘初心的语境
            guidance = thought_energy[0] * 0.5
            combined_weights = next_step_weights + guidance

            # C. 采样下一个词
            if torch.max(combined_weights) == -float("inf"):
                break  # 无路可走(死胡同)

            next_probs = torch.softmax(
                combined_weights * 3.0, dim=0
            )  # Temp=3.0 增加确定性
            next_idx = torch.multinomial(next_probs, 1).item()

            reply_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        return reply_indices
