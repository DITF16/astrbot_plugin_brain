import torch
import torch.nn as nn
import math

# === 通道定义常量 ===
CHANNEL_IS_A = 0       # 定义: A 是 B
CHANNEL_HAS_PROP = 1   # 属性: A 有 B
CHANNEL_CAUSES = 2     # 因果: A 导致 B
CHANNEL_ASSOCIATED = 3 # 直觉: A 联想 B (保留旧有逻辑)
CHANNEL_REPRESENTS = 4 # 表征: A 对应 B (多模态锚点)
NUM_CHANNELS = 5

class CognitiveGraphModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_channels = NUM_CHANNELS

        # 1. 静态基因：词向量 (L0 感知层)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # 2. 动态突触：三维图连接张量 [Channels, N, N]
        initial_tensor = torch.zeros(NUM_CHANNELS, vocab_size, vocab_size)
        
        # 初始化直觉通道
        initial_tensor[CHANNEL_ASSOCIATED] = torch.eye(vocab_size) + torch.randn(vocab_size, vocab_size) * 0.01
        
        self.synapse_tensor = nn.Parameter(initial_tensor)

        # 3. 动态经验：词频统计 (记忆 Buffer)
        self.register_buffer("word_counts", torch.ones(vocab_size))
        self.register_buffer("total_experience", torch.tensor(float(vocab_size)))
        
        # 4. 情绪调节器
        self.register_buffer("mood_bias", torch.zeros(vocab_size)) 

    def get_attention_weights(self, input_indices):
        counts = self.word_counts[input_indices]
        total = self.total_experience
        weights = torch.log(total + 1) / (torch.log(counts + 1) + 1e-6)
        weights = torch.clamp(weights * 0.5, min=0.1, max=3.0)
        return weights

    def learn_from_input(self, input_indices):
        flat_indices = input_indices.view(-1)
        for idx in flat_indices:
            self.word_counts[idx] += 1
            self.total_experience += 1

    def reinforce(self, indices, reward_sign):
        if len(indices) < 2: return
        with torch.no_grad():
            for i in range(len(indices) - 1):
                u, v = indices[i], indices[i+1]
                self.synapse_tensor[CHANNEL_ASSOCIATED, u, v] += (reward_sign * 0.5)
                if reward_sign > 0:
                    self.mood_bias[v] += 0.1
                else:
                    self.mood_bias[v] -= 0.1

    def process_sleep_cycle(self, pruning_threshold=0.05):
        """
        [V3.1] 睡眠机制：压力驱动 + 晶体化保护 + 专注度门控预算 + 对比度增强
        """
        with torch.no_grad():
            # === 1. 压力检测 (Capacity Pressure) ===
            target_capacity = float(self.vocab_size) * 10.0
            current_energy = self.synapse_tensor.abs().sum()
            
            if current_energy > target_capacity:
                pressure_ratio = target_capacity / current_energy
                global_decay = max(pressure_ratio, 0.8) 
            else:
                global_decay = 1.0
            
            # === 2. 晶体化保护 (Crystallization) ===
            weights_abs = self.synapse_tensor.abs()
            shield = torch.sigmoid((weights_abs - 0.5) * 5.0)
            final_decay = global_decay * (1.0 - shield) + 1.0 * shield
            self.synapse_tensor *= final_decay

            # === 3. 对比度增强 (Contrast Enhancement) ===
            self.synapse_tensor = torch.sign(self.synapse_tensor) * torch.pow(self.synapse_tensor.abs(), 1.1)
            
            # === 4. 动态能量守恒 (带专注度门控) ===
            # [修正] 防止"了"、"，"等高频停用词获得过高预算。
            
            # 1. 计算行能量 (Sum)
            row_sums = self.synapse_tensor.abs().sum(dim=2, keepdim=True) + 1e-6
            
            # 2. 计算行峰值 (Max)
            # dim=2 是目标节点维度。max[0] 返回 values
            row_maxs = self.synapse_tensor.abs().max(dim=2, keepdim=True)[0]
            
            # 3. 计算稀疏度 (Sparsity = Max / Sum)
            # 停用词(扁平分布): Sparsity -> 1/N (很小)
            # 核心词(尖峰分布): Sparsity -> 0.5~1.0 (很大)
            sparsity = row_maxs / row_sums
            
            # 4. 计算门控系数 (Gate)
            # 使用 Tanh 放大差异。 Sparsity=0.1 -> Gate=0.46; Sparsity=0.01 -> Gate=0.05
            gate = torch.tanh(sparsity * 5.0) 
            
            # 5. 计算动态预算
            counts = self.word_counts
            base_limit = 10.0
            freq_bonus = torch.log1p(counts) * 2.0
            
            # 核心修正：红利必须乘以门控系数！
            # 注意维度广播: freq_bonus [Vocab] -> [1, Vocab, 1]
            # gate [Channels, Vocab, 1] (其实我们希望针对 Source Node，所以用 Channel 平均一下或者保留 Channel 差异)
            # 这里 Sparsity 是针对每个 Channel 计算的，这很好，不同通道的处理方式不同
            
            gate_expanded = gate # [C, N, 1]
            bonus_expanded = freq_bonus.view(1, -1, 1).to(self.synapse_tensor.device) # [1, N, 1]
            
            dynamic_limits = base_limit + bonus_expanded * gate_expanded
            
            # 6. 应用归一化
            scaling_factor = dynamic_limits / row_sums
            scaling_factor = torch.clamp(scaling_factor, max=1.0)
            self.synapse_tensor *= scaling_factor
            
            # === 5. 死亡剪枝 (Pruning) ===
            mask = torch.abs(self.synapse_tensor) < pruning_threshold
            
            pruned_count = mask.sum().item()
            total_count = self.synapse_tensor.numel()
            
            self.synapse_tensor.masked_fill_(mask, 0.0)
            
            # 情绪回归
            self.mood_bias *= 0.9
            
            return pruned_count, total_count, global_decay

    def forward(self, input_indices, steps=3, channel_weights=None):
        batch_size, seq_len = input_indices.shape
        device = input_indices.device

        if channel_weights is None:
            channel_weights = torch.tensor([0.3, 0.2, 0.3, 0.2, 0.0], device=device)
        
        channel_weights = channel_weights / channel_weights.sum()
        mixed_synapse = torch.einsum('c,cij->ij', channel_weights, self.synapse_tensor)
        attn_weights = self.get_attention_weights(input_indices)

        current_thought = torch.zeros(
            batch_size, self.vocab_size, device=device
        )
        current_thought.scatter_add_(1, input_indices, attn_weights)

        for _ in range(steps):
            current_thought = torch.matmul(current_thought, mixed_synapse)
            current_thought = torch.relu(current_thought - 0.1) 
            current_thought += self.mood_bias * 0.05

        if self.training:
            self.learn_from_input(input_indices)

        return current_thought

    def generate_reply(self, input_indices, max_len=20, channel_weights=None):
        self.eval()
        device = input_indices.device

        with torch.no_grad():
            thought_energy = self(input_indices, steps=1, channel_weights=channel_weights)

        if channel_weights is None:
             channel_weights = torch.tensor([0.3, 0.2, 0.3, 0.2, 0.0], device=device)
        channel_weights = channel_weights / channel_weights.sum()
        mixed_synapse = torch.einsum('c,cij->ij', channel_weights, self.synapse_tensor)

        visited = set(input_indices[0].tolist())
        visited.add(0) 
        visited.add(1)

        start_energy = thought_energy[0].clone()
        for v in visited:
            start_energy[v] = -float("inf")

        max_val, max_idx = torch.max(start_energy, dim=0)
        
        current_idx = -1
        if max_val > 0.5: 
            current_idx = max_idx.item()
        else:
            if max_val == -float("inf"):
                 candidates = torch.ones_like(start_energy)
                 candidates[0] = 0
                 candidates[1] = 0
                 probs = candidates / candidates.sum()
            else:
                probs = torch.softmax(start_energy * 5.0, dim=0)
            
            current_idx = torch.multinomial(probs, 1).item()

        if current_idx in [0, 1]:
             return []

        reply_indices = [current_idx]
        visited.add(current_idx)
        
        for _ in range(max_len):
            next_step_weights = mixed_synapse[current_idx].clone()

            strongest_conn = torch.max(next_step_weights)
            strongest_idx = torch.argmax(next_step_weights).item()
            
            if strongest_conn > 0.8 and strongest_idx not in visited and strongest_idx > 1:
                 next_idx = strongest_idx
            else:
                for v in visited:
                    next_step_weights[v] = -float("inf")
                
                next_step_weights[0] = -float("inf")
                next_step_weights[1] = -float("inf")

                guidance = thought_energy[0] * 0.5
                combined_weights = next_step_weights + guidance

                if torch.max(combined_weights) == -float("inf"):
                    break

                next_probs = torch.softmax(combined_weights * 3.0, dim=0)
                next_idx = torch.multinomial(next_probs, 1).item()

            reply_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        return reply_indices
