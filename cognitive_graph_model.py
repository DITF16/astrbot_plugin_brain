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

    def process_sleep_cycle(self, pruning_threshold=0.005):
        """
        睡眠机制：修复过度清理
        """
        with torch.no_grad():
            total_pruned = 0
            total_count = 0

            # === 1. 压力检测 ===
            target_capacity = float(self.vocab_size) * 50.0
            current_energy = self.synapse_tensor.abs().sum().item()

            if current_energy > target_capacity:
                pressure_ratio = target_capacity / current_energy
                global_decay = max(pressure_ratio, 0.95)
            else:
                global_decay = 1.0

            num_channels = self.synapse_tensor.shape[0]

            for c in range(num_channels):
                channel_data = self.synapse_tensor.data[c]

                # --- 晶体化保护 (降低保护阈值) ---
                weights_abs = channel_data.abs()
                shield = torch.sigmoid((weights_abs - 0.2) * 8.0)
                final_decay_c = global_decay * (1.0 - shield) + shield
                channel_data.mul_(final_decay_c)
                del weights_abs, shield, final_decay_c

                # --- 对比度增强 (更温和) ---
                signs = torch.sign(channel_data)
                channel_data.abs_()
                channel_data.pow_(1.02)  # 从1.1降到1.02，非常温和
                channel_data.mul_(signs)
                del signs

                # --- 动态能量守恒 ---
                row_sums = channel_data.abs().sum(dim=1, keepdim=True) + 1e-6
                row_maxs = channel_data.abs().max(dim=1, keepdim=True)[0]
                sparsity = row_maxs / row_sums
                gate = torch.tanh(sparsity * 5.0)

                counts = self.word_counts
                base_limit = 30.0
                freq_bonus = torch.log1p(counts).view(-1, 1).to(channel_data.device) * 3.0

                dynamic_limits = base_limit + freq_bonus * gate
                scaling_factor = (dynamic_limits / row_sums).clamp(max=1.0)
                channel_data.mul_(scaling_factor)
                del row_sums, row_maxs, sparsity, gate, scaling_factor

                # --- 剪枝 (只清理真正的噪声) ---
                mask = channel_data.abs() < pruning_threshold
                total_pruned += mask.sum().item()
                total_count += mask.numel()
                channel_data.masked_fill_(mask, 0.0)
                del mask

            self.mood_bias.mul_(0.95)

            if self.synapse_tensor.is_cuda:
                torch.cuda.empty_cache()

            return total_pruned, total_count, global_decay

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

        return current_thought

    def _learn_associations(self, input_indices, learning_rate=0.1):
        """
        自动学习：相邻的词建立关联连接
        """
        with torch.no_grad():
            for batch in input_indices:
                indices = batch.tolist()
                for i in range(len(indices) - 1):
                    u, v = indices[i], indices[i + 1]

                    # 跳过特殊token (PAD=0, UNK=1 等)
                    if u <= 1 or v <= 1:
                        continue

                    # 双向增强关联通道
                    current_weight = self.synapse_tensor[CHANNEL_ASSOCIATED, u, v].item()

                    # 使用递减学习率：已有强连接时学得慢，弱连接学得快
                    effective_lr = learning_rate / (1.0 + abs(current_weight))

                    self.synapse_tensor.data[CHANNEL_ASSOCIATED, u, v] += effective_lr
                    self.synapse_tensor.data[CHANNEL_ASSOCIATED, v, u] += effective_lr * 0.5  # 反向弱一点

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
