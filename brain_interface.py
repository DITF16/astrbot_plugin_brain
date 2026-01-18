import torch
import os
import jieba  # 需要 pip install jieba
from cognitive_graph_model import CognitiveGraphModel
from cognitive_trainer import HebbianTrainer


class BrainInterface:
    def __init__(self, model_path="my_brain.pth", vocab_limit=5000):
        self.model_path = model_path
        self.vocab_limit = vocab_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 词表管理 (Mapping)
        # 我们预留一个大的空间 (vocab_limit)，就像婴儿大脑预先长好了神经元
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

        # 2. 初始化模型
        self.model = CognitiveGraphModel(vocab_size=vocab_limit, embed_dim=64).to(self.device)
        self.trainer = HebbianTrainer(self.model, learning_rate=0.1)

        # 3. 尝试加载存档
        self.load_brain()

    def load_brain(self):
        if os.path.exists(self.model_path):
            print(f"🧠 正在唤醒大脑: {self.model_path} ...")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 恢复模型参数
            self.model.load_state_dict(checkpoint['model_state'])

            # 恢复词表 (这是关键！没有词表，模型就是废铁)
            vocab_data = checkpoint['vocab']
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}  # JSON key是str，需转int
            self.next_idx = vocab_data['next_idx']
            print(f"✅ 唤醒成功。当前词汇量: {self.next_idx}/{self.vocab_limit}")
        else:
            print("✨ 创建了一个全新的大脑。")

    def save_brain(self):
        print("💾 正在进入睡眠 (保存记忆)...")
        state = {
            'model_state': self.model.state_dict(),
            'vocab': {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'next_idx': self.next_idx
            }
        }
        torch.save(state, self.model_path)
        print("✅ 记忆已固化。")

    def _encode(self, text):
        """将字符串转换为索引列表，动态学习新词"""
        # 使用 jieba 分词 (处理中文)
        words = list(jieba.cut(text))
        indices = []

        for w in words:
            if w.strip() == "": continue  # 跳过空格

            if w in self.word2idx:
                indices.append(self.word2idx[w])
            else:
                # 遇到新词：如果是新概念且大脑还有空间，就注册它
                if self.next_idx < self.vocab_limit:
                    new_id = self.next_idx
                    self.word2idx[w] = new_id
                    self.idx2word[new_id] = w
                    indices.append(new_id)
                    self.next_idx += 1
                else:
                    # 大脑满了，视为未知 (或者你可以实现淘汰机制)
                    indices.append(self.word2idx["<UNK>"])

        return indices

    def learn(self, text):
        """[输入接口] 听到一句话 -> 学习"""
        indices = self._encode(text)
        if len(indices) < 2: return 0.0  # 太短没法联想

        # 调用我们之前的训练器
        # 注意：这里我们不做 batch 处理，来一句学一句 (Online Learning)
        loss = self.trainer.train_step(indices)
        return loss

    def reply(self, text):
        """[输出接口] 听到一句话 -> 联想回复"""
        indices = self._encode(text)
        if not indices: return "..."

        # 把 list 转 tensor
        input_tensor = torch.tensor([indices], device=self.device)

        # 调用生成
        out_indices = self.model.generate_reply(input_tensor)

        # 解码回文字
        reply_words = [self.idx2word.get(idx, "") for idx in out_indices]
        return "".join(reply_words)
