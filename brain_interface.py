import torch
import os
import jieba
import re
from .cognitive_graph_model import CognitiveGraphModel
from .cognitive_trainer import HebbianTrainer


class BrainInterface:
    def __init__(self, model_path="my_brain.pth", vocab_limit=5000):
        self.model_path = model_path
        self.vocab_limit = vocab_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 词表管理
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

        # 2. 初始化模型
        self.model = CognitiveGraphModel(vocab_size=vocab_limit, embed_dim=64).to(self.device)
        self.trainer = HebbianTrainer(self.model, learning_rate=0.1)

        # 3. 加载存档
        self.load_brain()

    def load_brain(self):
        if os.path.exists(self.model_path):
            print(f"🧠 正在唤醒大脑: {self.model_path} ...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            vocab_data = checkpoint['vocab']
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
            self.next_idx = vocab_data['next_idx']
            print(f"✅ 唤醒成功。当前词汇量: {self.next_idx}/{self.vocab_limit}")
        else:
            print("✨ 创建了一个全新的大脑。")

    def save_brain(self):
        print("💾 正在写入海马体...")
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

    def trigger_sleep(self):
        """
        [新功能] 触发睡眠整理
        """
        print("💤 进入 REM 睡眠阶段 (压力驱动 & 晶体化保护)...")
        pruned, total, decay = self.model.process_sleep_cycle()
        ratio = pruned / total * 100 if total > 0 else 0
        print(f"✨ 睡眠完成。清理了 {pruned} 个微弱突触 ({ratio:.2f}%)。当前衰减系数: {decay:.3f}")
        self.save_brain()
        return pruned, ratio, decay

    def _clean_text(self, text):
        if not text: return ""
        text_no_cq = re.sub(r'\[CQ:[^\]]+\]', '', text)
        cleaned = re.sub(r'[^\u4e00-\u9fa5]', '', text_no_cq)
        return cleaned

    def _get_or_add_word(self, word):
        """
        获取词ID，如果不存在且词表未满则添加
        """
        if word in self.word2idx:
            return self.word2idx[word]
        
        if self.next_idx < self.vocab_limit:
            new_id = self.next_idx
            self.word2idx[word] = new_id
            self.idx2word[new_id] = word
            self.next_idx += 1
            return new_id
        
        return self.word2idx["<UNK>"]

    def _encode(self, text):
        clean_text = self._clean_text(text)
        if not clean_text:
            return []

        words = list(jieba.cut(clean_text))
        indices = []

        for w in words:
            if w.strip() == "": continue
            idx = self._get_or_add_word(w)
            indices.append(idx)
        return indices

    def learn(self, text):
        """
        [模式1] 联想学习 (兼容旧接口)
        """
        indices = self._encode(text)
        if len(indices) < 2: return 0.0
        # 调用 trainer 的新接口
        loss = self.trainer.train_step_associative(indices)
        return loss

    def learn_logical(self, triplets):
        """
        [模式2] 逻辑学习
        triplets: list of (head_word, relation, tail_word)
        """
        if not triplets: return 0.0
        
        indices_triplets = []
        for head, rel, tail in triplets:
            # 逻辑学习必须精确，所以我们要确保概念进入词表
            h_idx = self._get_or_add_word(head)
            t_idx = self._get_or_add_word(tail)
            
            if h_idx == 1 or t_idx == 1: # UNK
                # 如果词表满了导致全是UNK，逻辑就学不进去了
                continue
                
            indices_triplets.append((h_idx, rel, t_idx))
            
        cnt = self.trainer.train_step_logical(indices_triplets)
        return cnt

    def reply(self, text):
        """
        [Modified] 返回 (reply_text, reply_indices)
        """
        indices = self._encode(text)
        if not indices: return "", []

        input_tensor = torch.tensor([indices], device=self.device)
        
        # 调用 Model 生成
        out_indices = self.model.generate_reply(input_tensor)
        
        reply_words = [self.idx2word.get(idx, "") for idx in out_indices]
        return "".join(reply_words), out_indices

    def reinforce(self, indices, reward_sign):
        """
        [New!] 传递奖惩信号给 Model
        """
        if not indices: return
        self.model.reinforce(indices, reward_sign)
