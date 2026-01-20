import torch
import os
import jieba
import re
import json
import aiohttp
import asyncio
from .cognitive_graph_model import CognitiveGraphModel
from .cognitive_trainer import HebbianTrainer
from astrbot.api import logger

# 逻辑关系映射
RELATION_MAP = {
    0: "IS_A",
    1: "HAS_PROP",
    2: "CAUSES"
}

class LogicDiscriminator:
    """
    [前额叶] 逻辑判别器
    负责调用外部 LLM 提取逻辑三元组
    """
    def __init__(self, config: dict):
        self.enable = config.get("enable", False)
        self.api_base = config.get("api_base", "https://api.openai.com/v1")
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.system_prompt = config.get("system_prompt", "")
        self.timeout = config.get("timeout", 5)
        self.temperature = config.get("temperature", 0.0)

    async def analyze(self, text: str):
        if not self.enable or not self.api_key:
            return {"type": "intuition"}

        if len(text) < 4: 
            return {"type": "intuition"}

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"} 
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        logger.warning(f"[Brain] Logic LLM Error {resp.status}: {await resp.text()}")
                        return {"type": "intuition"}
                    
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    try:
                        result = json.loads(content)
                        if result.get("type") == "logic" and "triplets" in result:
                            return result
                        return {"type": "intuition"}
                    except json.JSONDecodeError:
                        return {"type": "intuition"}

        except Exception as e:
            logger.warning(f"[Brain] Logic Discriminator Failed: {e}")
            return {"type": "intuition"}


class ExpressionCenter:
    """
    [布罗卡氏区] 表达中枢
    负责将脑图生成的关键词串联成通顺的回复
    """
    def __init__(self, config: dict, logic_api_key: str = ""):
        self.enable = config.get("enable", False)
        self.api_base = config.get("api_base", "https://api.openai.com/v1")
        # 允许回退使用 Logic 的 Key
        self.api_key = config.get("api_key") or logic_api_key
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.system_prompt = config.get("system_prompt", "")
        self.timeout = config.get("timeout", 10)
        self.temperature = config.get("temperature", 0.7)

    async def articulate(self, user_input: str, keywords: list):
        """
        生成自然语言回复
        """
        if not keywords:
            return ""

        # 如果未启用或者没有 key，直接回退到简单的词拼接
        if not self.enable or not self.api_key:
            return "".join(keywords)

        prompt = f"User Input: {user_input}\nMemory Keywords: {', '.join(keywords)}"

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        logger.warning(f"[Brain] Expression LLM Error {resp.status}: {await resp.text()}")
                        return "".join(keywords) # Fallback
                    
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content.strip()

        except Exception as e:
            logger.warning(f"[Brain] Expression Center Failed: {e}")
            return "".join(keywords) # Fallback


class BrainInterface:
    def __init__(self, config: dict, model_path="my_brain.pth", vocab_limit=10000):
        self.model_path = model_path
        self.vocab_limit = vocab_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载 LLM 配置
        llm_conf = config.get("llm", {})
        expr_conf = config.get("expression_llm", {})
        
        self.logic_engine = LogicDiscriminator(llm_conf)
        # 传递 logic key 作为 fallback
        self.expression_engine = ExpressionCenter(expr_conf, logic_api_key=llm_conf.get("api_key", ""))

        # 2. 词表管理
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

        # 3. 初始化模型
        self.model = CognitiveGraphModel(vocab_size=vocab_limit, embed_dim=64).to(self.device)
        self.trainer = HebbianTrainer(self.model, learning_rate=0.1)

        # 4. 加载存档
        self.load_brain()

    def load_brain(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                vocab_data = checkpoint['vocab']
                self.word2idx = vocab_data['word2idx']
                self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
                self.next_idx = vocab_data['next_idx']
                logger.info(f"[Brain] Awakened. Vocab: {self.next_idx}/{self.vocab_limit}")
            except Exception as e:
                logger.error(f"[Brain] Load failed, creating new brain: {e}")
        else:
            logger.info("[Brain] Creating a new brain.")

    def save_brain(self):
        state = {
            'model_state': self.model.state_dict(),
            'vocab': {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'next_idx': self.next_idx
            }
        }
        torch.save(state, self.model_path)
        logger.info("[Brain] Memory consolidated.")

    def trigger_sleep(self):
        logger.info("[Brain] Entering REM sleep...")
        pruned, total, decay = self.model.process_sleep_cycle()
        ratio = pruned / total * 100 if total > 0 else 0
        self.save_brain()
        return pruned, ratio, decay

    def _clean_text(self, text):
        if not text: return ""
        text_no_cq = re.sub(r'\[CQ:[^\]]+\]', '', text)
        # 保留空格用于英文分词
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text_no_cq)
        return cleaned

    def _get_or_add_word(self, word):
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
        if not clean_text: return []
        words = list(jieba.cut(clean_text))
        indices = []
        for w in words:
            w = w.strip()  # 去掉空格
            if w == "": continue
            idx = self._get_or_add_word(w)
            indices.append(idx)
        return indices

    async def learn_dual_coding(self, text):
        """
        [双重编码核心]
        """
        analysis = await self.logic_engine.analyze(text)
        
        if analysis["type"] == "logic":
            triplets = analysis.get("triplets", [])
            if triplets:
                indices_triplets = []
                learned_concepts = []
                for head, rel, tail in triplets:
                    h_idx = self._get_or_add_word(str(head))
                    t_idx = self._get_or_add_word(str(tail))
                    if h_idx != 1 and t_idx != 1:
                        REL_MAP = {0: 'is_a', 1: 'has', 2: 'cause'}
                        rel_str = REL_MAP.get(int(rel), None)
                        if rel_str:
                            indices_triplets.append((h_idx, rel_str, t_idx))
                        learned_concepts.append(f"{head}->{RELATION_MAP.get(int(rel), '?')}->{tail}")
                
                if indices_triplets:
                    cnt = self.trainer.train_step_logical(indices_triplets)
                    return f"Logic Imprinted: {', '.join(learned_concepts)}"
        
        indices = self._encode(text)
        if len(indices) >= 2:
            # 联想学习
            loss = self.trainer.train_step_associative(indices)
            return f"Intuition Reinforced (Loss: {loss:.4f})"
        return "Ignored (Too short)"

    async def reply(self, text):
        """
        [主动回复]
        1. 编码输入
        2. 图模型生成关键词索引 (indices)
        3. 表达中枢将关键词转为句子
        """
        indices = self._encode(text)
        if not indices: return "", []
        
        input_tensor = torch.tensor([indices], device=self.device)
        out_indices = self.model.generate_reply(input_tensor)
        
        # 将索引转回词
        reply_words = [self.idx2word.get(idx, "") for idx in out_indices]
        # 过滤掉 PAD 和 UNK
        keywords = [w for w in reply_words if w not in ["<PAD>", "<UNK>", ""]]
        
        # 调用表达中枢进行串联
        final_reply = await self.expression_engine.articulate(text, keywords)
        
        return final_reply, out_indices

    def reinforce(self, indices, reward_sign):
        if not indices: return
        self.model.reinforce(indices, reward_sign)
