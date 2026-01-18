import torch
from cognitive_graph_model import CognitiveGraphModel  # 假设之前的类保存在这
from cognitive_trainer import HebbianTrainer

# --- 1. 准备极简的世界知识 (Micro World) ---
# 手动构建一个微型词表
vocab = [
    "<PAD>",
    "女王",
    "是",
    "女性",
    "苹果",
    "水果",
    "毒",
    "好吃",
    "红色的",
    "喜欢",
    "吃",
    "权力",
    "宫殿",
]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

# 准备几条简单的训练语料
corpus = [
    ["女王", "是", "女性"],
    ["女王", "喜欢", "权力"],
    [
        "女王",
        "住",
        "在",
        "宫殿",
    ],  # 注意：'住'和'在'不在词表里，演示时会报错，下面需处理
    ["苹果", "是", "水果"],
    ["苹果", "是", "红色的"],
    ["女王", "吃", "苹果"],
    ["毒", "苹果", "是", "红色的"],  # 制造一点混淆逻辑
]


# 预处理：把不在词表里的词过滤掉 (模拟听不懂)
def tokenize(text_list):
    return [word2idx[w] for w in text_list if w in word2idx]


clean_corpus = [tokenize(s) for s in corpus]

# --- 2. 初始化大脑 ---
print("🧠 初始化大脑...")
model = CognitiveGraphModel(vocab_size=len(vocab), embed_dim=16)
trainer = HebbianTrainer(model, learning_rate=0.5)  # 学习率设大点，效果立竿见影

# --- 3. 开始学习 (Training Loop) ---
print("\n📚 开始学习阶段...")
for epoch in range(5):  # 读5遍书
    print(f"--- Epoch {epoch+1} ---")
    for sentence in clean_corpus:
        if not sentence:
            continue
        avg_weight = trainer.train_step(sentence)

    # 打印一点内部状态看看
    # 看看"女王"现在的经验值
    q_idx = word2idx["女王"]
    count = model.word_counts[q_idx].item()
    print(f"   [状态] '女王' 被激活次数: {int(count)}")

# --- 4. 检验成果 (Inference) ---
print("\n✨ 学习结束，开始测试联想能力...")


def chat(start_word):
    if start_word not in word2idx:
        print("??? 我没学过这个词。")
        return

    start_idx = word2idx[start_word]
    input_tensor = torch.tensor([[start_idx]])  # [1, 1]

    print(f"\nQ: 说说关于'{start_word}'的事？")

    # 使用我们之前写的 generate_reply
    reply_ids = model.generate_reply(input_tensor, max_len=5)

    # 解码回文字
    reply_words = [idx2word[idx] for idx in reply_ids]
    print(f"A: {' -> '.join(reply_words)}")


# 测试 1: 问女王
chat("女王")
# 预期逻辑链：女王 -> 权力 / 喜欢 / 苹果

# 测试 2: 问苹果
chat("苹果")
# 预期逻辑链：苹果 -> 红色的 / 水果 / 吃
