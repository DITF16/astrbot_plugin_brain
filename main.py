from astrbot.core.star import StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import AstrBotConfig, logger
from .brain_interface import BrainInterface


PLUGIN_NAME = "astrbot_plugin_brain"
DATA_DIR = StarTools.get_data_dir(PLUGIN_NAME)
# 路径: /AstrBot/data/plugin_data/astrbot_plugin_brain/my_brain.pth
BRAIN_PATH = DATA_DIR / "my_brain.pth"

class CognitiveBrainPlugin(Star):
    """
    [V3.0] 认知大脑插件
    特性:
    1. Hebbian Learning (联想学习)
    2. Logical Imprinting (逻辑刻印)
    3. Pressure-Driven Forgetting (压力驱动遗忘)
    """
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        # 初始化大脑接口
        self.brain = BrainInterface(model_path=BRAIN_PATH, vocab_limit=10000)
        
        # 记录上一句回复的内容，用于RL (Reinforcement Learning)
        # 格式: {user_id: [indices]}
        self.last_reply_indices = {}

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        """
        监听群聊消息:
        1. 只要你在说话，我就在学习 (Passive Learning)
        2. 如果你叫我的名字，或者 @我，我会回复 (Active Reply)
        """
        text = event.message_str
        user_id = event.get_sender_id()

        # === 1. 被动学习 (Listening) ===
        # 无论是否回复，都在默默强化突触
        # 过滤掉指令类消息
        if not text.startswith("/") and len(text) > 1:
            loss = self.brain.learn(text)
            if loss > 0.0:
                 # 可以在日志里看，但别发出来吵人
                 logger.debug(f"[Brain] Learned from input. Loss: {loss:.4f}")

        # === 2. 逻辑刻印 (Teaching) ===
        # 简单句式: "A是B", "A有B"
        if "是" in text and len(text) < 10:
            parts = text.split("是")
            if len(parts) == 2:
                A, B = parts[0].strip(), parts[1].strip()
                if A and B:
                    # CHANNEL 0: IS_A
                    cnt = self.brain.learn_logical([(A, 0, B)])
                    if cnt > 0:
                        logger.info(f"[Brain] Logic Imprinted: {A} IS {B}")

        # === 3. 主动回复 (Replying) ===
        # 只有被 @ 或者提到关键词才回复 (防止插嘴)
        # 这里假设机器人名字叫 "夏娃" 或 "Eve"
        trigger_words = ["夏娃", "Eve", "eve"]
        is_at = False # 暂时拿不到 at 信息，简化处理
        
        should_reply = any(w in text for w in trigger_words)

        if should_reply:
            reply_text, indices = self.brain.reply(text)
            if reply_text:
                self.last_reply_indices[user_id] = indices # 记住这次回复，等待反馈
                yield event.plain_result(f"{reply_text}")
    
    @filter.command("夏娃好棒")
    async def good_girl(self, event: AstrMessageEvent):
        """
        [RL] 正向反馈
        """
        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=1.0)
            yield event.plain_result("(脸红) 真的吗... 嘿嘿，我会记住这种感觉的！【开心】")
        else:
            yield event.plain_result("欸？我刚才说什么了吗？【疑惑】")

    @filter.command("夏娃闭嘴")
    async def bad_girl(self, event: AstrMessageEvent):
        """
        [RL] 负向反馈
        """
        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=-1.0)
            yield event.plain_result("(耳朵耷拉下来) 呜... 对不起，我再也不这样说了...【难过】")
        else:
            yield event.plain_result("我明明什么都没说...【委屈】")

    @filter.command("夏娃睡觉")
    async def sleep_now(self, event: AstrMessageEvent):
        """
        强制触发睡眠整理
        """
        if not self.brain: return
        yield event.plain_result("💤 正在整理记忆突触... (请勿打扰)")
        try:
            pruned, ratio, decay = self.brain.trigger_sleep()
            msg = f"✨ 睡醒啦！精神百倍！\n本次睡眠清理了 {pruned} 个微弱连接 (占比 {ratio:.1f}%)。"
            if decay < 1.0:
                msg += f"\n⚠️ 大脑压力过大，已启动强制遗忘 (衰减系数: {decay:.2f})"
            else:
                msg += "\n🧠 大脑容量充足，无需强制遗忘。"
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"Sleep failed: {e}")
            yield event.plain_result("😫 睡不着... (睡眠程序出错)")
