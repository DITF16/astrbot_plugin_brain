from astrbot.core.star import StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import AstrBotConfig, logger
from .brain_interface import BrainInterface


PLUGIN_NAME = "astrbot_plugin_brain"
DATA_DIR = StarTools.get_data_dir(PLUGIN_NAME)
BRAIN_PATH = DATA_DIR / "my_brain.pth"

class CognitiveBrainPlugin(Star):
    """
    [V3.3] 双重编码大脑 + 表达中枢
    特性:
    1. Dual Coding (LLM Logic + Hebbian Intuition)
    2. Reinforcement Learning
    3. Sleep Consolidation
    4. Expression Center (Broca's Area)
    """
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        brain_config = config.get("brain")
        if not brain_config:
            brain_config = {}

        vocab_limit = brain_config.get("vocab_limit", 10000)
        
        self.brain = BrainInterface(
            config=dict(config), 
            model_path=BRAIN_PATH, 
            vocab_limit=vocab_limit
        )
        
        self.last_reply_indices = {}

    # === 修复：回退到标准的 event_message_type 装饰器 ===
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        """
        监听所有消息
        """
        if not event.message_obj.group_id:
            return

        text = event.message_str
        user_id = event.get_sender_id()

        # === 1. 双重编码学习 ===
        if not text.startswith("/") and len(text) > 1:
            try:
                log_msg = await self.brain.learn_dual_coding(text)
            except Exception as e:
                logger.error(f"[Brain] Learning error: {e}")

        # === 2. 主动回复 ===
        trigger_words = ["夏娃", "Eve", "eve"]
        should_reply = any(w in text for w in trigger_words)

        if should_reply:
            # 使用 await 调用 reply
            reply_text, indices = await self.brain.reply(text)
            if reply_text:
                self.last_reply_indices[user_id] = indices 
                yield event.plain_result(f"{reply_text}")
    
    @filter.command("夏娃好棒")
    async def good_girl(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=1.0)
            yield event.plain_result("(脸红) 真的吗... 嘿嘿，我会记住这种感觉的！【开心】")
        else:
            yield event.plain_result("欸？我刚才说什么了吗？【疑惑】")

    @filter.command("夏娃闭嘴")
    async def bad_girl(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=-1.0)
            yield event.plain_result("(耳朵耷拉下来) 呜... 对不起，我再也不这样说了...【难过】")
        else:
            yield event.plain_result("我明明什么都没说...【委屈】")

    @filter.command("夏娃睡觉")
    async def sleep_now(self, event: AstrMessageEvent):
        if not self.brain: return
        yield event.plain_result("💤 正在整理记忆突触... (请勿打扰)")
        try:
            pruned, ratio, decay = self.brain.trigger_sleep()
            msg = f"✨ 睡醒啦！精神百倍！\n本次睡眠清理了 {pruned} 个微弱连接 (占比 {ratio:.1f}%)。"
            if decay < 1.0:
                msg += f"\n⚠️ 大脑压力过大，已启动强制遗忘 (衰减系数: {decay:.2f})"
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"Sleep failed: {e}")
            yield event.plain_result("😫 睡不着... (睡眠程序出错)")

    @filter.command("夏娃状态")
    async def brain_status(self, event: AstrMessageEvent):
        """查看大脑当前状态"""
        if not self.brain:
            yield event.plain_result("🧠 大脑未连接！")
            return
        
        vocab_size = self.brain.next_idx
        device = str(self.brain.device)
        
        # Logic Info
        logic_enabled = self.brain.logic_engine.enable
        logic_temp = self.brain.logic_engine.temperature
        
        # Expr Info
        expr_enabled = self.brain.expression_engine.enable
        expr_temp = self.brain.expression_engine.temperature

        msg = (
            f"🧠 [夏娃系统状态]\n"
            f"---------------------------\n"
            f"📚 词汇量: {vocab_size} / {self.brain.vocab_limit}\n"
            f"⚙️ 运行设备: {device}\n"
            f"🔍 逻辑前额叶: {'✅' if logic_enabled else '❌'} (Temp: {logic_temp})\n"
            f"🗣️ 表达中枢: {'✅' if expr_enabled else '❌'} (Temp: {expr_temp})\n"
            f"---------------------------\n"
            f"💡 全脑协同工作中..."
        )
        yield event.plain_result(msg)
