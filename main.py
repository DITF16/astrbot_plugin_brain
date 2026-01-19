from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.star import StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import AstrBotConfig, logger
from .brain_interface import BrainInterface
from astrbot.api.provider import ProviderRequest

PLUGIN_NAME = "astrbot_plugin_brain"
DATA_DIR = StarTools.get_data_dir(PLUGIN_NAME)
BRAIN_PATH = DATA_DIR / "my_brain.pth"


class CognitiveBrainPlugin(Star):
    """
    夏娃模型学习及回复
    - 学习：监听白名单内所有消息（被动学习）
    - 回复：仅在 LLM 请求时拦截（主动回复）
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        brain_config = config.get("brain") or {}
        vocab_limit = brain_config.get("vocab_limit", 10000)

        self.brain = BrainInterface(
            config=dict(config),
            model_path=BRAIN_PATH,
            vocab_limit=vocab_limit
        )

        self.last_reply_indices = {}

        # === 读取白名单配置 ===
        whitelist_config = config.get("whitelist") or {}
        self.whitelist_enabled = whitelist_config.get("enable", False)
        self.whitelist_groups = set(str(g) for g in whitelist_config.get("groups", []))
        self.whitelist_users = set(str(u) for u in whitelist_config.get("users", []))

        logger.info(f"[Brain] 白名单状态: {'启用' if self.whitelist_enabled else '禁用'}")
        if self.whitelist_enabled:
            logger.info(f"[Brain] 群聊白名单: {self.whitelist_groups}")
            logger.info(f"[Brain] 私聊白名单: {self.whitelist_users}")

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        """检查消息来源是否在白名单中"""
        if not self.whitelist_enabled:
            return True

        group_id = getattr(event.message_obj, 'group_id', None)
        user_id = event.get_sender_id()

        if group_id:
            return str(group_id) in self.whitelist_groups
        else:
            return str(user_id) in self.whitelist_users

    # ============================================================
    # 📚 学习模块：监听所有消息（被动学习）
    # ============================================================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message_learn(self, event: AstrMessageEvent):
        """
        监听白名单内的所有消息，用于学习。
        不产生任何回复，只是默默学习。
        """
        # === 白名单检查 ===
        if not self._is_allowed(event):
            return

        text = event.message_str

        # 过滤指令和过短消息
        if text.startswith("/") or len(text) <= 1:
            return

        # === 双重编码学习 ===
        try:
            log_msg = await self.brain.learn_dual_coding(text)
            logger.info(f"[夏娃模型] 学习: {text[:20]}... -> {log_msg}")
        except Exception as e:
            logger.error(f"[夏娃模型] Learning error: {e}")

        # 不 yield 任何内容，不产生回复
        return

    # ============================================================
    # 💬 回复模块：拦截 LLM 请求（主动回复）
    # ============================================================
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        拦截 LLM 请求，使用大脑生成回复。
        此时消息已经在 on_message_learn 中学习过了。
        """
        # === 白名单检查 ===
        if not self._is_allowed(event):
            return  # 继续正常 LLM 流程

        text = event.message_str
        user_id = event.get_sender_id()

        # === 使用大脑生成回复 ===
        try:
            reply_text, indices = await self.brain.reply(text)

            if reply_text:
                self.last_reply_indices[user_id] = indices
                await event.send(MessageChain().message(reply_text))
                event.stop_event()
            else:
                logger.info("[夏娃模型] 无法生成回复，交给 LLM 处理")
                return  # 继续 LLM 流程

        except Exception as e:
            logger.error(f"[夏娃模型] Reply error: {e}")
            return

    # ============================================================
    # 🎮 指令模块
    # ============================================================
    @filter.command("夏娃好棒")
    async def good_girl(self, event: AstrMessageEvent):
        if not self._is_allowed(event):
            return

        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=1.0)
            yield event.plain_result("(脸红) 真的吗... 嘿嘿，我会记住这种感觉的！【开心】")
        else:
            yield event.plain_result("欸？我刚才说什么了吗？【疑惑】")

    @filter.command("夏娃闭嘴")
    async def bad_girl(self, event: AstrMessageEvent):
        if not self._is_allowed(event):
            return

        user_id = event.get_sender_id()
        indices = self.last_reply_indices.get(user_id)
        if indices:
            self.brain.reinforce(indices, reward_sign=-1.0)
            yield event.plain_result("(耳朵耷拉下来) 呜... 对不起，我再也不这样说了...【难过】")
        else:
            yield event.plain_result("我明明什么都没说...【委屈】")

    @filter.command("夏娃睡觉")
    async def sleep_now(self, event: AstrMessageEvent):
        if not self._is_allowed(event):
            return

        if not self.brain:
            return
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
        if not self._is_allowed(event):
            return

        if not self.brain:
            yield event.plain_result("🧠 大脑未连接！")
            return

        vocab_size = self.brain.next_idx
        device = str(self.brain.device)

        logic_enabled = self.brain.logic_engine.enable
        logic_temp = self.brain.logic_engine.temperature

        expr_enabled = self.brain.expression_engine.enable
        expr_temp = self.brain.expression_engine.temperature

        wl_status = "✅ 启用" if self.whitelist_enabled else "❌ 禁用"
        wl_groups_count = len(self.whitelist_groups)
        wl_users_count = len(self.whitelist_users)

        msg = (
            f"🧠 [夏娃系统状态]\n"
            f"---------------------------\n"
            f"📚 词汇量: {vocab_size} / {self.brain.vocab_limit}\n"
            f"⚙️ 运行设备: {device}\n"
            f"🔍 逻辑前额叶: {'✅' if logic_enabled else '❌'} (Temp: {logic_temp})\n"
            f"🗣️ 表达中枢: {'✅' if expr_enabled else '❌'} (Temp: {expr_temp})\n"
            f"📋 白名单: {wl_status} (群:{wl_groups_count} 私:{wl_users_count})\n"
            f"---------------------------\n"
            f"💡 全脑协同工作中..."
        )
        yield event.plain_result(msg)
