import asyncio
from datetime import datetime
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


class SleepScheduler:
    """
    🌙 睡眠调度器 - 模拟生物睡眠节律
    """

    def __init__(self, config: dict):
        # 获取 sleep 子配置（嵌套结构）
        sleep_config = config.get("sleep", {})

        # === 定时睡眠配置 (昼夜节律) ===
        self.scheduled_enabled = sleep_config.get("scheduled_enabled", True)
        self.scheduled_hour = sleep_config.get("scheduled_hour", 3)
        self.scheduled_minute = sleep_config.get("scheduled_minute", 0)

        # === 疲劳度睡眠配置 ===
        self.fatigue_enabled = sleep_config.get("fatigue_enabled", True)
        self.fatigue_threshold = sleep_config.get("fatigue_threshold", 500)
        self.fatigue_counter = 0

        # === 空闲睡眠配置 ===
        self.idle_enabled = sleep_config.get("idle_enabled", True)
        self.idle_timeout = sleep_config.get("idle_timeout", 3600)
        self.last_activity_time = datetime.now()

        # === 压力睡眠配置 ===
        self.pressure_enabled = sleep_config.get("pressure_enabled", True)
        self.pressure_check_interval = sleep_config.get("pressure_check_interval", 600)  # 10分钟检查一次
        self.pressure_threshold = sleep_config.get("pressure_threshold", 0.8)  # 80%触发

        # === 睡眠冷却 ===
        self.min_sleep_interval = sleep_config.get("min_sleep_interval", 1800)
        self.last_sleep_time = None

        # === 状态追踪 ===
        self.is_sleeping = False
        self.total_sleeps_today = 0
        self.last_reset_date = datetime.now().date()

    def record_activity(self):
        """记录活动，重置空闲计时器"""
        self.last_activity_time = datetime.now()
        self.fatigue_counter += 1

    def can_sleep(self) -> bool:
        """检查是否可以进入睡眠（冷却检查）"""
        if self.is_sleeping:
            return False
        if self.last_sleep_time:
            elapsed = (datetime.now() - self.last_sleep_time).total_seconds()
            if elapsed < self.min_sleep_interval:
                return False
        return True

    def mark_sleep_done(self):
        """标记睡眠完成"""
        self.last_sleep_time = datetime.now()
        self.is_sleeping = False
        self.fatigue_counter = 0  # 重置疲劳度
        self.total_sleeps_today += 1

        # 每日重置统计
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.total_sleeps_today = 1
            self.last_reset_date = today

    def check_fatigue_sleep(self) -> bool:
        """检查是否需要疲劳睡眠"""
        if not self.fatigue_enabled:
            return False
        return self.fatigue_counter >= self.fatigue_threshold

    def check_idle_sleep(self) -> bool:
        """检查是否需要空闲睡眠"""
        if not self.idle_enabled:
            return False
        elapsed = (datetime.now() - self.last_activity_time).total_seconds()
        return elapsed >= self.idle_timeout

    def check_scheduled_sleep(self) -> bool:
        """检查是否到达定时睡眠时间"""
        if not self.scheduled_enabled:
            return False
        now = datetime.now()
        # 检查是否在目标时间的5分钟窗口内
        target = now.replace(hour=self.scheduled_hour, minute=self.scheduled_minute, second=0)
        diff = abs((now - target).total_seconds())
        return diff < 300  # 5分钟窗口

    def get_status(self) -> dict:
        """获取睡眠调度器状态"""
        return {
            "fatigue": f"{self.fatigue_counter}/{self.fatigue_threshold}",
            "idle_seconds": int((datetime.now() - self.last_activity_time).total_seconds()),
            "sleeps_today": self.total_sleeps_today,
            "can_sleep": self.can_sleep(),
            "is_sleeping": self.is_sleeping
        }


class CognitiveBrainPlugin(Star):
    """
    夏娃模型学习及回复
    - 学习：监听白名单内所有消息（被动学习）
    - 回复：仅在 LLM 请求时拦截（主动回复）
    - 睡眠：多种触发机制的智能睡眠
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

        logger.info(f"[夏娃模型] 白名单状态: {'启用' if self.whitelist_enabled else '禁用'}")

        # === 初始化睡眠调度器 ===
        self.sleep_scheduler = SleepScheduler(dict(config))

        # === 后台任务管理 ===
        self._stop_flag = False
        self._auto_save_task: asyncio.Task | None = None
        self._sleep_monitor_task: asyncio.Task | None = None

        # 启动自动保存任务
        self.auto_save_interval = brain_config.get("save_interval", 300)
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        # 启动睡眠监控任务
        self._sleep_monitor_task = asyncio.create_task(self._sleep_monitor_loop())

        logger.info("[夏娃模型] 睡眠调度器已启动 🌙")

    # ============================================================
    # 🔄 后台任务
    # ============================================================

    async def _auto_save_loop(self):
        """每隔一段时间自动保存大脑"""
        while not self._stop_flag:
            try:
                await asyncio.sleep(self.auto_save_interval)
                if self._stop_flag:
                    break
                self.brain.save_brain()
                logger.info("[夏娃模型] 自动保存完成")
            except asyncio.CancelledError:
                logger.info("[夏娃模型] 自动保存任务被取消")
                break
            except Exception as e:
                logger.error(f"[夏娃模型] 自动保存失败: {e}")

    async def _sleep_monitor_loop(self):
        """睡眠监控循环"""
        await asyncio.sleep(60)

        last_pressure_check = datetime.now()

        while not self._stop_flag:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                if self._stop_flag:
                    break

                if not self.sleep_scheduler.can_sleep():
                    continue

                sleep_reason = None
                sleep_type = None

                # 1️⃣ 定时睡眠
                if self.sleep_scheduler.check_scheduled_sleep():
                    sleep_reason = "昼夜节律"
                    sleep_type = "deep"

                # 2️⃣ 疲劳睡眠
                elif self.sleep_scheduler.check_fatigue_sleep():
                    sleep_reason = "疲劳积累"
                    sleep_type = "nap"

                # 3️⃣ 空闲睡眠
                elif self.sleep_scheduler.check_idle_sleep():
                    sleep_reason = "空闲休眠"
                    sleep_type = "light"

                # 4️⃣ 压力睡眠（按配置间隔检查）
                elif self.sleep_scheduler.pressure_enabled:
                    elapsed = (datetime.now() - last_pressure_check).total_seconds()
                    if elapsed >= self.sleep_scheduler.pressure_check_interval:
                        last_pressure_check = datetime.now()
                        pressure = self._check_brain_pressure()
                        if pressure > self.sleep_scheduler.pressure_threshold:
                            sleep_reason = f"大脑压力过载 ({pressure:.0%})"
                            sleep_type = "emergency"

                if sleep_reason:
                    await self._auto_sleep(sleep_reason, sleep_type)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[夏娃模型] 睡眠监控异常: {e}")

    def _check_brain_pressure(self) -> float:
        """
        检查大脑压力（突触密度）
        返回 0.0 ~ 1.0 的压力值
        """
        try:
            synapse = self.brain.model.synapse_weights
            # 计算非零连接的比例
            total_elements = synapse.numel()
            nonzero_count = (synapse.abs() > 0.01).sum().item()
            density = nonzero_count / total_elements
            return density
        except Exception:
            return 0.0

    async def _auto_sleep(self, reason: str, sleep_type: str):
        """
        执行自动睡眠
        """
        self.sleep_scheduler.is_sleeping = True

        # 根据睡眠类型选择不同的处理方式
        sleep_configs = {
            "deep": {"name": "深度睡眠", "emoji": "🌙", "extra_decay": 0.0},
            "nap": {"name": "小憩", "emoji": "😴", "extra_decay": 0.0},
            "light": {"name": "浅睡眠", "emoji": "💤", "extra_decay": 0.0},
            "emergency": {"name": "紧急休眠", "emoji": "⚠️", "extra_decay": 0.1}
        }

        config = sleep_configs.get(sleep_type, sleep_configs["light"])

        logger.info(f"[夏娃模型] {config['emoji']} 触发{config['name']} - 原因: {reason}")

        try:
            # 执行睡眠周期
            pruned, total, decay = self.brain.trigger_sleep()
            ratio = pruned / total * 100 if total > 0 else 0

            self.sleep_scheduler.mark_sleep_done()

            logger.info(
                f"[夏娃模型] ✨ {config['name']}完成！"
                f"清理了 {pruned} 个连接 ({ratio:.1f}%), "
                f"衰减系数: {decay:.2f}"
            )

        except Exception as e:
            logger.error(f"[夏娃模型] 自动睡眠失败: {e}")
            self.sleep_scheduler.is_sleeping = False

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

    def extract_text(self, message_chain: list) -> str:
        """从消息链中提取纯文本"""
        texts = []
        for component in message_chain:
            if hasattr(component, 'type') and component.type.value == 'Plain':
                texts.append(component.text)
        return ''.join(texts)

    # ============================================================
    # 📚 学习模块
    # ============================================================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message_learn(self, event: AstrMessageEvent):
        """监听白名单内的所有消息，用于学习"""
        if not self._is_allowed(event):
            return

        message_chain = event.get_messages()
        text = self.extract_text(message_chain)
        # logger.info("text = " + text)

        if text.startswith("/") or len(text) <= 1:
            return

        # 记录活动（用于睡眠调度）
        self.sleep_scheduler.record_activity()

        try:
            log_msg = await self.brain.learn_dual_coding(text)
            logger.info(f"[夏娃模型] 学习: {text[:20]}... -> {log_msg}")
        except Exception as e:
            logger.error(f"[夏娃模型] Learning error: {e}")

        return

    # ============================================================
    # 💬 回复模块
    # ============================================================
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """拦截 LLM 请求，使用大脑生成回复"""
        if not self._is_allowed(event):
            return

        # 如果正在睡觉，不回复
        if self.sleep_scheduler.is_sleeping:
            logger.info("[夏娃模型] 正在睡眠中，跳过回复")
            return

        text = event.message_str
        user_id = event.get_sender_id()

        try:
            reply_text, indices = await self.brain.reply(text)

            if reply_text:
                self.last_reply_indices[user_id] = indices
                await event.send(MessageChain().message(reply_text))
                event.stop_event()
            else:
                return

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
        """手动触发睡眠"""
        if not self._is_allowed(event):
            return

        if not self.brain:
            return

        if self.sleep_scheduler.is_sleeping:
            yield event.plain_result("💤 我已经在睡了啦... zzZ")
            return

        yield event.plain_result("💤 正在整理记忆突触... (请勿打扰)")

        self.sleep_scheduler.is_sleeping = True
        try:
            pruned, ratio, decay = self.brain.trigger_sleep()
            self.sleep_scheduler.mark_sleep_done()

            msg = f"✨ 睡醒啦！精神百倍！\n本次睡眠清理了 {pruned} 个微弱连接 (占比 {ratio:.1f}%)。"
            if decay < 1.0:
                msg += f"\n⚠️ 大脑压力过大，已启动强制遗忘 (衰减系数: {decay:.2f})"
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"Sleep failed: {e}")
            self.sleep_scheduler.is_sleeping = False
            yield event.plain_result("😫 睡不着... (睡眠程序出错)")

    @filter.command("夏娃状态")
    async def brain_status(self, event: AstrMessageEvent):
        """查看系统状态"""
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

        # 睡眠状态
        sleep_status = self.sleep_scheduler.get_status()
        brain_pressure = self._check_brain_pressure()

        msg = (
            f"🧠 [夏娃系统状态]\n"
            f"---------------------------\n"
            f"📚 词汇量: {vocab_size} / {self.brain.vocab_limit}\n"
            f"⚙️ 运行设备: {device}\n"
            f"🔍 逻辑前额叶: {'✅' if logic_enabled else '❌'} (Temp: {logic_temp})\n"
            f"🗣️ 表达中枢: {'✅' if expr_enabled else '❌'} (Temp: {expr_temp})\n"
            f"📋 白名单: {wl_status} (群:{wl_groups_count} 私:{wl_users_count})\n"
            f"---------------------------\n"
            f"🌙 [睡眠状态]\n"
            f"💤 状态: {'睡眠中' if sleep_status['is_sleeping'] else '清醒'}\n"
            f"😫 疲劳度: {sleep_status['fatigue']}\n"
            f"⏰ 空闲时间: {sleep_status['idle_seconds']}秒\n"
            f"🧠 大脑压力: {brain_pressure:.1%}\n"
            f"😴 今日睡眠: {sleep_status['sleeps_today']}次\n"
            f"---------------------------\n"
            f"💡 全脑协同工作中..."
        )
        yield event.plain_result(msg)

    async def terminate(self):
        """插件关闭时的清理工作"""
        logger.info("[夏娃模型] 系统关闭，保存记忆中...")

        self._stop_flag = True

        # 取消所有后台任务
        tasks = [self._auto_save_task, self._sleep_monitor_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.brain.save_brain()
        logger.info("[夏娃模型] 记忆保存完毕，再见~")
