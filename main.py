import random
import os
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig
from .brain_interface import BrainInterface


class CognitiveBrainPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config  # 保存配置对象

        # 1. 初始化路径
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(plugin_dir, "my_brain.pth")

        # 2. 启动大脑
        self.brain = BrainInterface(model_path=save_path, vocab_limit=10000)
        self.learn_counter = 0

        print(f"🧠 [CognitiveBrain] 神经元连接完毕。记忆路径: {save_path}")

    # 将监听类型改为群聊 GROUP_MESSAGE，以符合您的需求
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        """
        核心监听逻辑：监听群聊消息 -> 检查配置 -> 学习 -> (概率)回复
        """
        text = event.message_str
        if not text:
            return

        # 0. 获取当前群号 (转为字符串以便比对)
        # 注意：不同平台 group_id 类型可能不同，统一转 str
        try:
            current_group_id = str(event.message_obj.group_id)
        except AttributeError:
            # 如果获取失败（极少数情况），暂不处理
            return

        # ================= 配置读取与判断 =================

        # 读取配置列表 (如果没有配置，默认为空列表)
        learn_whitelist = self.config.get("learn_group_ids", [])
        reply_whitelist = self.config.get("reply_group_ids", [])
        reply_rate = self.config.get("random_reply_rate", 0.1)

        # 判断是否允许学习
        # 逻辑：如果白名单为空 -> 允许所有；如果不为空 -> 必须在白名单内
        should_learn = True
        if learn_whitelist and current_group_id not in learn_whitelist:
            should_learn = False

        # 判断是否允许回复
        can_reply_location = True
        if reply_whitelist and current_group_id not in reply_whitelist:
            can_reply_location = False

        # ================================================

        # 2. 学习 (Fire together, wire together)
        if should_learn:
            self.brain.learn(text)
            self.learn_counter += 1

            # 3. 自动存盘 (记忆固化)
            if self.learn_counter >= 50:
                self.brain.save_brain()
                self.learn_counter = 0
        else:
            # 如果不允许学习，直接结束（也不触发回复，因为没过脑子？）
            # 根据需求，如果你希望“不学习但能回复”，可以注释掉下面这行 return
            pass

            # 4. 回复判定逻辑
        should_reply = False

        # 判定 A: 机器人被 @ 了 (始终回复，不受概率控制，但受地点控制)
        # 这里是一个简单的示例判断
        # if "你的机器人名字" in text: should_reply = True

        # 判定 B: 随机插嘴
        if can_reply_location:
            if random.random() < reply_rate:
                should_reply = True

        # 5. 生成并发送
        if should_reply:
            reply_text = self.brain.reply(text)
            if len(reply_text) > 1:
                await event.send(reply_text)

    # [指令] 手动强制保存记忆
    @filter.command("夏娃保存")
    async def manual_save(self, event: AstrMessageEvent):
        self.brain.save_brain()
        yield event.plain_result("🧠 记忆海马体已手动固化。")

    # [指令] 查看大脑健康状态
    @filter.command("夏娃状态")
    async def check_status(self, event: AstrMessageEvent):
        vocab_count = self.brain.next_idx
        limit = self.brain.vocab_limit

        # 获取当前配置用于展示
        curr_rate = self.config.get("random_reply_rate", 0.1)

        status_msg = (
            f"🧠 认知图谱状态:\n"
            f"----------------\n"
            f"📚 词汇量: {vocab_count} / {limit}\n"
            f"🎲 回复概率: {int(curr_rate * 100)}%\n"
            f"💾 下次自动保存: 还需 {50 - self.learn_counter} 条学习"
        )
        yield event.plain_result(status_msg)
