import random
import os
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from .brain_interface import BrainInterface


class CognitiveBrainPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

        # 1. 初始化路径
        # 获取当前插件文件夹的绝对路径，确保能找到 my_brain.pth
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(plugin_dir, "my_brain.pth")

        # 2. 启动大脑
        # vocab_limit 可以根据你的服务器内存调整
        self.brain = BrainInterface(model_path=save_path, vocab_limit=10000)
        self.learn_counter = 0

        # 日志输出
        print(f"🧠 [CognitiveBrain] 神经元连接完毕。记忆路径: {save_path}")

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        """
        核心监听逻辑：监听群聊消息 -> 学习 -> (概率)回复
        """

        # 1. 获取纯文本消息
        # AstrBot 文档：event.message_str 获取处理后的纯文本
        text = event.message_str
        if not text:
            return

        # 2. 学习 (Fire together, wire together)
        # 即使不回复，大脑也在后台静默建立连接
        self.brain.learn(text)
        self.learn_counter += 1

        # 3. 自动存盘 (记忆固化)
        # 每学习 50 句话保存一次
        if self.learn_counter >= 50:
            self.brain.save_brain()
            self.learn_counter = 0

        # 4. 回复判定逻辑
        should_reply = False

        # 判定 A: 机器人被 @ 了 (需要检查 event 属性)
        # 注意：不同适配器实现可能不同，这里检查消息文本是否包含机器人名字或特定触发
        # 也可以检查 event.message_obj.mentions 等，这里用最通用的文本判断
        # if "机器人名字" in text: should_reply = True

        # 判定 B: 随机插嘴 (模仿人类)
        # 设定 10% 的概率插话
        if random.random() < 0.1:
            should_reply = True

        # 5. 生成并发送
        if should_reply:
            reply_text = self.brain.reply(text)

            # 过滤掉无意义的短回复
            if len(reply_text) > 1:
                # 使用 event.send 发送文本
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

        status_msg = (
            f"🧠 认知图谱状态:\n"
            f"----------------\n"
            f"📚 词汇量: {vocab_count} / {limit}\n"
            f"⚡ 突触连接: 正常\n"
            f"💾 下次自动保存: 还需 {50 - self.learn_counter} 条学习"
        )
        yield event.plain_result(status_msg)
