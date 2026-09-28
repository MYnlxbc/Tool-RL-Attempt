# Tool-RL-Attempt

模型：Qwen/Qwen3.5-4B
模式：text-only
thinking：关闭 #Qwen3.5会默认生成thinking,第一版RL先关闭
精度：BF16
训练：LoRA
上下文：8192 起步
最大 API 轮数：8～12

hugging face 风格调用：
messages = [
    {
        "role": "user",
        "content": "请查询联系人 Alice 的邮箱地址。",
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template_kwargs={
        "enable_thinking": False,
    },
)

thinking的对照实验：开or关
