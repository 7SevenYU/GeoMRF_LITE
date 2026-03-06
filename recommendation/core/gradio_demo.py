"""
Gradio Web界面 - 完整版（包含Qwen LLM）
"""
import sys
import gradio as gr
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.core.conversation_manager import TBMRiskConversationManager

# 加载LLM配置获取默认参数
config_path = project_root / "config" / "llm_config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    llm_config = json.load(f)
    current_model = llm_config.get("current_model", "qwen")
    model_params = llm_config["models"][current_model]

# 实例化对话管理器
manager = TBMRiskConversationManager()


def reset_all():
    """清除全部内容并重置为默认参数"""
    manager.reset()
    return [], "", model_params["temperature"], model_params["max_tokens"]


def tbm_chat_fn(message, history, temperature, max_tokens):
    """聊天调用函数"""
    history = history or []
    if not message.strip():
        return history, ""

    reply = manager.chat(
        query=message,
        temperature=temperature,
        max_tokens=max_tokens
    )
    history.append([message, reply])
    return history, ""


# Gradio UI 构建
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🛠️ TBM 隧道地质风险防控助手
    <style>
    h1 {
      color: #1a73e8;
      text-align: center;
      margin-bottom: 20px;
    }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 对话记录")
            msg = gr.Textbox(placeholder="请输入您的 TBM 风险问题...", label="用户输入", lines=1)
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清除会话")

        with gr.Column(scale=1):
            gr.Markdown("### 🔧 模型参数调节")
            temperature = gr.Slider(
                0.0, 1.0,
                value=model_params["temperature"],
                step=0.05,
                label="Temperature"
            )
            max_tokens = gr.Slider(
                64, 4096,
                value=model_params["max_tokens"],
                step=64,
                label="Max tokens"
            )

    # 提交按钮和回车都绑定聊天函数
    submit_btn.click(
        fn=tbm_chat_fn,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )

    msg.submit(
        fn=tbm_chat_fn,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )

    # 清除按钮重置聊天、输入框和参数滑块
    clear_btn.click(
        fn=reset_all,
        outputs=[chatbot, msg, temperature, max_tokens]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )
