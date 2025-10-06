"""
Math-To-Manim Web Interface
Powered by Claude Sonnet 4.5 and the Claude Agent SDK
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import gradio as gr
from anthropic import Anthropic

# 本地代理导入（支持包和脚本执行）
try:  # pragma: no cover - import shim
    from .agents import VideoReviewAgent, VideoReviewResult  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback when running as `python src/app_claude.py`
    import sys

    SRC_DIR = Path(__file__).resolve().parent
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from agents import VideoReviewAgent, VideoReviewResult  # type: ignore[import]

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client for Claude Sonnet 4.5
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Model configuration
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Latest Sonnet 4.5

# 可选的视频审查代理（在生成输出后使用）
video_review_agent: Optional[VideoReviewAgent] = None

# Verify API key is present
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please check your .env file.")


def format_latex(text):
    """Format inline LaTeX expressions for proper rendering in Gradio."""
    # 将单美元符号替换为双美元符号以便更好地显示
    lines = text.split('\n')
    formatted_lines = []

    for line in lines:
        # 跳过已经有双美元符号的行
        if '$$' in line:
            formatted_lines.append(line)
            continue

        # 格式化单美元符号表达式
        in_math = False
        new_line = ''
        for i, char in enumerate(line):
            if char == '$' and (i == 0 or line[i-1] != '\\'):
                in_math = not in_math
                new_line += '$$' if in_math else '$$'
            else:
                new_line += char
        formatted_lines.append(new_line)

    return '\n'.join(formatted_lines)



def process_simple_prompt(simple_prompt):
    """
    Process a simple prompt using Claude to create a detailed Manim prompt.
    使用 Claude 处理一个简单提示词，以创建一个详细的 Manim 提示词。

    This will eventually use the reverse knowledge tree system,
    but for now provides a template-based expansion.
    这最终将使用逆向知识树系统，但目前提供基于模板的扩展。
    """
    """
    系统提示词：
    您是一位创建用于 Manim 动画的详细、富含 LaTeX 的提示词的专家。
    将用户的简单描述转换成一个全面的、超过 2000 个 token 的提示词，该提示词应：
    指定每一个视觉元素（颜色、位置、大小）
    为所有方程式使用正确的 LaTeX 格式
    提供顺序指令（“首先...”、“接着...”、“然后...”）
    保持场景之间的视觉连续性
    包含时间信息
    指定摄像机运动
    一致地对数学对象进行颜色编码
    输出必须足够详细，以便 AI 能够生成可运行的 Manim Community Edition 代码。
    """
    system_prompt = """You are an expert at creating detailed, LaTeX-rich prompts for Manim animations.

Transform the user's simple description into a comprehensive, 2000+ token prompt that:
1. Specifies every visual element (colors, positions, sizes)
2. Uses proper LaTeX formatting for all equations
3. Provides sequential instructions ("Begin by...", "Next...", "Then...")
4. Maintains visual continuity between scenes
5. Includes timing information
6. Specifies camera movements
7. Color-codes mathematical objects consistently

The output should be detailed enough for an AI to generate working Manim Community Edition code."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0.7,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Create a detailed Manim animation prompt for: {simple_prompt}"
            }]
        )

        return format_latex(response.content[0].text)
    except Exception as e:
        return f"Error: {str(e)}"


def run_video_review(video_path: str) -> str:
    """
    Invoke the prototype VideoReview agent on a rendered video.
    在一个已渲染的视频上调用原型 VideoReview 代理。
    """

    global video_review_agent  # noqa: PLW0603

    if video_review_agent is None:
        video_review_agent = VideoReviewAgent()

    try:
        result: VideoReviewResult = video_review_agent.review(Path(video_path))
        return (
            "Video review completed.\n\n"
            f"Frames directory: {result.frames_dir}\n"
            f"Web player: {result.web_player_path}\n"
            f"Metadata: {result.metadata}\n"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Video review failed: {exc}"


def chat_with_claude(message, history):
    """
    Chat with Claude Sonnet 4.5 for generating Manim code or discussing concepts.
    与 Claude Sonnet 4.5 聊天，用于生成 Manim 代码或讨论概念。
    """

    # 将历史记录转换为 API 期望的格式
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    """
    你是一个专业的 Manim 动画师和数学教育者。
    你可以帮助用户：
    理解数学概念
    生成 Manim Community Edition 代码用于制作动画
    创建详细的动画提示词
    调试 Manim 代码问题
    为数学思想提出视觉表现建议
    在生成 Manim 代码时，我会：
    使用适当的导入：from manim import *
    定义带有 construct() 方法的 Scene 类
    使用 LaTeX 表示数学表达式（原始字符串 r""）
    提供注释解释动画逻辑
    使用适当的颜色和定位
    包含时间信息（wait、play 的持续时间）
    你总是会用适当的转义来格式化 LaTeX，并使用 MathTex() 来处理方程式。
    """

    system_prompt = """You are an expert Manim animator and mathematics educator.

You help users:
1. Understand mathematical concepts
2. Generate Manim Community Edition code for animations
3. Create detailed animation prompts
4. Debug Manim code issues
5. Suggest visual representations for mathematical ideas

When generating Manim code:
- Use proper imports: from manim import *
- Define Scene classes with construct() method
- Use LaTeX for mathematical expressions (raw strings)
- Provide comments explaining the animation logic
- Use appropriate colors and positioning
- Include timing information (wait, play durations)

Always format LaTeX with proper escaping and use MathTex() for equations."""


    # 呼叫 Claude API
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            temperature=0.7,
            system=system_prompt,
            messages=messages
        )

        answer = format_latex(response.content[0].text)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


# 创建带有多模式标签页的 Gradio 界面
with gr.Blocks(theme="soft", title="Math-To-Manim - Claude Sonnet 4.5") as iface:
    gr.Markdown("# Math-To-Manim Generator")
    gr.Markdown("*由 Claude Sonnet 4.5 和 Claude Agent SDK 提供支持*")

    with gr.Tab("标准模式"):
        gr.Markdown("""
        ### 与 Claude Sonnet 4.5 聊天

        获取以下帮助：

        理解数学概念

        生成 Manim 代码

        创建动画创意

        调试问题

        Claude 已经针对数学可视化和 Manim 代码生成进行了优化。
        """)

        chat_interface = gr.ChatInterface(
            chat_with_claude,
            examples=[
                "生成 Manim 代码来可视化勾股定理",
                "解释如何在 Manim 中制作傅里叶级数的动画",
                "创建一个旋转环面的 3D 可视化",
                "向我展示如何使用适当的 LaTeX 来显示数学方程"
            ],
            title="",
            description=""
        )

    with gr.Tab("提示词扩展器"):
        gr.Markdown("""
            ### 将简单的想法转化为详细的提示词
                    
            该模式会接收您简单的描述，并将其扩展为一个全面的、富含 LaTeX 的提示词，适用于生成高质量的 Manim 动画。

            未来：这将使用逆向知识树系统（reverse knowledge tree system）来构建动画，从基础概念直到高级主题。
        """)

        simple_input = gr.Textbox(
            label="简单描述",
            placeholder="示例：用视觉证明展示勾股定理",
            lines=3
        )
        simple_submit = gr.Button("扩展提示词", variant="primary")
        detailed_output = gr.Textbox(
            label="详细的 Manim 提示词",
            lines=15
        )

        simple_submit.click(
            fn=process_simple_prompt,
            inputs=simple_input,
            outputs=detailed_output
        )

        gr.Examples(
            examples=[
                "可视化量子纠缠",
                "用动画解释傅里叶变换",
                "几何地展示微积分导数是如何工作的",
                "对特征向量和特征值的概念进行动画演示"
            ],
            inputs=simple_input
        )

    with gr.Tab("知识树 (即将推出)"):
        gr.Markdown("""
        ### 逆向知识树系统
                    
        这种革命性的方法将：

        分析您的提问（“解释宇宙学”）

        递归分解概念，通过提问：

        “要理解宇宙学，我必须首先了解什么？”

        “要理解广义相对论，我必须首先了解什么？”

        持续进行直到达到基础概念（高中水平）

        从基础开始，逐步构建到目标概念

        生成动画，从零开始进行教学

        状态：架构已设计，正在实施中

        技术支持：由 Claude Sonnet 4.5 卓越的推理能力提供支持

        请查看 prerequisite_explorer_claude.py 以获取工作原型。
        """)

        gr.Image(
            value=None,
            label="知识树可视化 (即将推出)",
            interactive=False
        )

    with gr.Tab("视频审查 (原型)"):
        gr.Markdown("""
        ### Automate Post-Render QA (Prototype)

        一旦您的动画渲染成 MP4 文件，您可以将其指向 VideoReview 代理。

        该代理将：

        将帧提取到 media/review_frames/<scene>/ 目录下

        生成一个 HTML5 审查播放器

        从 ffprobe 收集视频元数据

        此标签页目前直接调用该代理；很快它将在管道结束时自动运行。
        """)

        review_input = gr.Textbox(
            label="渲染后的 MP4 路径",
            placeholder="media/videos/bhaskara_epic_manim/480p15/BhaskaraEpic.mp4",
            lines=1,
        )
        review_button = gr.Button("运行视频审查", variant="primary")
        review_output = gr.Textbox(label="Agent Output", lines=6)

        review_button.click(fn=run_video_review, inputs=review_input, outputs=review_output)

    with gr.Tab("关于"):
        gr.Markdown("""
        ## Math-To-Manim

        使用 AI 驱动的生成，将数学概念转化为精美的动画。

        技术栈
        AI 模型：Claude Sonnet 4.5（最新，2025 年 10 月）

        代理框架：Claude Agent SDK

        动画：Manim Community Edition v0.19.0

        界面：Gradio

        关键创新：逆向知识树
        与需要训练数据的传统 AI 系统不同，我们的方法使用递归概念分解：

        提问：“在理解 X 之前，我必须理解什么？”

        从基础构建一个完整的知识树

        渐进式生成教学动画

        无需训练数据——纯粹推理！
        ### Resources

        - [GitHub Repository](https://github.com/HarleyCoops/Math-To-Manim)
        - [Documentation](docs/README.md)
        - [Roadmap](ROADMAP.md)
        - [Reverse Knowledge Tree Spec](REVERSE_KNOWLEDGE_TREE.md)

        ### 系统要求

        - Python 3.10+
        - FFmpeg (for video rendering)
        - Node.js (for Claude Agent SDK)
        - LaTeX distribution (for study notes)

        ### Environment Variables Required

        ```bash
        ANTHROPIC_API_KEY=your_claude_api_key_here
        ```

        Get your API key from: [https://console.anthropic.com/](https://console.anthropic.com/)
        """)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║             Math-To-Manim Web Interface                           ║
║                                                                   ║
║  Powered by: Claude Sonnet 4.5 (claude-sonnet-4.5-20251022)       ║
║  Framework: Claude Agent SDK                                      ║
║                                                                   ║
║  Starting Gradio interface...                                     ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    iface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
