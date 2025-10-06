"""Video review agent scaffolding.

This agent is designed to be appended to the Claude Agent SDK pipeline after
`CodeGenerator`. It leverages the existing `tools.video_review_toolkit` module
to automate post-render QA tasks such as frame extraction and HTML5 player
generation.

For now the agent exposes a synchronous `review` method returning a structured
result object. The plan is to wrap this inside the Claude agent runtime in a
future iteration so the VideoReview step can participate in the multi-agent
conversation.

视频审查代理脚手架 (Video Review Agent Scaffolding)
该代理旨在被附加到 Claude Agent SDK 管道中，位于 CodeGenerator 之后。
它利用现有的 tools.video_review_toolkit 模块来自动化渲染后的质量保证 (QA) 任务，例如帧提取和 HTML5 播放器生成。
目前，该代理公开了一个同步的 review 方法，用于返回一个结构化结果对象。
计划是在未来的迭代中将其封装到 Claude 代理运行时内部，以便 VideoReview 步骤可以参与到多代理对话中。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


# Ensure the project root (which contains the `tools` package) is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.video_review_toolkit import VideoReviewToolkit  # noqa: E402  pylint: disable=wrong-import-position


@dataclass
class VideoReviewResult:
    """
    Structured output produced by the VideoReview agent.
    VideoReview 代理生成的结构化输出。
    """

    video_path: Path
    frames_dir: Path
    web_player_path: Optional[Path]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable representation.
        返回一个可进行 JSON 序列化的表示。
        """

        payload = asdict(self)
        payload.update(
            {
                "video_path": str(self.video_path),
                "frames_dir": str(self.frames_dir),
                "web_player_path": str(self.web_player_path) if self.web_player_path else None,
            }
        )
        return payload

    def to_json(self, **dumps_kwargs: Any) -> str:
        """
        Serialize the payload to JSON (useful when returning via SDK).
        将有效载荷序列化为 JSON（在使用 SDK 返回时很有用）。
        """

        return json.dumps(self.to_dict(), **dumps_kwargs)


@dataclass
class VideoReviewConfig:
    """
    Optional configuration for the review step.
    审查步骤的可选配置。
    """

    fps: Optional[float] = None
    every_nth_frame: Optional[int] = 10
    quality: int = 4
    generate_web_player: bool = True
    output_frames_dir: Optional[Path] = None
    output_player_name: Optional[str] = None


class VideoReviewAgent:
    """
    Agent responsible for automating video QA helpers.
    负责自动化视频 QA 辅助工具的代理。
    """

    def __init__(self, toolkit: Optional[VideoReviewToolkit] = None) -> None:
        self.toolkit = toolkit or VideoReviewToolkit()

    """
    运行针对 ``video_path`` 的审查工作流程。
    参数 video_path: 由 CodeGenerator 生成的渲染后的 MP4 文件的绝对或相对路径。
    config: 可选的覆盖配置，用于控制帧采样和播放器生成。
    """
    def review(self, video_path: Path | str, config: Optional[VideoReviewConfig] = None) -> VideoReviewResult:
        """Run the review workflow for ``video_path``.

        Parameters
        ----------
        video_path:
            Absolute or relative path to the rendered MP4 produced by CodeGenerator.
        config:
            Optional overrides controlling frame sampling and player generation.
        """

        config = config or VideoReviewConfig()
        video_path = Path(video_path).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found at {video_path}")

        frames_dir = self.toolkit.extract_frames(
            str(video_path),
            output_dir=str(config.output_frames_dir) if config.output_frames_dir else None,
            fps=config.fps,
            every_nth_frame=config.every_nth_frame,
            quality=config.quality,
        )

        metadata = self.toolkit.get_video_info(str(video_path))

        web_player_path: Optional[Path] = None
        if config.generate_web_player:
            player_name = config.output_player_name or f"{video_path.stem}_review.html"
            web_player_path = self.toolkit.create_web_player(str(video_path), output_html=player_name)

        return VideoReviewResult(
            video_path=video_path,
            frames_dir=frames_dir,
            web_player_path=web_player_path,
            metadata=metadata,
        )


__all__ = ["VideoReviewAgent", "VideoReviewConfig", "VideoReviewResult"]

