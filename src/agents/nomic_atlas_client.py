"""Utility helpers for integrating with Nomic Atlas.

This module wraps the official `nomic` SDK to provide a typed, ergonomic
interface for embedding concept strings, creating datasets, and performing
vector search or topic exploration. It is the primary API that other agents
should use when interacting with Atlas.

Example usage:

>>> client = AtlasClient(dataset_name="math-to-manim-knowledge-graph")
>>> client.ensure_dataset()
>>> client.upsert_concepts([
...     AtlasConcept(concept="linear algebra", metadata={"domain": "math"})
... ])
>>> related = client.search_similar("quantum mechanics")

The implementation intentionally avoids importing the `nomic` package at
module import time to allow downstream code to continue functioning when the
SDK is not installed. Consumers should call `AtlasClient.check_install()` (or
handle the `ImportError`) before using Atlas-specific features.

用于与 Nomic Atlas 集成的实用工具函数。
本模块封装了官方的 nomic SDK，提供了一个具有类型提示、符合人体工程学（易用）的接口，用于嵌入概念字符串、创建数据集以及执行向量搜索或主题探索。它是其他代理程序在与 Atlas 交互时应使用的主要 API。
示例用法：

Python
>>> client = AtlasClient(dataset_name="math-to-manim-knowledge-graph")
>>> client.ensure_dataset() # 确保数据集存在
>>> client.upsert_concepts([ # 插入或更新概念
...     AtlasConcept(concept="linear algebra", metadata={"domain": "math"})
... ])
>>> related = client.search_similar("quantum mechanics") # 搜索相似概念
实现上刻意避免在模块导入时导入 nomic 包，以允许在 SDK 未安装时，下游代码仍然可以继续运行。使用者在调用 Atlas 特有的功能之前，应先调用 AtlasClient.check_install()（或处理可能出现的 ImportError 异常）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


class NomicNotInstalledError(RuntimeError):
    """
    Raised when the `nomic` dependency is missing for Atlas operations.
    当缺少 nomic 依赖项以进行 Atlas 操作时引发。
    """


def _import_nomic():
    """
    Import helper that raises a friendly error if `nomic` is missing.
    导入辅助函数，如果缺少 nomic 则会引发友好的错误。
    """

    try:
        from nomic import AtlasDataset, embed  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise NomicNotInstalledError(
            "The `nomic` package is required for Atlas integration. 使用 nomic 包进行 Atlas 集成。"
            "Install it with `pip install nomic` and ensure your 请使用 pip install nomic 进行安装，并确保您的"
            "NOMIC_API_KEY environment variable is set. NOMIC_API_KEY 环境变量已设置。"
        ) from exc

    return AtlasDataset, embed


@dataclass
class AtlasConcept:
    """
    Represents a concept and optional metadata for Atlas storage.
    表示一个概念以及用于 Atlas 存储的可选元数据。
    """

    concept: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_atlas_payload(self) -> Dict[str, Any]:
        """
        Return the data payload expected by Atlas for this concept.
        返回 Atlas 存储此概念所需的数据载荷。
        """

        payload = {"concept": self.concept}
        payload.update(self.metadata)
        return payload


class AtlasClient:
    """
    High level helper for interacting with a Nomic Atlas dataset.
    用于与 Nomic Atlas 数据集交互的高级辅助函数
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        embedding_model: str = "nomic-embed-text-v1.5",
        task_type: str = "clustering",
    ) -> None:
        self.dataset_name = dataset_name
        self.embedding_model = embedding_model
        self.task_type = task_type
        self._dataset = None

    # ------------------------------------------------------------------
    # Installation / health checks
    # ------------------------------------------------------------------
    @staticmethod
    def check_install() -> None:
        """
        Ensure the `nomic` dependency is available.
        确保 nomic 依赖项可用。
        """

        _import_nomic()

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------
    def ensure_dataset(self) -> Any:
        """
        Create or load the configured dataset from Atlas.
        从 Atlas 创建或加载已配置的数据集。
        """

        AtlasDataset, _ = _import_nomic()
        if self._dataset is None:
            self._dataset = AtlasDataset(self.dataset_name)
        return self._dataset

    @property
    def dataset(self) -> Any:
        """
        Return the cached dataset instance (requires `ensure_dataset`).
        返回缓存的数据集实例（需要 ensure_dataset）。
        """

        if self._dataset is None:
            raise RuntimeError(
                "Atlas dataset not initialised. Call `ensure_dataset()` Atlas 数据集未初始化。请调用 ensure_dataset()。"
                "before performing operations. 在执行操作之前。"
            )
        return self._dataset

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for the provided texts.
        为提供的文本生成嵌入。
        """

        _, embed = _import_nomic()
        response = embed.text(
            texts=texts,
            model=self.embedding_model,
            task_type=self.task_type,
        )
        return response["embeddings"]

    def upsert_concepts(self, concepts: Iterable[AtlasConcept]) -> None:
        """
        Embed and upload one or more concepts to the dataset.
        将一个或多个概念嵌入并上传到数据集。
        """

        concepts_list = list(concepts)
        if not concepts_list:
            return

        embeddings = self.embed_texts([c.concept for c in concepts_list])
        self.ensure_dataset().add_data(
            embeddings=embeddings,
            data=[c.as_atlas_payload() for c in concepts_list],
        )

    # ------------------------------------------------------------------
    # Search / topic helpers
    # ------------------------------------------------------------------
    def search_similar(
        self,
        query: str,
        *,
        k: int = 10,
        fields: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector search against the dataset.
        对数据集执行向量搜索。
        """

        dataset = self.ensure_dataset()
        return dataset.vector_search(
            query=query,
            k=k,
            fields=list(fields) if fields is not None else None,
        )

    def list_topics(self) -> Dict[str, Any]:
        """
        Return the hierarchical topics discovered by Atlas.
        返回 Atlas 发现的层次主题。 / 返回 Atlas 发现的分层主题。
        """

        return self.ensure_dataset().topics

    def create_map(
        self,
        *,
        name: Optional[str] = None,
        colorable_fields: Optional[Sequence[str]] = None,
        id_field: Optional[str] = None,
    ) -> Any:
        """
        Create an interactive Atlas map for the dataset.
        为数据集创建一个交互式 Atlas 地图。
        """

        return self.ensure_dataset().create_index(
            name=name,
            colorable_fields=list(colorable_fields) if colorable_fields else None,
            id_field=id_field,
        )


__all__ = ["AtlasClient", "AtlasConcept", "NomicNotInstalledError"]


