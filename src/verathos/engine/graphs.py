"""CUDA graphs for a fixed-shape forward pass, one graph per (batch, length) bucket.

A pass is replayed from a graph when its padded shape fits the captured buckets, otherwise it runs eagerly. Every graph
reads its inputs from one shared input buffer and writes its output into one shared output buffer (views of the
bucket's shape at the start of each), so the memory held across graphs is the largest bucket's input and output plus
the graphs' shared pool, not one output per bucket. Graphs are captured on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)


def bucket_length(n: int, step: int) -> int:
    return -(-n // step) * step


def bucket_batch(n: int, sizes: tuple[int, ...]) -> int | None:
    return next((size for size in sizes if size >= n), None)


class GraphedForward:
    """``forward(ids [B, T]) -> hidden [B, T, H]`` replayed from CUDA graphs for B in ``batch_sizes`` and T a
    multiple of ``length_step`` up to ``max_length``."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        length_step: int = 128,
        max_length: int = 2048,
        max_tokens: int = 8192,
    ) -> None:
        self.forward = forward
        self.batch_sizes = batch_sizes
        self.length_step = length_step
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.out = torch.empty(max_tokens * hidden_size, dtype=dtype, device=device)
        self.hidden_size = hidden_size
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

    def fits(self, batch: int, length: int) -> bool:
        return length <= self.max_length and batch <= self.batch_sizes[-1] and batch * length <= self.max_tokens

    def _views(self, batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch * length
        return self.ids[:n].view(batch, length), self.out[: n * self.hidden_size].view(batch, length, self.hidden_size)

    def _capture(self, batch: int, length: int) -> torch.cuda.CUDAGraph:
        ids, out = self._views(batch, length)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm up: Triton autotuning and cuBLAS workspaces before capture
            for _ in range(2):
                out.copy_(self.forward(ids))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out.copy_(self.forward(ids))
        log.info("captured a CUDA graph for batch %d x %d tokens", batch, length)
        return graph

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """Hidden states for right-padded ids [B, T] (B and T already bucketed). Graph outputs are views into the shared
        output buffer: read them before the next call."""
        batch, length = ids.shape
        if not self.fits(batch, length):
            return self.forward(ids)
        key = (batch, length)
        if key not in self.graphs:
            self.graphs[key] = self._capture(batch, length)
        static_ids, out = self._views(batch, length)
        static_ids.copy_(ids)
        self.graphs[key].replay()
        return out
