"""CUDA graphs for the small passes of a fixed-shape forward, one graph per (batch, length) bucket.

A pass costs the CPU about the same to launch at any size, so only small passes are limited by the launches: there a
graph replay is several times faster, while a pass of a few thousand tokens keeps the GPU busier than the CPU and runs
as fast eagerly. Graphs therefore cover every bucket of at most ``max_tokens`` tokens, and all of them are captured at
start-up, so serving never stalls on a capture; larger passes run eagerly. If a capture fails, no graph is kept and
every pass runs eagerly.

Every graph reads its inputs from one shared input buffer and writes its output into one shared output buffer (views
of the bucket's shape at the start of each), and the graphs share one memory pool, so the memory held across graphs
is one small pass's, not one per bucket. A forward that also takes each row's length (a noncausal backbone) reads them
from a shared buffer as well.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)


def bucket_length(n: int, step: int) -> int:
    return -(-n // step) * step


class GraphedForward:
    """``forward(ids [B, T]) -> hidden [B, T, H]``, or ``forward(ids, lengths [B])`` with ``with_lengths``, replayed
    from a CUDA graph when B is in ``batch_sizes``, T is a multiple of ``length_step`` and B x T is at most
    ``max_tokens``; eager otherwise."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        batch_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
        length_step: int = 32,
        max_tokens: int = 2048,
        with_lengths: bool = False,
    ) -> None:
        self.forward = forward
        self.batch_sizes = batch_sizes
        self.length_step = length_step
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.out = torch.empty(max_tokens * hidden_size, dtype=dtype, device=device)
        self.lengths = torch.ones(max(batch_sizes), dtype=torch.long, device=device) if with_lengths else None
        self.pool = None
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

    def shapes(self) -> list[tuple[int, int]]:
        """The (batch, length) buckets that get a graph."""
        return [(batch, length) for batch in self.batch_sizes
                for length in range(self.length_step, self.max_tokens // batch + 1, self.length_step)]

    def graphed_batch(self, rows: int, length: int) -> int | None:
        """The graph batch size a pass of ``rows`` rows of ``length`` tokens is padded to, or None to run it eagerly."""
        batch = next((size for size in self.batch_sizes if size >= rows), None)
        return batch if (batch, length) in self.graphs else None

    def _views(self, batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch * length
        return self.ids[:n].view(batch, length), self.out[: n * self.hidden_size].view(batch, length, self.hidden_size)

    def _forward(self, ids: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        return self.forward(ids) if self.lengths is None else self.forward(ids, lengths)

    def _capture(self, batch: int, length: int) -> None:
        ids, out = self._views(batch, length)
        lengths = None if self.lengths is None else self.lengths[:batch].fill_(length)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm up: Triton autotuning and cuBLAS workspaces before capture
            for _ in range(2):
                out.copy_(self._forward(ids, lengths))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out.copy_(self._forward(ids, lengths))
        self.graphs[(batch, length)] = graph

    @torch.inference_mode()
    def capture_all(self) -> bool:
        """Capture every bucket's graph. On a failure (a kernel that cannot be captured, or no memory left for the
        graphs) drop them all, so every pass runs eagerly; False then."""
        start = time.perf_counter()
        try:
            self.pool = torch.cuda.graph_pool_handle()
            for batch, length in self.shapes():
                self._capture(batch, length)
        except Exception as exc:
            self.graphs.clear()
            self.pool = None
            torch.cuda.synchronize()  # a device that fails here too is broken: the engine does not start
            torch.cuda.empty_cache()
            log.warning("capturing CUDA graphs failed (%s: %s); every pass runs eagerly, which is slower for small "
                        "batches", type(exc).__name__, exc)
            return False
        log.info("captured %d CUDA graphs (passes up to %d tokens) in %.0f s", len(self.graphs), self.max_tokens,
                 time.perf_counter() - start)
        return True

    def __call__(self, ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Hidden states for right-padded ids [B, T] (and each row's length [B], with ``with_lengths``). A graph's
        output is a view into the shared output buffer: read it before the next call."""
        graph = self.graphs.get(tuple(ids.shape))
        if graph is None:
            return self._forward(ids, lengths)
        static_ids, out = self._views(*ids.shape)
        static_ids.copy_(ids)
        if self.lengths is not None:
            self.lengths[: ids.shape[0]].copy_(lengths)
        graph.replay()
        return out
