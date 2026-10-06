"""
Merkle tree implementation for tensor commitments.

Supports:
- Block-wise commitment (for large matrices)
- Efficient path extraction
- Domain-separated hashing
"""

from typing import List, Tuple, Optional
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

from verathos.zk.types import MerklePath
from verathos.zk.crypto.hash_utils import hash_digest, HASH_NAME


# Try to import CUDA BLAKE3 support
_HAS_CUDA_BLAKE3 = False
try:
    from verathos.zk.cuda import zkllm_native as _native
    if _native is None:
        raise ImportError("zkllm_native unavailable")
    _HAS_CUDA_BLAKE3 = (
        getattr(_native, 'HAS_CUDA', False)
        and hasattr(_native, 'cuda_blake3_merkle_leaves')
        and HASH_NAME == "blake3"  # GPU path must match CPU hash function
    )
except ImportError:
    try:
        import sys
        from pathlib import Path
        _cuda_path = Path(__file__).parent.parent / "cuda"
        if str(_cuda_path) not in sys.path:
            sys.path.insert(0, str(_cuda_path))
        import zkllm_native as _native
        _HAS_CUDA_BLAKE3 = (
            getattr(_native, 'HAS_CUDA', False)
            and hasattr(_native, 'cuda_blake3_merkle_leaves')
            and HASH_NAME == "blake3"
        )
    except ImportError:
        pass


# Domain separation tags
DOMAIN_LEAF = b"VERILLM_LEAF_V1"
DOMAIN_NODE = b"VERILLM_NODE_V1"
DOMAIN_BLOCK = b"VERILLM_BLOCK_V1"
DOMAIN_FLAT_CHUNK = b"VERILLM_FLAT_CHUNK_V1"


def hash_leaf(data: bytes) -> bytes:
    """Hash leaf data with domain separation."""
    return hash_digest(DOMAIN_LEAF + data)


def hash_node(left: bytes, right: bytes) -> bytes:
    """Hash two child nodes with domain separation."""
    return hash_digest(DOMAIN_NODE + left + right)


def hash_block(block_data: bytes) -> bytes:
    """Hash block content with domain separation."""
    return hash_digest(DOMAIN_BLOCK + block_data)


def hash_flat_chunk(chunk_data: bytes) -> bytes:
    """Hash flat weight chunk with domain separation."""
    return hash_digest(DOMAIN_FLAT_CHUNK + chunk_data)


_HASH_SIZE = 32  # BLAKE3 / SHA-256 output size

# Shared BULK chunk-hash pool (see FlatWeightMerkle._hash_chunks_parallel):
# one process-wide executor instead of a pool per tensor. Sized once from
# the first request; the env cap makes that deterministic per process.
# ONLY bulk builders (cache warm, manifest builds) use it - a
# latency-critical per-draw tree submitted here would queue behind
# hundreds of bulk batches (live 2026-08-12: a gate's hard probe timed
# out waiting for the background cache warm to drain).
_HASH_POOL = None
_HASH_POOL_LOCK = None
_BULK_HASHING = None  # threading.local, created lazily


def _shared_hash_pool(num_workers: int):
    global _HASH_POOL, _HASH_POOL_LOCK
    if _HASH_POOL is not None:
        return _HASH_POOL
    import threading

    if _HASH_POOL_LOCK is None:
        _HASH_POOL_LOCK = threading.Lock()
    with _HASH_POOL_LOCK:
        if _HASH_POOL is None:
            from concurrent.futures import ThreadPoolExecutor

            _HASH_POOL = ThreadPoolExecutor(
                max_workers=max(1, int(num_workers)),
                thread_name_prefix="merkle-chunk-hash",
            )
    return _HASH_POOL


def _bulk_hashing_state():
    global _BULK_HASHING
    if _BULK_HASHING is None:
        import threading

        _BULK_HASHING = threading.local()
    return _BULK_HASHING


class bulk_chunk_hashing:
    """Mark this thread's tree builds as BULK work.

    Bulk builds (proof-weight-cache warm, gguf-manifest builds) hash
    hundreds of tensors and share one process-wide pool; everything else
    hashes sequentially, so a per-draw proof tree can never queue behind
    a background warm, and no per-tensor pool churn exists to deadlock
    (both failure modes live-observed 2026-08-12 on a rented 5090 box).
    """

    def __enter__(self):
        _bulk_hashing_state().active = True
        return self

    def __exit__(self, *exc_info):
        _bulk_hashing_state().active = False
        return False


class MerkleTree:
    """
    Merkle tree with compact flat storage.

    All hashes are stored in a single bytearray (32 bytes per node)
    instead of individual Python bytes objects.  This saves ~50% RAM
    on large trees (e.g. 24k nodes: 768KB compact vs 1.7MB List[bytes]).

    Layout: [leaves | level1 | level2 | ... | root]

    For odd number of nodes at a level, the last node
    is paired with itself (odd-dup strategy).
    """

    __slots__ = ('num_leaves', '_data', '_node_count', 'level_offsets')

    def __init__(self, leaves: List[bytes]):
        """Build Merkle tree from leaves."""
        if not leaves:
            raise ValueError("Cannot build tree from empty leaves")

        self.num_leaves = len(leaves)
        self._data = bytearray()
        self._node_count = 0
        self.level_offsets: List[int] = []

        self._build_tree(leaves)

    def _append_hash(self, h: bytes) -> None:
        """Append a single 32-byte hash to the flat buffer."""
        self._data.extend(h)
        self._node_count += 1

    def _get_hash(self, idx: int) -> bytes:
        """Get the 32-byte hash at node index idx."""
        start = idx * _HASH_SIZE
        return bytes(self._data[start:start + _HASH_SIZE])

    def _build_tree(self, leaves: List[bytes]) -> None:
        """Build the tree bottom-up."""
        # Hash leaves
        current_level = [hash_leaf(leaf) for leaf in leaves]
        for h in current_level:
            self._append_hash(h)
        self.level_offsets.append(0)

        # Build upper levels
        while len(current_level) > 1:
            self.level_offsets.append(self._node_count)
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                h = hash_node(left, right)
                next_level.append(h)
                self._append_hash(h)

            current_level = next_level

        self.level_offsets.append(self._node_count)

    @property
    def root(self) -> bytes:
        """Get the Merkle root."""
        return self._get_hash(self._node_count - 1)

    @property
    def num_levels(self) -> int:
        """Number of levels in the tree."""
        return len(self.level_offsets) - 1

    def get_leaf(self, index: int) -> bytes:
        """Get leaf hash at index."""
        if index < 0 or index >= self.num_leaves:
            raise IndexError(f"Leaf index {index} out of range [0, {self.num_leaves})")
        return self._get_hash(index)

    def get_node_hash(self, level: int, index: int) -> bytes:
        """Return one canonical node by level-local coordinate."""

        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level < 0
            or level >= self.num_levels
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
        ):
            raise IndexError("Merkle node coordinate is out of range")
        level_start = self.level_offsets[level]
        level_size = self.level_offsets[level + 1] - level_start
        if index >= level_size:
            raise IndexError("Merkle node coordinate is out of range")
        return self._get_hash(level_start + index)

    def get_path(self, leaf_index: int) -> MerklePath:
        """
        Get authentication path for a leaf.

        Returns list of (sibling_hash, is_left) pairs from leaf to root.
        """
        if leaf_index < 0 or leaf_index >= self.num_leaves:
            raise IndexError(f"Leaf index {leaf_index} out of range")

        siblings: List[Tuple[bytes, bool]] = []
        idx = leaf_index

        for level in range(self.num_levels - 1):
            level_start = self.level_offsets[level]
            level_end = self.level_offsets[level + 1]
            level_size = level_end - level_start

            # Find sibling
            if idx % 2 == 0:
                sib_idx = idx + 1
                is_left = False
            else:
                sib_idx = idx - 1
                is_left = True

            # Handle odd-dup: if sibling doesn't exist, use self
            if sib_idx >= level_size:
                sib_idx = idx

            sib_hash = self._get_hash(level_start + sib_idx)
            siblings.append((sib_hash, is_left))

            idx = idx // 2

        return MerklePath(leaf_index=leaf_index, siblings=siblings)

    @classmethod
    def from_saved(cls, tree_data, num_leaves: int,
                   level_offsets: List[int]) -> 'MerkleTree':
        """Reconstruct a MerkleTree from saved data without rehashing.

        tree_data can be either:
        - bytearray/bytes: compact format (new)
        - List[bytes]: legacy format (auto-converted)
        """
        obj = cls.__new__(cls)
        obj.num_leaves = num_leaves
        obj.level_offsets = level_offsets
        if isinstance(tree_data, bytearray):
            # Cache loaders may read a multi-gigabyte compact tree directly
            # into a bytearray.  Take ownership instead of briefly retaining
            # a second full copy during reconstruction.
            obj._data = tree_data
            obj._node_count = len(obj._data) // _HASH_SIZE
        elif isinstance(tree_data, bytes):
            obj._data = bytearray(tree_data)
            obj._node_count = len(obj._data) // _HASH_SIZE
        else:
            # Legacy List[bytes] format — convert to compact
            obj._data = bytearray()
            for h in tree_data:
                obj._data.extend(h)
            obj._node_count = len(tree_data)
        return obj

    @property
    def tree_data_compact(self) -> bytes:
        """Return tree hashes as a single bytes object for serialization."""
        return bytes(self._data)

    @classmethod
    def from_leaf_hashes(cls, leaf_hashes: List[bytes]) -> 'MerkleTree':
        """Build tree from pre-computed leaf hashes (skips hash_leaf step).

        Used by the GPU BLAKE3 path where leaf hashes are already computed
        as hash_leaf(hash_flat_chunk(data)) on GPU.  Only builds internal
        tree nodes (hash_node pairs).

        Args:
            leaf_hashes: List of 32-byte hashes, one per leaf.
        """
        if not leaf_hashes:
            raise ValueError("Cannot build tree from empty leaves")

        obj = cls.__new__(cls)
        obj.num_leaves = len(leaf_hashes)
        obj._data = bytearray()
        obj._node_count = 0
        obj.level_offsets = []

        # Level 0: pre-computed leaf hashes (no hash_leaf applied)
        current_level = list(leaf_hashes)
        for h in current_level:
            obj._data.extend(h)
            obj._node_count += 1
        obj.level_offsets.append(0)

        # Build upper levels (same logic as _build_tree)
        while len(current_level) > 1:
            obj.level_offsets.append(obj._node_count)
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                h = hash_node(left, right)
                next_level.append(h)
                obj._data.extend(h)
                obj._node_count += 1
            current_level = next_level

        obj.level_offsets.append(obj._node_count)
        return obj


def verify_merkle_path(root: bytes, leaf_data: bytes, path: MerklePath) -> bool:
    """
    Verify a Merkle authentication path.

    Args:
        root: Expected Merkle root
        leaf_data: Raw leaf data (will be hashed)
        path: Authentication path

    Returns:
        True if path is valid
    """
    current = hash_leaf(leaf_data)

    for sibling_hash, is_left in path.siblings:
        if is_left:
            # Sibling is left child
            current = hash_node(sibling_hash, current)
        else:
            # Sibling is right child
            current = hash_node(current, sibling_hash)

    return current == root


def verify_merkle_path_with_leaf_hash(
    root: bytes, leaf_hash: bytes, path: MerklePath
) -> bool:
    """
    Verify a Merkle authentication path with pre-computed leaf hash.

    Args:
        root: Expected Merkle root
        leaf_hash: Already-hashed leaf value
        path: Authentication path

    Returns:
        True if path is valid
    """
    current = leaf_hash

    for sibling_hash, is_left in path.siblings:
        if is_left:
            current = hash_node(sibling_hash, current)
        else:
            current = hash_node(current, sibling_hash)

    return current == root


class BlockMerkleTree:
    """
    Merkle tree over blocks of a 2D tensor.

    Used for committing to large matrices block-wise.
    """

    def __init__(
        self, num_block_rows: int, num_block_cols: int, block_hashes: List[List[bytes]]
    ):
        """
        Build tree from block hashes.

        Args:
            num_block_rows: Number of block rows
            num_block_cols: Number of block columns
            block_hashes: 2D list of block hashes [row][col]
        """
        self.num_block_rows = num_block_rows
        self.num_block_cols = num_block_cols
        self.block_hashes = block_hashes

        # Flatten blocks in row-major order
        flat_leaves = []
        for row in block_hashes:
            flat_leaves.extend(row)

        # Build underlying Merkle tree
        # Note: we pass raw hashes, which will be hashed again as leaves
        self._tree = MerkleTree([bh for bh in flat_leaves])

    @property
    def root(self) -> bytes:
        """Get the Merkle root."""
        return self._tree.root

    def get_leaf(self, bi: int, bj: int) -> bytes:
        """Get leaf hash for block (bi, bj)."""
        flat_idx = bi * self.num_block_cols + bj
        return self._tree.get_leaf(flat_idx)

    def get_path(self, bi: int, bj: int) -> MerklePath:
        """Get authentication path for block (bi, bj)."""
        flat_idx = bi * self.num_block_cols + bj
        return self._tree.get_path(flat_idx)


def build_block_merkle(
    tensor: "torch.Tensor", block_size: int = 256
) -> BlockMerkleTree:
    """
    Build block-wise Merkle tree for a 2D tensor.

    Uses ThreadPoolExecutor for parallel hashing when there are many blocks
    (e.g., full fine-tuning with large weight matrices).

    Args:
        tensor: 2D tensor [M, N]
        block_size: Size of each block (default 256)

    Returns:
        BlockMerkleTree with commitment
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")

    M, N = tensor.shape
    num_block_rows = math.ceil(M / block_size)
    num_block_cols = math.ceil(N / block_size)
    total_blocks = num_block_rows * num_block_cols

    # Pre-convert entire tensor to numpy once (avoids per-block conversion)
    tensor_np = tensor.contiguous().numpy().astype('<i8')

    def _hash_one_block(bi_bj):
        bi, bj = bi_bj
        row_start = bi * block_size
        row_end = min(row_start + block_size, M)
        col_start = bj * block_size
        col_end = min(col_start + block_size, N)
        block = tensor_np[row_start:row_end, col_start:col_end]
        return (bi, bj, hash_block(np.ascontiguousarray(block).tobytes()))

    block_indices = [
        (bi, bj)
        for bi in range(num_block_rows)
        for bj in range(num_block_cols)
    ]

    if total_blocks >= 8:
        num_workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_hash_one_block, block_indices))
    else:
        results = [_hash_one_block(idx) for idx in block_indices]

    # Reconstruct 2D block_hashes
    block_hashes: List[List[bytes]] = [[] for _ in range(num_block_rows)]
    for bi, bj, bh in results:
        block_hashes[bi].append(bh)

    return BlockMerkleTree(num_block_rows, num_block_cols, block_hashes)


def commit_tensor(tensor: "torch.Tensor") -> bytes:
    """
    Create a simple hash commitment to a tensor.

    For small tensors or when full Merkle tree is not needed.
    """
    data = tensor.flatten().contiguous().numpy().astype('<i8').tobytes()
    return hash_digest(DOMAIN_BLOCK + data)


class FlatWeightMerkle:
    """
    Flat Merkle tree over a weight matrix using small chunks.

    Instead of BlockMerkleTree's 2D block structure (256x256 = 512KB leaves),
    this flattens the matrix in row-major order and splits into small chunks
    (default 128 elements = 1KB leaves).

    Key design:
    - Serialization: W.numpy().astype('<i8').tobytes() (zero Python loops)
    - Each leaf = chunk_size * 8 bytes of raw little-endian int64
    - Tree: standard MerkleTree over hash_flat_chunk(chunk_bytes) leaves
    - Proof lookup: chunk_idx = (row * num_cols + col) // chunk_size
    """

    def __init__(self, W: "torch.Tensor", chunk_size: int = 128,
                 store_raw: bool = True):
        """
        Build flat Merkle tree for weight matrix W.

        Args:
            W: 2D weight tensor [K, N], int8 or int64 on CPU
            chunk_size: Number of elements per chunk (default 128)
            store_raw: If True, keep raw bytes in RAM for fast get_proof().
                       If False, discard after hashing (use W_tensor param
                       in get_proof() for on-demand extraction).

        Supports both int8 (1 byte/element) and int64 (8 bytes/element).
        The dtype is auto-detected and stored for verification.

        When W is a CUDA int8 tensor and GPU BLAKE3 is available, hashing
        is done entirely on GPU (no CPU serialization or GPU→CPU weight
        transfer).  Falls back to CPU for int64 or when CUDA is unavailable.
        """
        import torch

        if W.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

        self.num_rows = W.shape[0]
        self.num_cols = W.shape[1]
        self.chunk_size = chunk_size
        self.total_elements = self.num_rows * self.num_cols
        self.num_chunks = math.ceil(self.total_elements / chunk_size)

        # ---- GPU fast path: BLAKE3 hashing directly on CUDA ----
        if W.dtype == torch.int8 and _HAS_CUDA_BLAKE3 and W.is_cuda:
            self._bytes_per_element = 1
            self._dtype_code = '<i1'
            self._bytes_per_chunk = chunk_size * self._bytes_per_element

            # Hash all chunks on GPU → returns CPU tensor [num_chunks, 32] uint8
            leaf_hashes_tensor = _native.cuda_blake3_merkle_leaves(
                W.contiguous(),
                chunk_size,
                W.device.index or 0,
            )
            # Convert to list of 32-byte Python bytes for MerkleTree
            leaf_hash_list = [
                bytes(leaf_hashes_tensor[i].numpy())
                for i in range(self.num_chunks)
            ]
            # Build tree from pre-computed leaf hashes (skip hash_leaf)
            self._tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
            self._raw_bytes = None  # No CPU copy during construction
            # If store_raw requested, transfer now (rare — only when store_raw=True)
            if store_raw:
                self._raw_bytes = W.contiguous().cpu().numpy().astype('<i1').tobytes()
            return

        # ---- GPU chunked path: CPU int8 tensor, hash via chunked GPU transfer ----
        # Falls through to CPU path on CUDA OOM (e.g. lm_head after model +
        # KV cache + CUDA graphs fill VRAM).
        if W.dtype == torch.int8 and _HAS_CUDA_BLAKE3 and not W.is_cuda:
            try:
                self._bytes_per_element = 1
                self._dtype_code = '<i1'
                self._bytes_per_chunk = chunk_size * self._bytes_per_element

                leaf_hash_list = _gpu_blake3_leaf_hashes_chunked(
                    W.contiguous().view(-1), chunk_size)
                if not leaf_hash_list:
                    leaf_hash_list = [hash_flat_chunk(b"")]
                self._tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
                self._raw_bytes = None
                if store_raw:
                    self._raw_bytes = W.contiguous().numpy().astype('<i1').tobytes()
                return
            except RuntimeError:
                pass  # GPU OOM — fall through to CPU path

        # ---- CPU path (non-int8 or no CUDA BLAKE3) ----
        # Detect dtype and serialize appropriately
        # int8: 1 byte per element (for bitsandbytes INT8 native mode)
        # int64: 8 bytes per element (for FP16 scaled mode)
        if W.dtype == torch.int8:
            self._bytes_per_element = 1
            self._dtype_code = '<i1'  # little-endian int8
            raw_bytes = W.contiguous().numpy().astype('<i1').tobytes()
        else:
            self._bytes_per_element = 8
            self._dtype_code = '<i8'  # little-endian int64
            raw_bytes = W.contiguous().numpy().astype('<i8').tobytes()

        self._bytes_per_chunk = chunk_size * self._bytes_per_element

        # Hash each chunk - use parallel hashing for large trees
        self._raw_bytes = raw_bytes  # Needed during hashing
        chunk_hashes = self._hash_chunks_parallel()

        # Build standard MerkleTree (will hash_leaf each chunk_hash)
        self._tree = MerkleTree(chunk_hashes)

        # Optionally discard raw bytes to save RAM
        if not store_raw:
            self._raw_bytes = None

    def _hash_chunks_parallel(self) -> list:
        """
        Hash all chunks, using parallel execution for large trees.

        For trees with >= 16 chunks, uses a SHARED ThreadPoolExecutor.
        """
        import os

        # Prepare chunk byte ranges
        chunk_ranges = []
        for i in range(self.num_chunks):
            byte_start = i * self._bytes_per_chunk
            byte_end = min(byte_start + self._bytes_per_chunk, len(self._raw_bytes))
            chunk_ranges.append((byte_start, byte_end))

        # BULK builders (inside a bulk_chunk_hashing context) parallelize
        # through ONE shared module-level pool; everything else hashes
        # sequentially. History, both live-observed 2026-08-12 on a rented
        # 5090 box: a pool PER TENSOR deadlocked around the ~71st pool
        # (workers idle in queue-wait, main blocked in executor.map
        # forever, py-spy verified), and a single shared pool for ALL
        # callers made per-draw proof trees queue behind hundreds of
        # background cache-warm batches, timing out a deploy gate's hard
        # probe. Sequential per-draw hashing is sub-second with native
        # BLAKE3 and immune to both. VERATHOS_MERKLE_HASH_THREADS caps the
        # bulk pool; 1 (or 0) forces sequential everywhere.
        try:
            _cap = int(os.environ.get("VERATHOS_MERKLE_HASH_THREADS", "8") or 8)
        except ValueError:
            _cap = 8
        num_workers = min(os.cpu_count() or 4, max(1, _cap))
        bulk = bool(getattr(_bulk_hashing_state(), "active", False))
        if bulk and num_workers > 1 and self.num_chunks >= 16:
            def hash_chunk(range_tuple):
                start, end = range_tuple
                return hash_flat_chunk(self._raw_bytes[start:end])

            executor = _shared_hash_pool(num_workers)
            chunk_hashes = list(executor.map(hash_chunk, chunk_ranges))
        else:
            # Sequential: small trees, and every latency-critical caller.
            chunk_hashes = [
                hash_flat_chunk(self._raw_bytes[start:end])
                for start, end in chunk_ranges
            ]

        return chunk_hashes

    @property
    def root(self) -> bytes:
        """Get Merkle root."""
        return self._tree.root

    def get_proof(self, chunk_idx: int,
                  W_tensor: "Optional[torch.Tensor]" = None,
                  raw_bytes: "Optional[bytes]" = None) -> Tuple[MerklePath, bytes]:
        """
        Get Merkle proof for a specific chunk.

        Args:
            chunk_idx: Index of the chunk (0-based)
            W_tensor: Optional weight tensor for on-demand chunk extraction.
                      Required when store_raw=False and _raw_bytes is None.
                      Can be on GPU (only the needed chunk is transferred).
            raw_bytes: Optional pre-serialized weight bytes for O(1) extraction.
                       Bypasses numpy conversion entirely.

        Returns:
            (merkle_path, leaf_data) where leaf_data is raw bytes of the chunk
        """
        if chunk_idx < 0 or chunk_idx >= self.num_chunks:
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {self.num_chunks})")

        merkle_path = self._tree.get_path(chunk_idx)

        if self._raw_bytes is not None:
            byte_start = chunk_idx * self._bytes_per_chunk
            byte_end = min(byte_start + self._bytes_per_chunk, len(self._raw_bytes))
            leaf_data = self._raw_bytes[byte_start:byte_end]
        elif raw_bytes is not None:
            byte_start = chunk_idx * self._bytes_per_chunk
            byte_end = min(byte_start + self._bytes_per_chunk, len(raw_bytes))
            leaf_data = raw_bytes[byte_start:byte_end]
        elif W_tensor is not None:
            # On-demand extraction: slice tiny chunk from tensor (~1KB)
            # W_tensor can be 2D (original matrix) or 1D (pre-flattened for perf).
            # Pre-flattening avoids repeated .contiguous() copies on non-contiguous
            # tensors (e.g. 32MB copy per call × 100 calls = 1s bottleneck).
            elem_start = chunk_idx * self.chunk_size
            elem_end = min(elem_start + self.chunk_size, self.total_elements)
            if W_tensor.dim() == 1:
                flat = W_tensor[elem_start:elem_end]
            else:
                flat = W_tensor.contiguous().view(-1)[elem_start:elem_end]
            if flat.is_cuda:
                flat = flat.cpu()
            leaf_data = flat.numpy().astype(self._dtype_code).tobytes()
        else:
            raise ValueError(
                "No _raw_bytes stored and no W_tensor provided. "
                "Either build with store_raw=True or pass W_tensor to get_proof()."
            )

        return merkle_path, leaf_data

    def get_proof_batch(self, chunk_indices: list,
                        W_tensor: "Optional[torch.Tensor]" = None,
                        raw_bytes: "Optional[bytes]" = None) -> list:
        """
        Batch version of get_proof - fetches multiple chunks in one GPU→CPU transfer.

        Instead of 100 individual GPU→CPU syncs (each ~0.05ms overhead),
        this gathers all needed chunks into one contiguous GPU tensor,
        does a SINGLE GPU→CPU transfer, then splits into individual leaf_data.

        Args:
            chunk_indices: List of unique chunk indices to fetch
            W_tensor: 1D pre-flattened weight tensor (GPU or CPU)
            raw_bytes: Optional pre-serialized weight bytes for O(1) extraction.
                       Bypasses numpy conversion entirely.

        Returns:
            List of (merkle_path, leaf_data) tuples, one per chunk_idx
        """
        import torch
        results = []

        if not chunk_indices:
            return results

        # Get all Merkle paths (CPU-only, fast)
        paths = [self._tree.get_path(ci) for ci in chunk_indices]

        if self._raw_bytes is not None:
            for ci, path in zip(chunk_indices, paths):
                byte_start = ci * self._bytes_per_chunk
                byte_end = min(byte_start + self._bytes_per_chunk, len(self._raw_bytes))
                results.append((path, self._raw_bytes[byte_start:byte_end]))
        elif raw_bytes is not None:
            for ci, path in zip(chunk_indices, paths):
                byte_start = ci * self._bytes_per_chunk
                byte_end = min(byte_start + self._bytes_per_chunk, len(raw_bytes))
                results.append((path, raw_bytes[byte_start:byte_end]))
        elif W_tensor is not None:
            if W_tensor.dim() != 1:
                W_tensor = W_tensor.contiguous().view(-1)

            if W_tensor.is_cuda:
                # BATCHED GPU→CPU: Gather all chunks into one tensor, single transfer
                chunk_tensors = []
                chunk_sizes = []
                for ci in chunk_indices:
                    elem_start = ci * self.chunk_size
                    elem_end = min(elem_start + self.chunk_size, self.total_elements)
                    chunk_tensors.append(W_tensor[elem_start:elem_end])
                    chunk_sizes.append(elem_end - elem_start)

                # Single GPU→CPU transfer of all chunks concatenated
                all_chunks = torch.cat(chunk_tensors).cpu().numpy().astype(self._dtype_code)

                # Split back into individual leaf_data
                offset = 0
                for path, size in zip(paths, chunk_sizes):
                    leaf_data = all_chunks[offset:offset + size].tobytes()
                    results.append((path, leaf_data))
                    offset += size
            else:
                # CPU tensor - no batching benefit, just iterate
                for ci, path in zip(chunk_indices, paths):
                    elem_start = ci * self.chunk_size
                    elem_end = min(elem_start + self.chunk_size, self.total_elements)
                    flat = W_tensor[elem_start:elem_end]
                    leaf_data = flat.numpy().astype(self._dtype_code).tobytes()
                    results.append((path, leaf_data))
        else:
            raise ValueError(
                "No _raw_bytes stored and no W_tensor provided."
            )

        return results

    def get_chunk_for_element(self, row: int, col: int) -> int:
        """Get chunk index for a specific matrix element."""
        flat_idx = row * self.num_cols + col
        return flat_idx // self.chunk_size

    def get_cache_data(self) -> dict:
        """Return serializable data for disk caching.

        Does NOT include _raw_bytes — reconstructed trees use
        on-demand extraction via W_tensor in get_proof().
        """
        return {
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'chunk_size': self.chunk_size,
            'total_elements': self.total_elements,
            'num_chunks': self.num_chunks,
            'bytes_per_element': self._bytes_per_element,
            'dtype_code': self._dtype_code,
            'bytes_per_chunk': self._bytes_per_chunk,
            'tree_data': self._tree.tree_data_compact,
            'tree_num_leaves': self._tree.num_leaves,
            'tree_level_offsets': self._tree.level_offsets,
        }

    @classmethod
    def from_cached(cls, data: dict) -> 'FlatWeightMerkle':
        """Reconstruct from cached data without rehashing.

        The returned tree has _raw_bytes=None, so get_proof() requires
        a W_tensor parameter for on-demand chunk extraction.
        """
        obj = cls.__new__(cls)
        obj.num_rows = data['num_rows']
        obj.num_cols = data['num_cols']
        obj.chunk_size = data['chunk_size']
        obj.total_elements = data['total_elements']
        obj.num_chunks = data['num_chunks']
        obj._bytes_per_element = data['bytes_per_element']
        obj._dtype_code = data['dtype_code']
        obj._bytes_per_chunk = data['bytes_per_chunk']
        obj._raw_bytes = None
        obj._tree = MerkleTree.from_saved(
            data['tree_data'],
            data['tree_num_leaves'],
            data['tree_level_offsets'],
        )
        return obj


def _gpu_blake3_leaf_hashes_chunked(
    W_flat: "torch.Tensor", chunk_size: int, slice_bytes: int = 256 * 1024 * 1024,
) -> List[bytes]:
    """Hash a CPU int8 tensor on GPU in fixed-size slices.

    Transfers ``slice_bytes`` (default 256 MiB) at a time to avoid exhausting
    GPU memory regardless of tensor size.  Slice boundaries are aligned to
    ``chunk_size`` so leaf hashes are identical to hashing the whole tensor
    in one shot.

    Returns a list of 32-byte leaf hashes.
    """
    import torch

    total_elements = W_flat.numel()
    # Align slice to chunk boundary so each GPU call produces complete leaves
    elements_per_slice = max(chunk_size, (slice_bytes // 1) // chunk_size * chunk_size)
    device_index = torch.cuda.current_device()
    leaf_hashes: List[bytes] = []

    for start in range(0, total_elements, elements_per_slice):
        end = min(start + elements_per_slice, total_elements)
        gpu_slice = W_flat[start:end].cuda()
        # Reshape to 2D (required by cuda_blake3_merkle_leaves)
        gpu_slice = gpu_slice.unsqueeze(0) if gpu_slice.dim() == 1 else gpu_slice
        n_chunks = math.ceil(gpu_slice.numel() / chunk_size)
        hashes_tensor = _native.cuda_blake3_merkle_leaves(
            gpu_slice.contiguous(), chunk_size, device_index)
        for i in range(n_chunks):
            leaf_hashes.append(bytes(hashes_tensor[i].numpy()))
        del gpu_slice
    return leaf_hashes


def compute_flat_weight_root(W: "torch.Tensor", chunk_size: int = 128) -> bytes:
    """
    Compute the Merkle root for a weight matrix without retaining the tree.

    Memory-efficient: builds tree, extracts root, discards everything.
    Used during miner setup to compute per-layer roots.

    When W is int8 and GPU BLAKE3 is available, hashing is always done on
    GPU — either in one shot (CUDA tensor) or via chunked transfer (CPU
    tensor, 256 MiB slices) so it works regardless of free GPU memory.

    Args:
        W: 2D weight tensor [K, N], int8 or int64 (CPU or CUDA)
        chunk_size: Elements per chunk

    Returns:
        32-byte Merkle root
    """
    import torch

    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    # GPU fast path — tensor already on CUDA
    if W.dtype == torch.int8 and _HAS_CUDA_BLAKE3 and W.is_cuda:
        num_chunks = math.ceil(W.numel() / chunk_size)
        leaf_hashes_tensor = _native.cuda_blake3_merkle_leaves(
            W.contiguous(), chunk_size, W.device.index or 0)
        leaf_hash_list = [
            bytes(leaf_hashes_tensor[i].numpy())
            for i in range(num_chunks)
        ]
        tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
        return tree.root

    # GPU chunked path — CPU int8 tensor, transfer slices to GPU for hashing.
    # Falls through to CPU path on CUDA OOM (e.g. lm_head after model + KV
    # cache + CUDA graphs fill VRAM).
    if W.dtype == torch.int8 and _HAS_CUDA_BLAKE3:
        try:
            leaf_hash_list = _gpu_blake3_leaf_hashes_chunked(
                W.contiguous().view(-1), chunk_size)
            if not leaf_hash_list:
                leaf_hash_list = [hash_flat_chunk(b"")]
            tree = MerkleTree.from_leaf_hashes(leaf_hash_list)
            return tree.root
        except RuntimeError:
            pass  # GPU OOM — fall through to CPU path

    # CPU path (non-int8 or no CUDA BLAKE3)
    if W.dtype == torch.int8:
        raw_bytes = W.contiguous().numpy().astype('<i1').tobytes()
        bytes_per_chunk = chunk_size * 1
    else:
        raw_bytes = W.contiguous().numpy().astype('<i8').tobytes()
        bytes_per_chunk = chunk_size * 8

    chunk_hashes = []
    for i in range(0, len(raw_bytes), bytes_per_chunk):
        chunk_data = raw_bytes[i:min(i + bytes_per_chunk, len(raw_bytes))]
        chunk_hashes.append(hash_flat_chunk(chunk_data))

    if not chunk_hashes:
        chunk_hashes = [hash_flat_chunk(b"")]

    tree = MerkleTree(chunk_hashes)
    return tree.root


def verify_flat_chunk_merkle_path(
    root: bytes, chunk_data: bytes, path: MerklePath
) -> bool:
    """
    Verify a Merkle path for a flat weight chunk.

    The FlatWeightMerkle two-level hashing:
    1. chunk_bytes -> hash_flat_chunk() -> chunk_hash
    2. chunk_hash -> hash_leaf() -> tree leaf (inside MerkleTree)

    So we pass chunk_hash as leaf_data to verify_merkle_path,
    which calls hash_leaf(chunk_hash) internally.
    """
    chunk_hash = hash_flat_chunk(chunk_data)
    return verify_merkle_path(root, chunk_hash, path)
