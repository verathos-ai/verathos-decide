"""Native field arithmetic, sumcheck and BLAKE3 Merkle kernels, built in place by ``verathos-build-native``.

``native`` is None when the extension is not built; callers fall back to NumPy where a fallback exists.
"""

try:
    from verathos.capacity.zk.cuda import _native as native
except ImportError:
    native = None

HAS_NATIVE = native is not None
HAS_CUDA = bool(getattr(native, "HAS_CUDA", False))
