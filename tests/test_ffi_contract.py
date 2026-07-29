import numpy as np
import pytest

from mojonltk._lib import address


def test_address_enforces_dtype_layout_and_writability():
    contiguous = np.arange(8, dtype=np.int64)
    assert address(contiguous, np.int64) == contiguous.ctypes.data

    with pytest.raises(TypeError, match="dtype"):
        address(contiguous, np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        address(contiguous[::2], np.int64)

    contiguous.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        address(contiguous, np.int64, writable=True)


def test_address_rejects_empty_non_abi_buffer():
    with pytest.raises(ValueError, match="non-null storage"):
        address(np.empty(0, dtype=np.int64), np.int64)
