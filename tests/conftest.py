import math
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

# Absolute slack used for every float comparison against upstream nltk. The
# upstream formulas are sums of at most three divisions of small integers, so a
# correctly ported kernel lands within a few ulps of 1.0.
FLOAT_ABS = 1e-15


def assert_close(ours, reference, tol=FLOAT_ABS):
    """Compare a similarity score to upstream, including NaN and inf cases."""
    if math.isnan(reference):
        assert math.isnan(ours), (ours, reference)
    elif math.isinf(reference):
        assert ours == reference, (ours, reference)
    else:
        assert ours == pytest.approx(reference, abs=tol), (ours, reference)

