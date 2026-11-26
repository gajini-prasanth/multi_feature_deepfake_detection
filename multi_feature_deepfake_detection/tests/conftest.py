import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def dummy_batch():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_single():
    return torch.randn(1, 3, 224, 224)
