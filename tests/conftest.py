"""Pytest configuration and shared fixtures for SDA tests."""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Compute device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_tensor():
    """Sample tensor for testing."""
    return torch.randn(4, 2, 32, 32)  # (B, C, H, W)


@pytest.fixture
def sample_timeseries():
    """Sample time series tensor for testing."""
    return torch.randn(4, 10, 2, 32, 32)  # (B, L, C, H, W)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def skip_mcs():
    """Skip MCS module if JAX is not available."""
    os.environ["SDA_SKIP_MCS"] = "1"
    yield
    os.environ.pop("SDA_SKIP_MCS", None)
