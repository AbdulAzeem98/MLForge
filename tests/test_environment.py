"""Unit test suite verifying python environment and core dependencies."""

import importlib
import sys
import pytest


def test_python_version():
    """Verify that Python version is 3.11."""
    major, minor = sys.version_info[:2]
    assert (major, minor) == (3, 11), f"Expected Python 3.11, found {major}.{minor}"


@pytest.mark.parametrize(
    "package_name",
    [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "matplotlib",
        "seaborn",
        "mlflow",
        "fastapi",
        "pytest",
    ],
)
def test_package_import(package_name):
    """Verify core packages can be imported without error."""
    mod = importlib.import_module(package_name)
    assert mod is not None


def test_torch_cpu_configuration():
    """Verify PyTorch is configured for CPU-only execution."""
    import torch

    assert not torch.cuda.is_available(), "CUDA should not be available on CPU-only setup"
    
    # Test CPU tensor calculation
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    assert z.tolist() == [5.0, 7.0, 9.0]
