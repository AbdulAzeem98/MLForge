"""MLForge Environment Verification Script.

Checks Python version, imports and versions of core dependencies,
verifies CPU-only PyTorch build, and performs basic sanity checks.
"""

import sys
import platform
import importlib
from typing import Dict, Any, Tuple


REQUIRED_PACKAGES = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("sklearn", "Scikit-Learn"),
    ("torch", "PyTorch"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("mlflow", "MLflow"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("pytest", "Pytest"),
]


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f" {title}")
    print(f"{'=' * 65}")


def check_python_environment() -> bool:
    print_header("1. Python Environment")
    py_ver = sys.version_info
    ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    exe_path = sys.executable
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    print(f"  Operating System : {os_info}")
    print(f"  Python Version   : {ver_str}")
    print(f"  Python Path      : {exe_path}")

    # Verify Python 3.11
    if (py_ver.major, py_ver.minor) == (3, 11):
        print(f"  Python Status    : [PASS] Python 3.11 detected")
        return True
    else:
        print(f"  Python Status    : [FAIL] Expected Python 3.11, detected {ver_str}")
        return False


def check_packages() -> Tuple[bool, Dict[str, str]]:
    print_header("2. Package Imports & Versions")
    all_passed = True
    installed_versions = {}

    print(f"  {'Package':<15} {'Status':<10} {'Version':<20}")
    print(f"  {'-' * 45}")

    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "unknown")
            installed_versions[module_name] = ver
            print(f"  {display_name:<15} {'[PASS]':<10} {ver:<20}")
        except ImportError as e:
            all_passed = False
            installed_versions[module_name] = "NOT INSTALLED"
            print(f"  {display_name:<15} {'[FAIL]':<10} {str(e):<20}")

    return all_passed, installed_versions


def check_torch_cpu() -> bool:
    print_header("3. PyTorch CPU Configuration Verification")
    try:
        import torch

        print(f"  PyTorch Version      : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda

        print(f"  CUDA Available       : {cuda_available}")
        print(f"  CUDA Version Tag     : {cuda_version}")

        # Check that CUDA is disabled (CPU-only requirement)
        if cuda_available:
            print(f"  GPU / CUDA Check     : [WARN] CUDA is reported available, CPU-only requested.")
        else:
            print(f"  GPU / CUDA Check     : [PASS] CPU execution mode confirmed.")

        # Tensor operation test on CPU
        x = torch.randn(2, 3)
        y = torch.matmul(x, x.T)
        print(f"  CPU Tensor Test      : [PASS] Matrix multiplication executed successfully.")
        print(f"    Input shape: {tuple(x.shape)} -> Output shape: {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"  CPU Tensor Test      : [FAIL] Error: {e}")
        return False


def check_sklearn_numpy_sanity() -> bool:
    print_header("4. Scikit-Learn & NumPy Sanity Test")
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Create small synthetic dataset: y = 2x + 1
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])

        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array([[5.0]]))
        expected = 11.0
        
        diff = abs(pred[0] - expected)
        if diff < 1e-4:
            print(f"  LinearRegression Test: [PASS] Model trained and predicted correctly (y(5) ~ {pred[0]:.2f})")
            return True
        else:
            print(f"  LinearRegression Test: [FAIL] Unexpected prediction {pred[0]}, expected {expected}")
            return False
    except Exception as e:
        print(f"  LinearRegression Test: [FAIL] Error: {e}")
        return False


def check_fastapi_sanity() -> bool:
    print_header("5. FastAPI Sanity Test")
    try:
        from fastapi import FastAPI
        app = FastAPI(title="MLForge Test App")
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy"}
            
        print(f"  FastAPI App Test     : [PASS] App instance created with route: /health")
        return True
    except Exception as e:
        print(f"  FastAPI App Test     : [FAIL] Error: {e}")
        return False


def main():
    print(f"\nMLForge Environment Diagnostic Utility")
    print(f"Antigravity IDE - Workspace: D:\\MLForge")

    py_ok = check_python_environment()
    pkgs_ok, _ = check_packages()
    torch_ok = check_torch_cpu()
    sklearn_ok = check_sklearn_numpy_sanity()
    fastapi_ok = check_fastapi_sanity()

    print_header("Summary")
    all_ok = py_ok and pkgs_ok and torch_ok and sklearn_ok and fastapi_ok
    if all_ok:
        print("  Status: [ALL CHECKS PASSED] Environment is fully ready for MLForge.")
        sys.exit(0)
    else:
        print("  Status: [ISSUES DETECTED] One or more environment checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
