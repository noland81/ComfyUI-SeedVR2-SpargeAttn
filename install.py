"""
Auto-install script for ComfyUI-SeedVR2-SpargeAttn.

Runs automatically when ComfyUI loads this custom node for the first time.
Installs SpargeAttn and its dependencies (SageAttention 2, ninja).

If installation fails (no CUDA toolkit, incompatible GPU, etc.),
the node will still work — it falls back to PyTorch SDPA attention.
"""

import subprocess
import sys
import importlib

def pip_install(*args):
    """Run pip install with given arguments."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def is_installed(package_name):
    """Check if a Python package is importable."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install():
    print("\n[SeedVR2-SpargeAttn] Checking dependencies...")

    # 1. ninja (build tool for CUDA extensions)
    if not is_installed("ninja"):
        print("[SeedVR2-SpargeAttn] Installing ninja (build tool)...")
        try:
            pip_install("ninja")
        except Exception as e:
            print(f"[SeedVR2-SpargeAttn] WARNING: Failed to install ninja: {e}")

    # 2. SageAttention 2 (required by SpargeAttn)
    if not is_installed("sageattention"):
        print("[SeedVR2-SpargeAttn] Installing SageAttention 2...")
        try:
            pip_install("sageattention")
            print("[SeedVR2-SpargeAttn] SageAttention 2 installed successfully.")
        except Exception as e:
            print(f"[SeedVR2-SpargeAttn] WARNING: Failed to install SageAttention 2: {e}")
            print("[SeedVR2-SpargeAttn] SpargeAttn requires SageAttention 2. Falling back to SDPA.")
            return

    # 3. SpargeAttn (Sparse + Sage Attention)
    if not is_installed("spas_sage_attn"):
        print("[SeedVR2-SpargeAttn] Installing SpargeAttn from source (this may take a few minutes)...")
        try:
            pip_install("--no-build-isolation", "git+https://github.com/thu-ml/SpargeAttn.git")
            print("[SeedVR2-SpargeAttn] SpargeAttn installed successfully.")
        except Exception as e:
            print(f"[SeedVR2-SpargeAttn] WARNING: Failed to install SpargeAttn: {e}")
            print("[SeedVR2-SpargeAttn] This is OK — the node will use SageAttention 2 or SDPA as fallback.")
            return

    print("[SeedVR2-SpargeAttn] All dependencies ready.")

install()
