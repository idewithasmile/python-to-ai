from __future__ import annotations

import importlib.util
import platform
import sys

MINIMUM_PYTHON = (3, 10)
PACKAGE_GROUPS = {
    "Core": ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn", "jupyterlab"],
    "Deep learning": ["torch", "torchvision", "transformers", "datasets"],
    "Computer vision": ["cv2", "PIL", "albumentations", "timm"],
    "NLP and generative AI": ["evaluate", "peft", "diffusers", "sentence_transformers"],
    "Deployment": ["fastapi", "gradio"],
}


def is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    print(f"Python: {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < MINIMUM_PYTHON:
        print("Python 3.10 or newer is required.")
        return 1

    missing_core: list[str] = []
    for group, modules in PACKAGE_GROUPS.items():
        print(f"\n{group}")
        for module in modules:
            available = is_available(module)
            marker = "x" if available else " "
            print(f"  [{marker}] {module}")
            if group == "Core" and not available:
                missing_core.append(module)

    if is_available("torch"):
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nTorch: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if missing_core:
        print("\nMissing core packages. Install the full stack with:")
        print("  python -m pip install -r requirements.txt")
        return 1

    print("\nCore environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
