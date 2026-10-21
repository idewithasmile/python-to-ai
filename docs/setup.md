# Environment Setup

This guide builds a reproducible local environment for the full curriculum. The complete dependency stack is large because it covers classical machine learning, deep learning, computer vision, NLP, generative AI, graph learning, and deployment. Install only what you need if disk space or bandwidth is limited.

## 1. Prerequisites

| Requirement | Recommended |
| :--- | :--- |
| Python | 3.10 or newer; 3.12 is a conservative choice |
| Git | Current stable release |
| RAM | 16 GB or more for deep learning |
| Disk | 20 GB free for dependencies, datasets, and model caches |
| GPU | Optional NVIDIA GPU with a current driver |
| Editor | VS Code, PyCharm, or JupyterLab |

Verify the tools:

```powershell
python --version
git --version
```

## 2. Clone the Repository

```powershell
git clone https://github.com/idewithasmile/python-to-ai.git
Set-Location python-to-ai
```

## 3. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked by the execution policy, allow scripts for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirm that the active interpreter is inside the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

## 4. Install Dependencies

### Full Learning Stack

```powershell
python -m pip install -r requirements.txt
```

This installs the libraries used across all ten modules. Expect a large download and several minutes of installation time.

### Lighter Start for Module 01

If you want to begin with Python and data workflows before adding deep learning:

```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab requests beautifulsoup4 pytest ruff
```

Add the remaining packages when you reach the relevant module. The `requirements.txt` file is grouped by module so you can install individual tracks without guessing.

### Optional GPU and Quantization Packages

The default PyTorch wheel may be appropriate for many systems, but GPU support depends on your operating system, driver, and CUDA version. If `torch.cuda.is_available()` is `False` on a supported NVIDIA system, install the build recommended by the official PyTorch installation selector at <https://pytorch.org/get-started/locally/>.

`bitsandbytes` is intentionally optional because QLoRA support depends on the platform and CUDA environment. Install it only when a Module 09 lab requires it:

```powershell
python -m pip install "bitsandbytes>=0.45,<1"
```

## 5. Register the Jupyter Kernel

```powershell
python -m ipykernel install --user --name python-to-ai --display-name "Python (python-to-ai)"
```

Start JupyterLab from the repository root:

```powershell
python -m jupyter lab
```

Select the **Python (python-to-ai)** kernel for all notebooks.

## 6. Verify the Environment

Run the environment check:

```powershell
python scripts/verify_environment.py
```

The script reports Python, core scientific packages, deep learning packages, computer vision packages, NLP packages, and CUDA availability. A healthy Module 01 environment has all core packages marked as available.

Run the structural tests:

```powershell
python -m pytest
```

Check code quality when you add Python files:

```powershell
python -m ruff check .
python -m ruff format --check .
```

Optional pre-commit hooks:

```powershell
python -m pre_commit install
```

## 7. Configure Data and Model Caches

Datasets, model weights, and experiment artifacts should stay outside Git. The repository ignores `data/`, `models/`, `runs/`, `outputs/`, and common model formats.

A useful local layout:

```text
data/
├── raw/
├── interim/
├── processed/
└── samples/
```

To keep Hugging Face downloads inside the project workspace:

```powershell
$env:HF_HOME = "$PWD\.cache\huggingface"
```

On macOS or Linux:

```bash
export HF_HOME="$PWD/.cache/huggingface"
```

The `.cache/` directory is ignored by Git.

## 8. Reproducibility Checklist

Before starting a module, record the environment:

```powershell
python --version
python -m pip freeze > requirements-lock.txt
```

Every experiment should also record:

- Random seeds for Python, NumPy, and PyTorch.
- Train, validation, and test split strategy.
- Dataset version or access date.
- Hardware and runtime.
- Exact command used to train and evaluate.
- Metric definitions and final results.

## 9. Troubleshooting

### `python` is not recognized

Reinstall Python with the option to add it to `PATH`, or use the full path to the interpreter. Confirm with `python --version`.

### PowerShell blocks activation

Use a process-scoped execution policy as shown in Section 3. Avoid changing the machine-wide policy unless you understand the security implications.

### PyTorch does not detect the GPU

Check the NVIDIA driver first, then install the PyTorch build that matches the installed CUDA runtime. Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### CUDA runs out of memory

Reduce batch size, sequence length, or image resolution. Enable gradient accumulation or mixed precision. Clear unused GPU memory between experiments. For small labs, CPU execution is acceptable.

### Jupyter cannot find the kernel

Re-run the kernel registration command from Section 5 and confirm that JupyterLab was started from the active virtual environment.

### A package fails to install on Windows

Some optional packages require platform-specific wheels. Install the core stack first and defer the optional package until its module. Record the failure and the workaround in the module notes rather than changing unrelated dependencies.

### Disk space fills during model downloads

Model caches can grow quickly. Remove unused checkpoints and datasets, keep only the models required for the current module, and set `HF_HOME` to a drive with enough free space.

### SSL or proxy errors

Configure your organization's certificate or proxy settings for `pip` and Git, or use an approved package mirror. Do not disable certificate verification globally.

## 10. Daily Workflow

```powershell
.\.venv\Scripts\Activate.ps1
git pull
python -m jupyter lab
```

At the end of a session:

```powershell
python -m pytest
python -m ruff check .
git status
```

Commit only source, notebooks, documentation, and lightweight results. Never commit datasets larger than 50 MB, model weights, secrets, or generated media.
