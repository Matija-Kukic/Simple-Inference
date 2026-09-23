from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS = DATA_DIR / "qwen3-weights/"

OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

def set_symlink():
    target = Path(HF_HUB_CACHE).resolve() / "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
    link = WEIGHTS  

    if not link.exists():
        link.symlink_to(target, target_is_directory=True)

    print(f"{link} -> {target}")
