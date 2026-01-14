import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
