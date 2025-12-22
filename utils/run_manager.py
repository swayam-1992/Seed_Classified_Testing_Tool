import os
import shutil
from datetime import datetime

TEMP_DIR = "temp_run"
RUNS_DIR = "runs"


def init_temp_run():
    """
    Create a clean temp_run directory.
    """
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def finalize_run():
    """
    Move temp_run to runs/run_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    os.makedirs(RUNS_DIR, exist_ok=True)

    final_path = os.path.join(RUNS_DIR, run_name)
    shutil.move(TEMP_DIR, final_path)

    print(f"✅ Run saved to: {final_path}")
    return final_path

