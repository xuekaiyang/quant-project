"""一键启动 Streamlit UI。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "qflab" / "ui" / "streamlit_app.py"


def main() -> None:
    if not APP.exists():
        raise FileNotFoundError(f"Streamlit app not found: {APP}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP)]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
