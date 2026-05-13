"""Entry point for the Finnie AI Finance Assistant Streamlit app."""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "src" / "web_app" / "app.py"
    sys.exit(
        subprocess.call(
            ["streamlit", "run", str(app_path), "--server.headless", "false"],
            cwd=str(Path(__file__).parent),
        )
    )
