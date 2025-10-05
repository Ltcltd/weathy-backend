import subprocess
import sys
from typing import List

# Define the sequence of scripts to run
SCRIPTS_TO_RUN: List[str] = [
    "scripts/data/download_all.py",
    "scripts/preprocessing/preprocess_all.py",
    "scripts/training/train_all.py"
]

def run_script(script_path: str) -> None:
    """Executes a single Python script and checks for errors."""
    print(f"--- Starting: {script_path} ---")

    # Use 'subprocess.run' to execute the script
    # capture_output=False streams output directly to the console
    try:
        result = subprocess.run(
            [sys.executable, script_path],  # Use sys.executable for best practice
            check=True,  # Raise an exception if the script returns a non-zero exit code
            text=True,
            encoding='utf-8'
        )
        print(f"--- Finished: {script_path} (Return code: {result.returncode}) ---")
        print()
    except subprocess.CalledProcessError as e:
        # Handle the error if the script fails (check=True caused this)
        print(f"\n❌ ERROR: Script failed: {script_path}")
        print(f"Return Code: {e.returncode}")
        # Optionally print stdout/stderr from the failed process
        if e.stdout:
            print("--- Script STDOUT ---")
            print(e.stdout)
        if e.stderr:
            print("--- Script STDERR ---")
            print(e.stderr)

        # Stop the pipeline
        print("\nStopping pipeline due to error.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Python interpreter not found or script path is wrong: {script_path}")
        sys.exit(1)


if __name__ == "__main__":
    print("--- Starting ML Pipeline ---")
    
    # Iterate through the list and run each script sequentially
    for script in SCRIPTS_TO_RUN:
        run_script(script)

    print("\n--- Pipeline finished successfully! 🎉 ---")