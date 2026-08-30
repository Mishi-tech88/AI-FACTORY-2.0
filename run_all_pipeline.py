# run_all_pipeline.py
import subprocess
import sys
import os

stages = [
    ("Stage I - Data Generation", "src/data_generation.py"),
    ("Stage I - Preprocessing", "src/preprocess.py"),
    ("Stage II - Tabular Training", "src/train_tabular.py"),
    ("Stage II - TimeSeries LSTM", "src/train_timeseries.py"),
    ("Stage II - Image CNN", "src/train_image.py"),
    ("Stage III - NLP Enhancement", "src/nlp_pipeline.py"),
    # Stage IV is separate, we'll add later
]

def run_script(script_path):
    print(f"\n▶️ Running: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running {script_path}:")
        print(result.stderr)
        return False
    else:
        print(f"✅ {script_path} completed successfully.")
        # Print last few lines of stdout for feedback
        output_lines = result.stdout.splitlines()
        if output_lines:
            print("\n".join(output_lines[-5:]))
        return True

if __name__ == "__main__":
    print("="*60)
    print("AI FACTORY PIPELINE – Full Run")
    print("="*60)
    success = True
    for name, script in stages:
        if not run_script(script):
            success = False
            break
    if success:
        print("\n🎉 All stages completed successfully!")
    else:
        print("\n❌ Pipeline halted due to error.")