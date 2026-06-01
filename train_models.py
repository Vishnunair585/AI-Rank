"""
train_models.py
Run this script once to train and save all 25 task-specific
Linear Regression models before launching the Streamlit app.

Usage:
    python train_models.py
"""

import time
from utils.ml_model import train_all_models, MODELS_DIR
from utils.visualizer import plot_model_performance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    print("=" * 55)
    print("  AIRank — Training all task models")
    print("=" * 55)

    start = time.time()
    results = train_all_models()
    elapsed = round(time.time() - start, 2)

    print(f"\nTraining complete in {elapsed}s")
    print(f"Models saved to: {MODELS_DIR}\n")

    print(f"{'Task':<28} {'R²':>8} {'MAE':>8} {'CV Mean':>10}")
    print("-" * 58)
    passing = 0
    for task, m in results.items():
        status = "✓" if m["r2"] >= 0.80 else "✗"
        if m["r2"] >= 0.80:
            passing += 1
        print(
            f"{status} {task:<26} {m['r2']:>8.4f} "
            f"{m['mae']:>8.4f} {m['cv_mean']:>10.4f}"
        )

    print("-" * 58)
    print(f"\n{passing}/{len(results)} models meet R² ≥ 0.80 target\n")

    # Save performance chart
    fig = plot_model_performance(results)
    chart_path = "assets/model_performance.png"
    __import__("pathlib").Path("assets").mkdir(exist_ok=True)
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Performance chart saved → {chart_path}")
    print("\nReady! Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
