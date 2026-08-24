#!/usr/bin/env python3
"""Plot Step 5.11 linear-regression results saved by examples/linear_regression.cpp."""
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

HERE = Path(__file__).parent


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def plot_loss(rows):
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, label="MSE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    out = HERE / "loss_curve.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_predictions(rows):
    xs = [float(r["x"]) for r in rows]
    y_true = [float(r["y_true"]) for r in rows]
    y_pred = [float(r["y_pred"]) for r in rows]

    sorted_pairs = sorted(zip(xs, y_true, y_pred))
    xs = [p[0] for p in sorted_pairs]
    y_true = [p[1] for p in sorted_pairs]
    y_pred = [p[2] for p in sorted_pairs]

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, y_true, s=15, label="Target", alpha=0.6)
    plt.plot(xs, y_pred, color="red", linewidth=2, label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression: predictions vs targets")
    plt.legend()
    plt.tight_layout()
    out = HERE / "predictions.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    loss_rows = read_csv(HERE / "loss_history.csv")
    plot_loss(loss_rows)

    pred_rows = read_csv(HERE / "predictions.csv")
    plot_predictions(pred_rows)
