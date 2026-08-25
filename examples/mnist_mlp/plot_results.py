#!/usr/bin/env python3
"""Plot Step 5.12 MNIST MLP training results saved by examples/mnist_mlp/mnist_mlp.cpp."""
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
    plt.plot(epochs, losses, label="Cross-entropy loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST MLP training loss")
    plt.legend()
    plt.tight_layout()
    out = HERE / "loss_curve.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    loss_rows = read_csv(HERE / "loss_history.csv")
    plot_loss(loss_rows)
