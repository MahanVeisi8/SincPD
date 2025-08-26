from pathlib import Path
import matplotlib.pyplot as plt

def save_curve(values, out: Path, title: str = "Curve", ylabel: str = "Value"):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
