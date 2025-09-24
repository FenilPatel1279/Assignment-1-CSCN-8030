import os
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_dir(path):
    """Ensure the directory exists before saving a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_pairplot(data, save_path="../outputs/charts/pairplot.png"):
    """Pairplot of features colored by wildfire occurrence."""
    _ensure_dir(save_path)
    sns.pairplot(data, hue="Wildfire", diag_kind="kde")
    plt.suptitle("Feature Distributions by Wildfire Occurrence", y=1.02)
    plt.savefig(save_path)
    plt.show()

def plot_correlation_heatmap(data, save_path="../outputs/charts/correlation.png"):
    """Correlation heatmap of features."""
    _ensure_dir(save_path)
    corr = data.corr()
    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.savefig(save_path)
    plt.show()

def plot_probability_scatter(y_test, y_prob, save_path="../outputs/charts/probability.png"):
    """Scatterplot of predicted probabilities vs. test samples."""
    _ensure_dir(save_path)
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(y_prob)), y_prob, c=y_test, cmap="bwr", alpha=0.7)
    plt.axhline(0.5, color="black", linestyle="--", label="Decision Threshold")
    plt.title("Predicted Probability of Wildfire (0=No, 1=Yes)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Probability of Wildfire")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
