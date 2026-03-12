import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from  logger import logging
import  config as config


def safe_has(df, col):
    return col in df.columns


# =========================
# Figure 2
# =========================

def plot_figure_2(df, output_dir, run_name):

    if not all(c in df.columns for c in ["batch", "precision", "recall", "f1"]):
        logging.warning("Missing columns for Figure 2")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(df["batch"], df["precision"], marker="o", label="Precision")
    plt.plot(df["batch"], df["recall"], marker="s", label="Recall")
    plt.plot(df["batch"], df["f1"], marker="^", label="F1")

    plt.title(f"{run_name} — Batch Metrics")
    plt.xlabel("Batch")
    plt.ylabel("Score")

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "Figure_2_Batch_Metrics.png",
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f"Saved {save_path}")


# =========================
# Figure 3
# =========================

def plot_figure_3(df, output_dir, run_name):

    if not safe_has(df, "tau_alert"):
        logging.warning("Missing tau_alert")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["batch"],
        df["tau_alert"],
        marker="o",
    )

    plt.title(f"{run_name} — Threshold")
    plt.xlabel("Batch")
    plt.ylabel("tau_alert")

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "Figure_3_Threshold.png",
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f"Saved {save_path}")


# =========================
# Figure 4
# =========================

def plot_figure_4(df, output_dir, run_name):

    if not safe_has(df, "w"):
        logging.warning("Missing w")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["batch"],
        df["w"],
        marker="o",
    )

    plt.title(f"{run_name} — RF Weight")
    plt.xlabel("Batch")
    plt.ylabel("w")

    plt.ylim(0.65, 0.75)

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "Figure_4_RF_Weight.png",
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f"Saved {save_path}")


# =========================
# Sort runs correctly
# =========================

def run_sort_key(name):

    m = re.search(r"\d+", name)

    return int(m.group()) if m else 0


# =========================
# Main plot generator
# =========================

def generate_all_plots():

    logging.info("Generating plots")

    sns.set_theme(style="ticks")

    runs_dir = config.LOCAL_RESULTS_DIR

    if not os.path.exists(runs_dir):

        logging.error("runs folder not found")

        return

    run_folders = sorted(
        os.listdir(runs_dir),
        key=run_sort_key,
    )

    for run_folder in run_folders:

        run_path = os.path.join(
            runs_dir,
            run_folder,
        )

        if not os.path.isdir(run_path):
            continue

        csv_path = os.path.join(
            run_path,
            "batch_history.csv",
        )

        if not os.path.exists(csv_path):

            logging.warning(
                f"No history in {run_folder}"
            )

            continue

        logging.info(f"Plotting {run_folder}")

        df = pd.read_csv(csv_path)

        plot_figure_2(df, run_path, run_folder)
        plot_figure_3(df, run_path, run_folder)
        plot_figure_4(df, run_path, run_folder)

    logging.info("All plots done")


if __name__ == "__main__":
    generate_all_plots()