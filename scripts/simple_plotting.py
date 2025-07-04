import argparse
import os
from datetime import datetime
import pandas as pd
from robustness_experiment_box.analysis.report_creator_singleNetwork import ReportCreator

def timestamped_filename(base, ext, time_format="%Y%m%d-%H%M%S"):
    timestamp = datetime.now().strftime(time_format)
    return f"{base}_{timestamp}{ext}"

def save_figure(fig, out_dir, base_filename):
    filename = timestamped_filename(base_filename, ".png")
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Create and save timestamped figures from a DataFrame using ReportCreator.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the figures.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    rc = ReportCreator(df)

    save_figure(rc.create_hist_figure(), args.output_dir, "hist")
    save_figure(rc.create_box_figure(), args.output_dir, "box")
    save_figure(rc.create_kde_figure(), args.output_dir, "kde")
    save_figure(rc.create_ecdf_figure(), args.output_dir, "ecdf")

    # Anneplot: returns an Axes, so use its figure
    anne_ax = rc.create_anneplot()
    anne_filename = timestamped_filename("anneplot", ".png")
    anne_path = os.path.join(args.output_dir, anne_filename)
    anne_ax.figure.savefig(anne_path, bbox_inches="tight")
    print(f"Saved: {anne_path}")

if __name__ == "__main__":
    main()
