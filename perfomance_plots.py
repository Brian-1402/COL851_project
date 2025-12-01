'''
runs both part3, part4 performance plots 
Usage: 

python perfomance_plots.py --file ./pi_outputs/part4/patna_performance.csv --output_dir part4_performance_plots/ --prefix pi_patna
'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# --- Custom Logger Import ---
try:
    from log import setup_logger
except ImportError:
    import logging
    def setup_logger(name, log_file, log_dir):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

def plot_metric(df, x_col, y_col, title, ylabel, color, output_path):
    """Helper to generate and save a single metric plot."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(df[x_col], df[y_col], color=color, linewidth=1.5)
    
    # Fill area under curve for better visualization
    plt.fill_between(df[x_col], df[y_col], color=color, alpha=0.1)
    
    plt.xlabel("Cumulative Inference Time (seconds)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="General Performance Plotter for Part 3 & 4")
    parser.add_argument("--file", type=str, required=True, help="Path to performance CSV file")
    parser.add_argument("--output_dir", type=str, default="plots_performance", help="Directory to save plots")
    parser.add_argument("--prefix", type=str, default="metric", help="Prefix for output images (e.g., 'laptop' or 'pi')")
    
    args = parser.parse_args()

    # Setup Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("perf_plotter", "plot_log", args.output_dir)

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return

    logger.info(f"Loading data from {args.file}...")
    df = pd.read_csv(args.file)

    # 1. Calculate Cumulative Time (X-Axis)
    # The time for each row = latency summed up from previous rows
    if 'latency_sec' not in df.columns:
        logger.error("Column 'latency_sec' missing. Cannot calculate cumulative time.")
        return
    
    df['cum_time'] = df['latency_sec'].cumsum()
    max_time = df['cum_time'].max()
    logger.info(f"Total cumulative inference time: {max_time:.2f} seconds")

    # 2. Plot CPU Utilization
    if 'cpu_util_pct' in df.columns:
        out_path = os.path.join(args.output_dir, f"{args.prefix}_cpu_util.png")
        plot_metric(df, 'cum_time', 'cpu_util_pct', 
                   f"CPU Utilization ({args.prefix})", "CPU Utilization (%)", "#d35400", out_path)
        logger.info(f"Saved CPU plot to {out_path}")
    else:
        logger.warning("Column 'cpu_util_pct' missing. Skipping CPU plot.")

    # 3. Plot RAM Utilization
    if 'ram_util_mb' in df.columns:
        out_path = os.path.join(args.output_dir, f"{args.prefix}_ram_util.png")
        plot_metric(df, 'cum_time', 'ram_util_mb', 
                   f"RAM Utilization ({args.prefix})", "RAM Usage (MB)", "#27ae60", out_path)
        logger.info(f"Saved RAM plot to {out_path}")
    else:
        logger.warning("Column 'ram_util_mb' missing. Skipping RAM plot.")

    # 4. Plot CPU Temperature (Conditional)
    # Check if column exists AND has valid data (not all NaNs or zeros if that indicates missing)
    if 'cpu_temp_c' in df.columns:
        # Check if the column is effectively empty (e.g., all NaNs)
        if df['cpu_temp_c'].isnull().all():
            logger.info("Temperature column exists but is empty (Part 3 Laptop data?). Skipping Temp plot.")
        else:
            out_path = os.path.join(args.output_dir, f"{args.prefix}_cpu_temp.png")
            plot_metric(df, 'cum_time', 'cpu_temp_c', 
                       f"CPU Temperature ({args.prefix})", "Temperature (°C)", "#c0392b", out_path)
            logger.info(f"Saved Temperature plot to {out_path}")
    else:
        logger.info("Column 'cpu_temp_c' not found. Skipping Temp plot.")

    logger.info("All plots generated successfully.")

if __name__ == "__main__":
    main()