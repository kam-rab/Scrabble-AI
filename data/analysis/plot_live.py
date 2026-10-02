import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training_logs/training_log.csv'
out_path = 'data/analysis/live_plot.png'

while True:
    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        eval_df = df[(df['win_rate'] != 0) | (df['avg_margin'] != 0)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_sl, ax_wr, ax_sm = axes.flatten()

        ax_sl.plot(df['timestamp'], df['score_loss'], linewidth=0.5, color='darkorange')
        ax_sl.set_title('Score Loss')
        ax_sl.set_ylabel('Loss')
        recent = df['score_loss'].iloc[-200:]
        ax_sl.set_ylim(recent.min() * 0.9, recent.max() * 1.1)

        if not eval_df.empty:
            ax_wr.plot(eval_df['timestamp'], eval_df['win_rate'], marker='o', markersize=3, linewidth=1, color='seagreen')
            ax_wr.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
            ax_wr.set_title('Win Rate (eval)')
            ax_wr.set_ylabel('Win Rate')

            ax_sm.plot(eval_df['timestamp'], eval_df['avg_margin'], marker='o', markersize=3, linewidth=1, color='crimson')
            ax_sm.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax_sm.set_title('Avg Score Margin (eval)')
            ax_sm.set_ylabel('Margin')

        fig.suptitle(f'Training Monitor — {csv_path}', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f'Updated {out_path}')
    except Exception as e:
        print(f'Error: {e}')

    time.sleep(30)
