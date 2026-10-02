import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/training_logs/Overnight 3.17.csv', parse_dates=['timestamp'])

eval_df = df[(df['win_rate'] != 0) | (df['avg_margin'] != 0)].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(eval_df['timestamp'], eval_df['win_rate'], marker='o', markersize=3, linewidth=1, color='seagreen')
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50%')
ax1.set_ylabel('Win Rate')
ax1.set_title('Win Rate over Time (eval only)')
ax1.legend()

ax2.plot(eval_df['timestamp'], eval_df['avg_margin'], marker='o', markersize=3, linewidth=1, color='crimson')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_ylabel('Avg Score Margin')
ax2.set_title('Avg Score Margin over Time (eval only)')
ax2.set_xlabel('Time')

plt.tight_layout()
plt.savefig('data/analysis/eval.png', dpi=150)
plt.show()
