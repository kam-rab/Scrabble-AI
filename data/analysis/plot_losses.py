import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/training_logs/Overnight 3.17.csv', parse_dates=['timestamp'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df['timestamp'], df['win_loss'], linewidth=0.5, color='steelblue')
ax1.set_ylabel('Win Loss')
ax1.set_title('Win Loss over Time')

ax2.plot(df['timestamp'], df['score_loss'], linewidth=0.5, color='darkorange')
ax2.set_ylabel('Score Loss')
ax2.set_title('Score Loss over Time')
ax2.set_xlabel('Time')

plt.tight_layout()
plt.savefig('data/analysis/losses.png', dpi=150)
plt.show()
