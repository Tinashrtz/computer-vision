import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

df = pd.read_csv("BTC-USD.csv")
df_open = df["Open"]

fig, ax = plt.subplots()

# Create a line for the animation
animation_line, = ax.plot(range(len(df_open)), [np.min(df_open)] * len(df_open), 'b-', label='Animation')

# Plot the df_open data
open_line, = plt.plot(df_open, 'r-', label='df_open')

ax.set_xlim(0, len(df_open))  # Set the x-axis limits
ax.set_ylim(np.min(df_open), np.max(df_open))  # Set the y-axis limits

# Add legend
ax.legend()

def update(frame):
    y_position = np.min(df_open) + (frame / len(df_open)) * (np.max(df_open) - np.min(df_open))
    animation_line.set_ydata([y_position] * len(df_open))
    return animation_line,

# Create the animation with repeat=False
ani = FuncAnimation(fig, update, frames=range(len(df_open)), blit=True, interval=3, repeat=False)

plt.show()
