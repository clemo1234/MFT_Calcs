import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---- Replace with your real data ----
# x array (same size as each inner array)
x = np.linspace(0, 10, 200)

# outer array of inner arrays
data = np.array([np.sin(x + i*0.3) for i in range(30)])
# -----------------------------------

idx0 = 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(x, data[idx0])
ax.set_title(f"Index = {idx0}")

# slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Index", 0, len(data)-1,
                valinit=idx0, valstep=1)

def update(val):
    i = int(slider.val)
    line.set_ydata(data[i])
    ax.set_title(f"Index = {i}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
