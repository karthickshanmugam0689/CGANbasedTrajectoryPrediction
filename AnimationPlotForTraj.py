import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from csgan.constants import *

with open('scripts\obj.pkl', 'rb') as f:
    a = pickle.load(f)
    for b in a:
        c = SEQUENCE_TO_VISUALIZE
        positions = b.get(SEQUENCE_TO_VISUALIZE)
        if positions.size(0) > 0:
            break

num_ped = positions.size(1)

colors = np.array([0.1])
color_value = 0.2
for i in range(0, num_ped):
    colors = np.append(colors, np.array(color_value))
    color_value += 0.1


def init():
    scatterplot.set_offsets([[], []])
    return [scatterplot]


def update(i, scatterplot, positions, colors):
    scatterplot.set_offsets(positions[i])
    scatterplot.set_array(colors)
    return [scatterplot]

plt.show()
fig = plt.figure()
scatterplot = plt.scatter([], [], s=10)


plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.xlabel('Trajectory x coordinate value')
plt.ylabel('Trajectory y coordinate value')
plt.title("Simulated Trajectories")
anim = animation.FuncAnimation(fig, update, init_func=init, fargs=(scatterplot, positions, colors), interval=350, frames=8, blit=True, repeat=True)
#anim.save('Plot.gif', writer='imagemagick', fps=2)
plt.show()