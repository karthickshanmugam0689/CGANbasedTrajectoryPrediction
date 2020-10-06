import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation

with open('ResultTrajectories.pkl', 'rb') as f:
    a = pickle.load(f)
    for b in a:
        print(b.keys())
        print("Enter the sequence you want to visualize")
        seq_start = int(input("Enter the sequence start: "))
        seq_end = int(input("Enter the sequence end:"))
        positions = b.get((seq_start, seq_end))
        if positions.size(0) > 0:
            break

num_ped = positions.size(1)

colors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def init():
    scatterplot.set_offsets([[], []])
    return [scatterplot]


def update(i, scatterplot, positions, colors):
    scatterplot.set_offsets(positions[i])
    scatterplot.set_array(colors)
    return [scatterplot]


#plt.show()
fig = plt.figure()
scatterplot = plt.scatter([], [], s=10)

plt.xlim(-2, 15)
plt.ylim(0, 15)

plt.xlabel('Trajectory x coordinate value')
plt.ylabel('Trajectory y coordinate value')
plt.title("Simulated Trajectories")
anim = animation.FuncAnimation(fig, update, init_func=init, fargs=(scatterplot, positions, colors), interval=500,
                               frames=12, blit=True, repeat=True)
# anim.save('SimulatedTraj3.gif', writer='imagemagick', fps=2)
plt.show()
