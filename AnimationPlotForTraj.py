import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation

with open('Sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

with open('SimulatedTraj.pkl', 'rb') as f:
    trajectories = pickle.load(f)
    print("Enter the sequence you want to visualize", sequences)
    seq_start = int(input("Enter the sequence start: "))
    seq_end = int(input("Enter the sequence end:"))
    positions = trajectories[:, seq_start:seq_end, :]

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

# Adjust the x and y limits according to the min and max range from the trajectories for better visualization
plt.xlim(-2, 15)
plt.ylim(0, 15)

plt.xlabel('Trajectory x coordinate value')
plt.ylabel('Trajectory y coordinate value')
plt.title("Simulated Trajectories")
anim = animation.FuncAnimation(fig, update, init_func=init, fargs=(scatterplot, positions, colors), interval=500,
                               frames=PRED_LEN, blit=True, repeat=True)
plt.show()
