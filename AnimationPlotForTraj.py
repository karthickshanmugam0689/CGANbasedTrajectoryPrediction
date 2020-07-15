import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Each column represents particular pedestrian trajectories. Eg: Column1: Ped1, Column2: Ped2...
positions = np.array([[[ 0.2907,  0.2606], [-0.8884, -6.7368], [-0.5884,  2.8360], [-0.6548,  3.3607], [ 1.7687, -4.2044], [ 2.4916, -4.3365], [ 1.1101, -1.2508], [ 1.6908, -0.9436], [ 2.2341, -1.0901]],
                      [[ 0.3367,  0.1755], [-0.8987, -6.7787], [-0.5936,  2.8392], [-0.6474,  3.3609], [ 1.7548, -4.7662], [ 2.5043, -4.8626], [ 0.9810, -1.6482], [ 1.4765, -1.2774], [ 1.9942, -1.4231]],
                    [[ 0.3673,  0.0832], [-0.9191, -6.8266], [-0.6023,  2.8414], [-0.6434,  3.3683], [ 1.6985, -5.2862], [ 2.4542, -5.3132], [ 0.7973, -2.0043], [ 1.1939, -1.5474], [ 1.6592, -1.6669]],
                        [[ 0.3831, -0.0166], [-0.9536, -6.8803], [-0.6158,  2.8408], [-0.6455,  3.3765], [ 1.5966, -5.7718], [ 2.3188, -5.6945], [ 0.5781, -2.3555], [ 0.8569, -1.7820], [ 1.2372, -1.8298]],
                        [[ 0.3923, -0.1107], [-1.0031, -6.9355], [-0.6298,  2.8373], [-0.6509,  3.3813], [ 1.4701, -6.2220], [ 2.1166, -6.0036], [ 0.3509, -2.7125], [ 0.4864, -1.9938], [ 0.7549, -1.9195]],
                        [[ 0.4129, -0.1842], [-1.0650, -6.9864], [-0.6418,  2.8321], [-0.6566,  3.3821], [ 1.3355, -6.6572], [ 1.8705, -6.2728], [ 0.1348, -3.0907], [ 0.1060, -2.2068], [ 0.2556, -1.9654]],
                        [[ 0.4712, -0.2160], [-1.1374, -7.0295], [-0.6520,  2.8280], [-0.6616,  3.3820], [ 1.2142, -7.0942], [ 1.6215, -6.4978], [-0.0546, -3.4971], [-0.2569, -2.4410], [-0.2257, -1.9996]],
                      [[ 0.5973, -0.1898], [-1.2194, -7.0626], [-0.6612,  2.8243], [-0.6662,  3.3809], [ 1.1255, -7.5291], [ 1.4043, -6.6527], [-0.2094, -3.9248], [-0.5809, -2.6967], [-0.6824, -2.0490]]])

# Different colors for different pedestrians. In above example, 10 colors for 10 ped
colors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def init():
    scatterplot.set_offsets([[], []])
    return [scatterplot]


def update(i, scatterplot, positions, colors):
    scatterplot.set_offsets(positions[i])
    scatterplot.set_array(colors)
    return [scatterplot]

plt.show()
fig = plt.figure()
scatterplot = plt.scatter([], [], s=30)

# Change xlim and ylim with respect to Min and Max values of different trajectories
plt.xlim(-2,3)
plt.ylim(-11,5)

plt.xlabel('Trajectory x coordinate value')
plt.ylabel('Trajectory y coordinate value')
plt.title("Predicted Trajectories")
anim = animation.FuncAnimation(fig, update, init_func=init, fargs=(scatterplot, positions, colors), interval=350, frames=8, blit=True, repeat=True)
#anim.save('Plot1.gif', writer='imagemagick', fps=2)
plt.show()