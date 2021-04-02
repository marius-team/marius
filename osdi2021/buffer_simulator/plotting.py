from collections import defaultdict

import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.ticker import FixedLocator
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import buffer_simulator.buffer as buffer
import buffer_simulator.orderings as order

plt.style.use("ggplot")


# --------------These are copied from SO--------------
def colorline(
        x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# --------------These are copied from SO--------------

def plot_ordering(coords, n, sym=True, title="Ordering"):
    coords.reverse()
    figure = plt.figure(figsize=(12, 10), dpi=100)
    cmap = plt.cm.jet
    ax = figure.add_subplot(111)
    path = mpath.Path(coords)
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]

    if sym:
        xs = []
        ys = []
        for i in range(len(x)):
            if x[i] < y[i]:
                xs.append(x[i])
                ys.append(y[i])
        x = xs
        y = ys

    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, ax=ax, cmap=cmap, linewidth=4, alpha=.5)

    ax.set_ylim(0, n - 1)
    ax.set_xlim(0, n - 1)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Destination Node")
    plt.ylabel("Source Node")
    plt.show()


def plot_varying_num_partitions(n0, c0, num):
    data_dict = defaultdict(list)

    figure = plt.figure(figsize=(10, 8), dpi=100)
    ax = figure.add_subplot(111)

    n = n0
    c = c0

    ns = []
    for j in range(num):
        ns.append(n)
        data_dict["Hilbert"].append(buffer.hilbert_ordering(n, c, output=False)[1])
        data_dict["Greedy"].append(buffer.greedy_ordering(n, c, output=False)[1])
        data_dict["Hilbert Symmetric"].append(buffer.hilbert_ordering_symmetric(n, c, output=False)[1])
        data_dict["Lower Bound"].append(order.get_lower_bound(n, c))
        n = n * 2
        c = c * 2

    cmap = plt.get_cmap("tab10")
    idx = 0
    ls = "-"
    for k, v in data_dict.items():
        if k == "Lower Bound":
            ls = "--"
        ax.plot(ns, v, color=cmap(idx), linestyle=ls, label="Ordering: %s" % (k))
        idx += 1

    plt.legend()
    plt.title("Total Swaps when varying N. N0 = %s, C0 = %s" % (n0, c0))
    plt.xlabel("N")
    plt.ylabel("Number of swaps")
    plt.show()


def plot_varying_num_partitions_io(n0, c0, num, total_size, output_path=None):
    data_dict = defaultdict(list)

    figure = plt.figure(figsize=(10, 8), dpi=100)
    ax = figure.add_subplot(111)

    n = n0
    c = c0

    ns = []

    for j in range(num):
        ns.append(n)

        block_size = total_size / n

        data_dict["Hilbert"].append(buffer.hilbert_ordering(n, c, output=False)[1] * block_size)
        data_dict["Elimination"].append(buffer.greedy_ordering(n, c, output=False)[1] * block_size)
        data_dict["Hilbert Symmetric"].append(buffer.hilbert_ordering_symmetric(n, c, output=False)[1] * block_size)
        data_dict["Lower Bound"].append(order.get_lower_bound(n, c) * block_size)
        n = n * 2
        c = c * 2

    cmap = plt.get_cmap("tab10")
    idx = 0
    ls = "-"
    for k, v in data_dict.items():
        if k == "Lower Bound":
            ls = "--"
        ax.plot(ns, v, color=cmap(idx), linestyle=ls, label="Ordering: %s" % (k))
        idx += 1

    plt.legend()
    plt.title("Total IO when varying N")
    plt.xlabel("N")
    plt.ylabel("Total IO size (GB)")

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def plot_colorgrid(coords, n, misses = [], sym=True, title="Ordering"):
    # coords.reverse()

    plt.rc('font', family='serif')
    plt.rcParams['font.size'] = 18
    # plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR


    figure = plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap('Greys', 10)
    ax = figure.add_subplot(111)
    path = mpath.Path(coords)
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]

    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(5)
        ax.spines[side].set_color("black")

    grid_vals = np.ones((n+1, n+1))

    for i in range(len(x)):

        grid_vals[int(x[i]), int(y[i])] = 0

        if ((int(x[i]), int(y[i])) in misses):
            ax.text(int(x[i]), int(y[i]), str(i), horizontalalignment='center', verticalalignment='center', color="white", size =25)
        else:
            ax.text(int(x[i]), int(y[i]), str(i), horizontalalignment='center', verticalalignment='center', color="black", size=25)

    # draw gridlines

    for m in misses:
        grid_vals[m[1], m[0]] = .5

    ax.imshow(grid_vals, cmap=cmap)


    # ax.set_xticks(np.arange(0, n, 1))
    # ax.set_yticks(np.arange(0, n, 1))

    major_locator = FixedLocator(np.arange(-.5, n - 1, 1))
    ax.set_ylim(-.5, n-.5)
    ax.set_xlim(-.5, n-.5)
    plt.gca().invert_yaxis()
    ax.xaxis.tick_top()
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_major_locator(major_locator)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.xaxis.set_ticklabels(np.arange(0, n, 1))
    ax.yaxis.set_ticklabels(np.arange(0, n, 1))
    plt.xlabel("Destination Node Partition")
    plt.ylabel("Source Node Partition")
    # plt.show()
    plt.savefig("hilbert.pdf")

def animate_ordering(coords, n):

    figure = plt.figure(figsize=(8, 5), dpi=90)
    cmap = plt.cm.get_cmap('jet', (len(coords)))
    ax = figure.add_subplot(111)
    path = mpath.Path(coords)
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]

    grid_vals = -np.ones((n, n))
    grid_plot = ax.matshow(grid_vals, cmap=cmap, vmin=0, vmax=len(coords))
    ax.set_xticks(np.arange(-.5, n - 1, 1))
    ax.set_yticks(np.arange(-.5, n - 1, 1))
    plt.grid(b=True, which='major', axis='both', linestyle='-', color='k', linewidth=2, animated=True)
    plt.axis('off')
    def update(i):
        grid_vals[int(x[i]), int(y[i])] = i
        grid_plot.set_data(grid_vals)
        return [grid_plot]

    ani = animation.FuncAnimation(figure, update, frames=len(x), interval=10, blit=True, repeat=True, repeat_delay=5)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('optimal.mp4', writer=writer)



