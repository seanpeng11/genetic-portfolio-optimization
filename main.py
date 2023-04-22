import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from ga import GeneticAlgorithm
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

file_path = "./3D_surface_and_contour.jpg"


# Multivariable multi-optimum example
# Base fitness function
def fit_x_y(x, y):
    return 0.01 * (x-1) * (x+2) * (x+5) * (x-3) * (y-3) * (y+2) * (y+5)


# Just here for convenience with decode function
def fit_func(tup):
    x = tup[0]
    y = tup[1]
    return 0.01 * (x-1) * (x+2) * (x+5) * (x-3) * (y-3) * (y+2) * (y+5)


def encode_x_y(x, y):
    # This is a continuous function over R, but since we're using floats, to prevent weirdness with bit manipulation of float64 and negative numbers,
    # we'll add 5 (to ensure positive values), round to the thousandths, multiply by 1000, and use that integer to convert to a 16-bit representation
    # for the bitstring. The decoder function will perform a similar, but different conversion, since this function is only necessary for us to get
    # some reasonable starting values for the population. So long as the decoder is consistent, the algorithm should function.
    x_mod = int((x+5) * 1000)
    y_mod = int((y+5) * 1000)
    x_arr = [int(bit) for bit in format(x_mod, '016b')]
    y_arr = [int(bit) for bit in format(x_mod, '016b')]
    return x_arr + y_arr


def decode_x_y(bitstring):
    decoded_coords = []
    bounds = [[-5.0, 3.0], [-5.0, 3.0]]
    max_16 = 2**16 - 1
    for i in range(2):
        var_string = ''.join([str(x) for x in bitstring[i * 16: (i + 1) * 16]])
        # This will take the var string, an arbitrary 16-bit binary string, parse it as a 16-bit integer and scale it down into our bounds.
        var = bounds[i][0] + ((float(int(var_string, 2)) / max_16) * (bounds[i][1] - bounds[i][0]))
        decoded_coords.append(var)

    return tuple(decoded_coords)


xlist = np.linspace(-5.0, 3.0, 1000)
ylist = np.linspace(-5.0, 3.0, 1000)
X, Y = np.meshgrid(xlist, ylist)
# Z = 0.01 * (X - 1) * (X + 2) * (X + 5) * (X - 3) * (Y - 3) * (Y + 2) * (Y + 5)
Z = fit_x_y(X, Y)

fig, ax = plt.subplots()

ax = plt.figure().add_subplot(projection='3d')


# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=100, cstride=100,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-50, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=4, cmap='coolwarm')

ax.set(xlim=(-5, 3), ylim=(-5, 4), zlim=(-50, 50),
       xlabel='X', ylabel='Y', zlabel='Z')

plt.savefig("3dOrig.pdf")

# plt.imshow(Z, extent=[-5, 3, -5, 3], origin='lower', cmap=mpl.colormaps.get('coolwarm'), alpha=1)
# # cp = ax.contour(X, Y, Z, levels=10, colors='black')
# plt.colorbar()
# ax.set_title('Contour graph of f(x,y) for x, y on [-5, 3]')
# plt.ion()
# plt.savefig('contour_graph.png')
# plt.show()

# Initial population size 20
rng = np.random.default_rng()
# Generate 20 random 32-bit bitstrings
initial_pop = (rng.integers(0, high=2, size=(20, 32))).tolist()
# Decode these for visualization purposes
initial_pop_coords = [decode_x_y(bitstring) for bitstring in initial_pop]

# plt.ion()
# fig, ax = plt.subplots()
# plt.show()
# cp = ax.contourf(X, Y, Z, levels=20, cmap=mpl.colormaps.get('turbo'))
# fig.colorbar(cp)
scatter = ax.scatter3D([coord[0] for coord in initial_pop_coords], [coord[1] for coord in initial_pop_coords], [fit_func(coord) for coord in initial_pop_coords], color='#e834eb', marker='.', s=50)
ax.set_title('Generation 0')
fig.canvas.draw()
fig.canvas.flush_events()
# os.system('pause')
plt.savefig('gen_0.pdf')

ga = GeneticAlgorithm(initial_pop, fit_func, decode_x_y, p_m=0.1)
print(f'Generation {ga.generation}:\t{ga.best[0]}\t with {ga.best[1]}')

for x in range(50):
    ga.iterate()
    pop_50 = [decode_x_y(bits) for bits in ga.population]
    pop_50_x = [coord[0] for coord in pop_50]
    pop_50_y = [coord[1] for coord in pop_50]
    pop_50_z = [fit_func(coord) for coord in pop_50]
    scatter.remove()
    scatter = ax.scatter3D(pop_50_x, pop_50_y, pop_50_z,color='#e834eb', marker='.', s=50)
    ax.set_title(f'Generation {ga.generation}')
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f'Generation {ga.generation}:\t{ga.best[0]}\t with {ga.best[1]}')

plt.savefig('gen_50.pdf')
