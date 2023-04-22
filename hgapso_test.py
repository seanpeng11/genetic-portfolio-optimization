from hgapso import HGAPSO, decode, encode, f_coord, f
from ga import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import time


# visualization setup
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
graph_x, graph_y = np.array(np.meshgrid(np.linspace(0, 5, 1000), np.linspace(0, 5, 1000)))
graph_z = f_coord(graph_x, graph_y)

x_min = graph_x.ravel()[graph_z.argmin()]
y_min = graph_y.ravel()[graph_z.argmin()]
z_min = graph_z.min()

# Old 2D stuff
# img = ax.imshow(graph_z, extent=[0, 5, 0, 5], origin='lower', cmap='bwr')
# fig.colorbar(img, ax=ax)
# contours = ax.contour(graph_x, graph_y, graph_z, 10, colors='black', alpha=0.3)
# ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

# Plot the 3D surface
ax.plot_surface(graph_x, graph_y, graph_z, edgecolor='royalblue', lw=0.5, rstride=100, cstride=100,
                alpha=0.3)

min_pt = ax.scatter(x_min, y_min, z_min, marker='x', color='white', alpha=0.5)
flat_min_pt = ax.scatter(x_min, y_min, -9, marker='x', color='white', alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(graph_x, graph_y, graph_z, zdir='z', offset=-10, cmap='coolwarm')
ax.contourf(graph_x, graph_y, graph_z, zdir='x', offset=0, cmap='coolwarm')
ax.contourf(graph_x, graph_y, graph_z, zdir='y', offset=5, cmap='coolwarm')

ax.set(xlim=(0, 5), ylim=(0, 5), zlim=(-10, 20),
       xlabel='X', ylabel='Y', zlabel='Z')

# standard GA setup
rng = np.random.default_rng()
ga_pop = (rng.integers(0, high=2, size=(20, 32))).tolist()
ga = GeneticAlgorithm(ga_pop, f, decode, 3, 0.8, 0.2, False)

# hybrid alg setup
hybrid = HGAPSO(20, 3, 0.6, 0.2, 0.1, 0.1, 0.8)

plt.ion()

ax.view_init(azim=-20, elev=20)
plt.savefig('hgapso_figs/initial_graph_2.pdf')
plt.show()

for i in range(200):
    ax.set_title(f'Generation {i}')

    particles_dec = np.asarray([decode(particle) for particle in hybrid.particles])
    scores = hybrid.scores
    vels = hybrid.vels
    gbest = decode(hybrid.gbest)
    gbest_score = hybrid.gbest_score
    flat_z = np.zeros(hybrid.pop_size) - 9

    particles_plot = ax.scatter(particles_dec[:, 0], particles_dec[:, 1], scores,  marker='o', color='black', s=10)
    flat_parts_plot = ax.scatter(particles_dec[:, 0], particles_dec[:, 1], flat_z,  marker='o', color='black', s=10, alpha=0.5)
    # elite_plot = ax.scatter(particles_dec[:2, 0], particles_dec[:2, 1], marker='o', color='orange', s=10)
    vels_plot = ax.quiver(particles_dec[:, 0], particles_dec[:, 1], scores, vels[:, 0], vels[:, 1], np.zeros(hybrid.pop_size), color='green',
                          length=2)
    flat_vels_plot = ax.quiver(particles_dec[:, 0], particles_dec[:, 1], flat_z, vels[:, 0], vels[:, 1], np.zeros(hybrid.pop_size), color='green', length=2, alpha=0.5)
    gbest_plot = ax.scatter(gbest[0], gbest[1], gbest_score, marker='*', s=100, color='purple', alpha=0.5)
    flat_gbest_plot = ax.scatter(gbest[0], gbest[1], -9, marker='*', s=100, color='purple', alpha=0.3)

    dist_to_min = np.sqrt(sum(np.power(gbest - np.asarray([x_min, y_min]), 2)))

    fig.canvas.draw()
    fig.canvas.flush_events()

    if i == 0:
        plt.savefig('hgapso_figs/hgapso_gen0.pdf')

    # time.sleep(1)
    # plt.savefig(f'hgapso_figs/gen_{i}.png')
    # print(f'Generation {i}\tpt: ({gbest[0]}, {gbest[1]})\tdist: {dist_to_min}')
    if dist_to_min < 1e-2:
        plt.savefig(f'hgapso_figs/hgapso_gen{hybrid.gen}.pdf')

    particles_plot.remove()
    flat_parts_plot.remove()
    # elite_plot.remove()
    vels_plot.remove()
    flat_vels_plot.remove()
    gbest_plot.remove()
    flat_gbest_plot.remove()

    if dist_to_min < 1e-2:
        break

    # hybrid.iterate()
    hybrid.iterate_v2()


print()
print(f'HGAPSO found best solution at ({gbest[0]}, {gbest[1]})')
print(f'Approx. global minimum at ({x_min}, {y_min})')
print(f'Distance to min: {dist_to_min}')
print(f'Generations until convergence: {hybrid.gen}')
print()

for i in range(200):
    ax.set_title(f'Generation {i}')

    sols = np.asarray([decode(chrom) for chrom in ga.population])
    scores = np.asarray(ga.scores)
    best = np.asarray(ga.best[0])
    best_score = ga.best[1]

    sols_plot = ax.scatter(sols[:, 0], sols[:, 1], scores, marker='o', color='black', s=10)
    flat_sols_plot = ax.scatter(sols[:, 0], sols[:, 1], np.zeros(ga.pop_size) - 9, marker='o', color='black', s=10, alpha=0.5)
    best_plot = ax.scatter(best[0], best[1], best_score, marker='*', s=100, color='purple', alpha=0.5)
    flat_best_plot = ax.scatter(best[0], best[1], -9, marker='*', s=100, color='purple', alpha=0.3)

    if i == 0:
        plt.savefig('hgapso_figs/ga_gen0.pdf')

    dist_to_min = np.sqrt(sum(np.power(best - np.asarray([x_min, y_min]), 2)))

    fig.canvas.draw()
    fig.canvas.flush_events()

    if dist_to_min <= 1e-2:
        plt.savefig(f'hgapso_figs/ga_gen{i}.pdf')
        break
    if i == 199:
        plt.savefig(f'hgapso_figs/ga_gen200.pdf')
    # print(f'Generation {i}\tpt: ({best[0]}, {best[1]})\tdist: {dist_to_min}')

    sols_plot.remove()
    flat_sols_plot.remove()
    best_plot.remove()
    flat_best_plot.remove()

    ga.iterate()

print()
print(f'GA found best solution at ({best[0]}, {best[1]})')
print(f'Approx. global minimum at ({x_min}, {y_min})')
print(f'Distance to min: {dist_to_min}')
print(f'Generations until convergence: {ga.generation}')
print()
