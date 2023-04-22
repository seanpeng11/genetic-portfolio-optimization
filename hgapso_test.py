from hgapso import HGAPSO, decode, encode, f_coord
import numpy as np
import matplotlib.pyplot as plt
import time


# visualization setup
fig, ax = plt.subplots()
fig.set_tight_layout(True)
graph_x, graph_y = np.array(np.meshgrid(np.linspace(0, 5, 1000), np.linspace(0, 5, 1000)))
graph_z = f_coord(graph_x, graph_y)
x_min = graph_x.ravel()[graph_z.argmin()]
y_min = graph_y.ravel()[graph_z.argmin()]
img = ax.imshow(graph_z, extent=[0, 5, 0, 5], origin='lower', cmap='bwr')
fig.colorbar(img, ax=ax)
contours = ax.contour(graph_x, graph_y, graph_z, 10, colors='black', alpha=0.3)
min_pt = ax.scatter(x_min, y_min, marker='x', color='white', alpha=0.3)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

alg = HGAPSO(20, 3, 0.6, 0.1, 0.1, 0.1, 0.8)

plt.ion()

plt.show()

for i in range(200):
    ax.set_title(f'Generation {i}')

    particles_dec = np.asarray([decode(particle) for particle in alg.particles])
    vels = alg.vels
    gbest = decode(alg.gbest)
    particles_plot = ax.scatter(particles_dec[:, 0], particles_dec[:, 1], marker='o', color='black', s=10)
    # elite_plot = ax.scatter(particles_dec[:2, 0], particles_dec[:2, 1], marker='o', color='orange', s=10)
    vels_plot = ax.quiver(particles_dec[:, 0], particles_dec[:, 1], vels[:, 0], vels[:, 1], color='green', width=0.005, angles='xy', scale_units='xy', scale=0.5)
    gbest_plot = ax.scatter(gbest[0], gbest[1], marker='*', s=100, color='purple', alpha=0.5)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # time.sleep(1)
    # plt.savefig(f'hgapso_figs/gen_{i}.png')
    if np.sqrt(sum(np.power(gbest - np.asarray([x_min, y_min]), 2))) < 1e-2:
        break

    particles_plot.remove()
    # elite_plot.remove()
    vels_plot.remove()
    gbest_plot.remove()

    # alg.iterate()
    alg.iterate_v2()

print(f'HGAPSO found best solution at ({gbest[0]}, {gbest[1]})')
print(f'Approx. global minimum at ({x_min}, {y_min})')
print(f'Generations until convergence: {alg.gen}')
