import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Tuple

def selection(k: int, scores: NDArray[float]) -> int:
    """
    Selects the chromosome yielding the largest score from a random set of k chromosomes.
    :param k: The length of the subset of scores to select from
    :param scores: The list of population scores
    :return: The index corresponding to the largest selected scores
    """
    # Select k random scores (by index)
    selected = np.random.randint(0, len(scores), k)
    return selected[np.argmin([scores[x] for x in selected])]


def crossover(c1: NDArray[int], c2: NDArray[int]) -> Tuple[NDArray[int], NDArray[int]]:
    """
    Performs a crossover on two given chromosomes.
    :param c1: The first chromosome
    :param c2: The second chromosome
    :return: A tuple containing the two crossed-over chromosomes
    """
    # Null check
    if c2 is None:
        return c1, c2

    cross_pt = np.random.randint(1, len(c1)-1)

    crossed1 = np.concatenate((c1[:cross_pt], c2[cross_pt:]))
    crossed2 = np.concatenate((c2[:cross_pt], c1[cross_pt:]))

    return crossed1, crossed2


def mutate(c: NDArray[int]) -> NDArray[int]:
    """
    Mutates a random bit on the given chromosome.
    :param c: The chromosome to mutate
    """
    if c is None:
        return

    # for x in range(2):
    #     idx = np.random.randint(x*len(c)/2, (x+1)*len(c)/2)
    #     c[idx] = 1 - c[idx]

    idx = np.random.randint(len(c))
    c[idx] = 1 - c[idx]


def arithmetic_crossover(c1: NDArray[float], c2: NDArray[float]) -> Tuple[NDArray[float], NDArray[float]]:
    alpha = np.random.rand()
    return alpha*c1 + (1 - alpha)*c2, alpha*c2 + (1-alpha)*c1


def f_coord(x: float, y: float) -> float:
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)


def f(p: Tuple[float, float]) -> float:
    return f_coord(p[0], p[1])


def decode(bitstring: List[int]) -> Tuple[float, float]:
    bounds = [[0.0, 5.0], [0.0, 5.0]]
    coords = []
    max_16 = 2**16 - 1
    for i in range(2):
        var_string = ''.join([str(x) for x in bitstring[i * 16: (i + 1) * 16]])
        # This will take the var string, an arbitrary 16-bit binary string, parse it as a 16-bit integer and scale it down into our bounds.
        var = bounds[i][0] + ((float(int(var_string, 2)) / max_16) * (bounds[i][1] - bounds[i][0]))
        coords.append(var)

    return coords[0], coords[1]


def encode(p: NDArray) -> NDArray[int]:
    x_bits = format(int((p[0] / 5.0) * (2**16 - 1)), '016b')
    y_bits = format(int((p[1] / 5.0) * (2**16 - 1)), '016b')
    bits = x_bits + y_bits
    bits = [int(bit) for bit in bits]

    return np.asarray(bits)


# t = (1.25135, 1.25135)
# t_b = np.random.randint(2, size=32)
# # print(f't: {t}')
# print('t_b:' + str(t_b))
#
# t_enc = encode(t)
# t_b_dec = decode(t_b)
#
# # print('t_enc: ' + str(t_enc))
# print('t_b_dec: ' + str(t_b_dec))
#
# t_redec = decode(t_enc)
# t_b_reenc = encode(t_b_dec)
#
# # print(f't_redec: ({t_redec})')
# print('t_b_reenc: ' + str(t_b_reenc))
# print('t_b_redec: ' + str(decode(t_b_reenc)))
# print(all(t_b_reenc == t_b))
# print(str(t_b))
# print(str(t_b_reenc))

class HGAPSO:
    def __init__(self, pop_size: int = 20,
                 k: int = 3, p_c: float = 0.6, p_m: float = 0.1, c1: float = 0.1, c2: float = 0.1, w: float = 0.8):
        # hyperparameters
        self.pop_size = pop_size
        self.k = k
        self.p_c = p_c
        self.p_m = p_m
        self.c1 = c1  # exploration
        self.c2 = c2  # exploitation
        self.w = w

        # initialize population
        self.particles = np.random.randint(2, size=(self.pop_size, 32))  # in bitstrings
        self.vels = np.random.randn(self.pop_size, 2) * 0.1  # in numbers
        self.scores = np.asarray([f(decode(particle)) for particle in self.particles])

        self.pbest = self.particles.copy()
        self.pbest_scores = np.asarray([f(decode(pbesti)) for pbesti in self.pbest])

        self.gbest = self.pbest[np.argmin(self.pbest_scores)]
        self.gbest_score = f(decode(self.gbest))

        self.gen = 0

    def iterate(self):
        # Select the best-performing half of the particles
        elite_ind = np.argpartition(self.scores, int(self.pop_size/2))[:int(self.pop_size/2)]
        elites = self.particles[elite_ind]
        elite_vels = self.vels[elite_ind]
        elite_pbest = self.pbest[elite_ind]
        elite_pbest_dec = np.asarray([decode(pbest) for pbest in elite_pbest])

        # Decode the elite particles in order to enhance them
        elites = np.asarray([decode(elite) for elite in elites])

        # Update vels via the formula:self.
        # Vi(t+1) = w*Vi(t) + c1r1*(pbesti - Xi(t)) + c2r2(gbest - Xi(t))
        # Where r1 and r2 are random numbers on [0,1), and Xi is the position of the ith elite particle
        r = np.random.rand(2)
        elite_vels = self.w * elite_vels + self.c1 * r[0] * (elite_pbest_dec - elites) + self.c2 * r[1] * (decode(self.gbest) - elites)
        # Enhance elites with new velocities
        elites += elite_vels
        np.clip(elites, 0.0, 5.0)

        # Calculate scores of the enhanced elites and update elite pbests
        elite_scores = np.asarray([f(elite) for elite in elites])
        for j in range(len(elites)):
            if elite_scores[j] < f(elite_pbest_dec[j]):
                elite_pbest[j] = encode(elites[j])

        # elite_pbest = np.asarray([encode(elites[i]) if elite_scores[i] > self.pbest_scores[elite_ind[i]] else elite_pbest[i] for i in range(len(elites))])

        # Now do genetic algorithm segment with the elites as the parents
        # The first pop_size/2 members of the next generation will be the enhanced elites of the previous generation
        # The second pop_size/2 members of the next generation will be the offspring of the enhanced elites of the previous generation
        # Offspring of elites should have their velocities set to 0 and their pbests set to their current positions
        # Select parents via tournament selection
        parents = elites.copy()[np.asarray([selection(self.k, elite_scores) for _ in range(len(elites))])]
        offspring = np.empty((int(self.pop_size/2), 32), dtype=np.int8)
        offspring_vels = np.zeros((int(self.pop_size/2), 2))
        for i in range(0, len(parents), 2):
            cross1 = encode(parents[i])
            cross2 = None if i+1 >= len(parents) else encode(parents[i+1])

            if np.random.rand() < self.p_c:
                cross1, cross2 = crossover(cross1, cross2)

            if np.random.rand() < self.p_m:
                mutate(cross1)

            if np.random.rand() < self.p_m:
                mutate(cross2)

            offspring[i] = cross1
            if cross2 is not None:
                offspring[i+1] = cross2

        # Calculate scores of offspring
        offspring_scores = np.asarray([f(decode(child)) for child in offspring])
        offspring_pbest = offspring.copy()

        # Re-encode the elites
        elites = np.asarray([encode(elite) for elite in elites])

        # Now we have our elites and our offspring, so join them and overwrite the previous generation with the new generation
        self.particles = np.concatenate((elites, offspring))
        self.vels = np.concatenate((elite_vels, offspring_vels))

        # Update pbests and gbest
        self.pbest = np.concatenate((elite_pbest, offspring_pbest))
        self.pbest_scores = np.asarray([f(decode(pbesti)) for pbesti in self.pbest])

        potential_gbest = self.pbest[np.argmin(self.pbest_scores)]
        self.gbest = potential_gbest if f(decode(potential_gbest)) < self.gbest_score else self.gbest
        self.gbest_score = f(decode(self.gbest))

        self.gen += 1

    def iterate_v2(self):
        # Enhance all the particles
        particles_dec = np.asarray([decode(particle) for particle in self.particles])
        pbest_dec = np.asarray([decode(pbest) for pbest in self.pbest])
        gbest_dec = decode(self.gbest)
        # Update vels via the formula:
        # Vi(t+1) = w*Vi(t) + c1r1*(pbesti - Xi(t)) + c2r2(gbest - Xi(t))
        # Where r1 and r2 are random numbers on [0,1), and Xi is the position of the ith elite particle
        # Select the best-performing half of the particles
        r = np.random.rand(2)
        self.vels = self.w * self.vels + self.c1 * r[0] * (pbest_dec - particles_dec) + self.c2 * r[1] * (gbest_dec - particles_dec)
        particles_dec += self.vels
        np.clip(particles_dec, 0.0, 5.0, particles_dec)

        # Update scores, pbests, and gbest
        self.particles = np.asarray([encode(particle) for particle in particles_dec])
        self.scores = np.asarray([f(particle) for particle in particles_dec])
        for i in range(self.pop_size):
            if self.scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = self.scores[i]
                self.pbest[i] = self.particles[i]

        gen_best_ind = np.argmin(self.pbest_scores)
        if self.pbest_scores[gen_best_ind] < self.gbest_score:
            self.gbest_score = self.pbest_scores[gen_best_ind]
            self.gbest = self.pbest[gen_best_ind]

        # Now perform tournament selection to select parents
        selected_inds = np.asarray([selection(self.k, self.scores) for _ in range(self.pop_size)])
        parents = self.particles.copy()[selected_inds]
        parent_vels = self.vels.copy()[selected_inds]
        offspring = np.empty((self.pop_size, 32), dtype=np.int8)
        offspring_vels = np.zeros((self.pop_size, 2))
        for i in range(0, len(parents), 2):
            cross1 = parents[i]
            cross2 = None if i+1 >= len(parents) else parents[i+1]

            vel1 = parent_vels[i]
            vel2 = None if i+1 >= len(parents) else parent_vels[i+1]

            if np.random.rand() < self.p_c:
                cross1, cross2 = crossover(cross1, cross2)
                vel1, vel2 = arithmetic_crossover(vel1, vel2)

            if np.random.rand() < self.p_m:
                mutate(cross1)

            if np.random.rand() < self.p_m:
                mutate(cross2)

            offspring[i] = cross1
            offspring_vels[i] = vel1
            if cross2 is not None:
                offspring[i+1] = cross2
                offspring_vels[i+1] = vel2

        self.particles = offspring
        self.vels = offspring_vels

        self.gen += 1
