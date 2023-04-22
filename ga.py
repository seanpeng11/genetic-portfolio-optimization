import numpy as np
from typing import Callable, List, Tuple


def selection(k: int, scores: List[float], optimum: bool = True) -> int:
    """
    Selects the chromosome yielding the best score from a random set of k chromosomes.
    :param k: The length of the subset of scores to select from
    :param scores: The list of population scores
    :param optimum: Whether to select by maximum or minimum score (True -> maximum, False -> minimum)
    :return: The index corresponding to the largest selected scores
    """
    # Select k random scores (by index)
    selected = np.random.randint(0, len(scores), k)
    if optimum:
        return selected[np.argmax([scores[x] for x in selected])]
    else:
        return selected[np.argmin([scores[x] for x in selected])]


def crossover(c1: List[int], c2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Performs a crossover on two given chromosomes.
    :param c1: The first chromosome
    :param c2: The second chromosome
    :return: A tuple containing the two crossed-over chromosomes
    """
    # Null check
    if not c2:
        return c1, c2

    cross_pt = np.random.randint(1, len(c1)-1)

    crossed1 = c1[:cross_pt] + c2[cross_pt:]
    crossed2 = c2[:cross_pt] + c1[cross_pt:]

    return crossed1, crossed2


def mutate(c: List[int]):
    """
    Mutates a random bit on the given chromosome.
    :param c: The chromosome to mutate
    """
    if not c:
        return

    # for x in range(2):
    #     idx = np.random.randint(x*len(c)/2, (x+1)*len(c)/2)
    #     c[idx] = 1 - c[idx]

    idx = np.random.randint(len(c))
    c[idx] = 1 - c[idx]


class GeneticAlgorithm:
    """
    A class representing and allowing for execution of a genetic algorithm for a given fitness function and hyperparameters.

    ...

    Attributes
    ----------
    population: List[List[int]]
        The initial population of solutions/chromosomes for this algorithm
    fitness_func: Callable[..., float]
        The fitness function for this algorithm, should return some kind of numeric scalar
    decode_func: Callable
        A decoding function for converting bitstrings in the population (search space) to the appropriate corresponding values (problem space)
    scores: List[float]
        A list containing the evaluations of the population according to the fitness function for the current generation
    pop_size: int
        The size of the population
    k: int
        Hyperparameter for selection; determines the size of the subset a parent will be selected from
    p_c: float
        Hyperparameter for crossover; determines the approximate probability two given parents will crossover to form children
    p_m: float
        Hyperparameter for mutation; determines the approximate probability a given parent will mutate to form a child
    optimum: bool
        Parameter to configure the algorithm to maximize or minimize the fitness function (true -> maximize, false -> minimize)
    generation: int
        The current generation number
    best: Tuple[..., float]
        A tuple containing the current generation's best score and corresponding input (represented in the problem space)
    """

    def __init__(self, population: List[List[int]], fitness_func: Callable[..., float], decode_func: Callable,
                 k: int = 3, p_c: float = 0.8, p_m: float = None, optimum: bool = True):
        self.population = population
        self.fitness = fitness_func
        self.decode = decode_func
        self.scores = None

        self.pop_size = len(population)
        self.k = k
        self.p_c = p_c
        self.p_m = 1.0 / len(self.population[0]) if not p_m else p_m
        self.optimum = optimum

        self.generation = 0
        self.best = ('null', 'null')

    def iterate(self):
        # Evaluate this generation
        decoded_pop = [self.decode(chrom) for chrom in self.population]
        self.scores = [self.fitness(decoded_chrom) for decoded_chrom in decoded_pop]
        if self.optimum:
            self.best = (self.decode(self.population[np.argmax(self.scores)]), np.max(self.scores))
        else:
            self.best = (self.decode(self.population[np.argmin(self.scores)]), np.min(self.scores))

        # Begin generating the next generation via selection, crossover, and mutation
        parents = [self.population[selection(self.k, self.scores, self.optimum)] for _ in range(self.pop_size)]
        new_gen = []
        for x in range(0, len(parents), 2):
            c1 = parents[x]
            c2 = None if x+1 >= len(parents) else parents[x+1]

            if np.random.random() < self.p_c:
                c1, c2 = crossover(c1, c2)

            if np.random.random() < self.p_m:
                mutate(c1)
            if np.random.random() < self.p_m:
                mutate(c2)

            new_gen.append(c1)
            # This will only not trigger when pop_size is odd and we are on the last set of parents
            if c2:
                new_gen.append(c2)

        self.population = new_gen
        self.generation = self.generation + 1



