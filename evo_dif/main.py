import numpy as np

# Rastrigin function
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Differential Evolution
def differential_evolution(func, bounds, pop_size=20, max_gen=1000, F=0.8, CR=0.9):
    dim = len(bounds)
    population = np.random.rand(pop_size, dim)
    for i in range(dim):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    
    for gen in range(max_gen):
        for i in range(pop_size):
            # Mutation
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            if func(trial) < func(population[i]):
                population[i] = trial
    
    # Return the best solution
    best_idx = np.argmin([func(ind) for ind in population])
    return population[best_idx], func(population[best_idx])

# Define bounds and run the algorithm
bounds = [(-5.12, 5.12)] * 10  # 3-dimensional Rastrigin function
best_solution, best_value = differential_evolution(rastrigin, bounds)

print("Best Solution:", best_solution)
print("Best Value:", best_value)