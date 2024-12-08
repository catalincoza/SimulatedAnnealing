import random
import matplotlib.pyplot as plt
import numpy as np


class SimulatedAnnealing:
    def __init__(self, temp, cooling_rate, iterations, local_searches, multiplier,
                 lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.temperature_0 = temp
        self.temperature = temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.local_searches = local_searches
        self.multiplier = multiplier
        self.lower_bound_x = lower_bound_x
        self.upper_bound_x = upper_bound_x
        self.lower_bound_y = lower_bound_y
        self.upper_bound_y = upper_bound_y
        self.history = []
        self.acceptance_probability_history = []

    def starting_point(self):
        x = random.uniform(self.lower_bound_x, self.upper_bound_x)
        y = random.uniform(self.lower_bound_y, self.upper_bound_y)
        return x, y

    def neighbour(self, x, y, multiplier=1.0):
        return (x + random.uniform(-1, 1) * multiplier,
                y + random.uniform(-1, 1) * multiplier)

    def himmelblau_function(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def acceptance_probability(self, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / self.temperature)

    def optimize(self):
        x, y = self.starting_point()
        current_cost = self.himmelblau_function(x, y)
        self.history.append((x, y))

        for _ in range(self.iterations):
            new_x, new_y = self.neighbour(x, y, self.multiplier[0])
            new_cost = self.himmelblau_function(new_x, new_y)
            acc_prob = self.acceptance_probability(current_cost, new_cost)

            if acc_prob > random.random():
                x, y = new_x, new_y
                current_cost = new_cost
                self.history.append((x, y))

            for _ in range(self.local_searches):
                new_x, new_y = self.neighbour(x, y, self.multiplier[1])
                new_cost = self.himmelblau_function(new_x, new_y)
                acc_prob = self.acceptance_probability(current_cost, new_cost)

                if acc_prob > random.random():
                    x, y = new_x, new_y
                    current_cost = new_cost
                    self.history.append((x, y))

            self.temperature *= self.cooling_rate

        return x, y, current_cost

    def plot(self, parameter_name, parameter_value):
        x = np.linspace(-6, 6, 400)
        y = np.linspace(-6, 6, 400)
        x, y = np.meshgrid(x, y)
        z = self.himmelblau_function(x, y)

        fig = plt.figure(figsize=(14, 7))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)
        hx, hy = zip(*self.history)
        hz = [self.himmelblau_function(px, py) for px, py in self.history]
        ax1.plot(hx, hy, hz, color='r', marker='o', markersize=3, label='Path Optimizare')
        ax1.set_title(f"Path Optimizare (Parametru: {parameter_name} = {parameter_value})")
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(hz, color='g', marker='.', linestyle='-', linewidth=1)
        ax2.set_title("Istoria Solutiilor")
        ax2.set_xlabel("Iteratie")
        ax2.set_ylabel("Valoarea Functiei")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    default_params = {
        'temp': 500,
        'cooling_rate': 0.9,
        'iterations': 200,
        'local_searches': 20,
        'multiplier': [0.8, 0.2],
        'lower_bound_x': -6,
        'upper_bound_x': 6,
        'lower_bound_y': -6,
        'upper_bound_y': 6
    }

    analysis = {
        'temp': [100, 500, 1000],
        'cooling_rate': [0.8, 0.9, 0.95],
        'iterations': [100, 200, 300],
        'local_searches': [10, 20, 30],
        'multiplier': [[0.5, 0.1], [0.8, 0.2], [1.0, 0.5]]
    }

    for param, values in analysis.items():
        for value in values:
            params = default_params.copy()
            params[param] = value
            sa = SimulatedAnnealing(**params)
            result = sa.optimize()
            print(f"Parametrul {param} = {value} => Rezultat: {result}")
            sa.plot(param, value)