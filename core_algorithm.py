"""
Core Algorithm Module: Cultural Algorithm with Queuing Theory
Extracted from CulturalCodeColab.ipynb (Trial 8)
"""

import time
import random
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.special import factorial
from deap import base, creator, tools

# ============================================================================
# PRODUCTION LINE DATA
# ============================================================================

cookie_recipes = {
    0: {"output_qty": 1, "time": 8, "machine_type": "mixer", "queue_type": "FIFO"},
    1: {"output_qty": 400, "time": 4, "machine_type": "dough extruder", "queue_type": "FIFO"},
    2: {"output_qty": 600, "time": 10, "machine_type": "shelf", "queue_type": "LIFO"},
    3: {"output_qty": 300, "time": 12, "machine_type": "oven", "queue_type": "FIFO"},
    4: {"output_qty": 600, "time": 20, "machine_type": "cooling rack", "queue_type": "FIFO"},
    5: {"output_qty": 100, "time": 3, "machine_type": "packager", "queue_type": "FIFO"},
}

scaling_factors = [360, 1, 1, 1, 1, 1]

station_names = ["Mixer", "Dough Extruder", "Shelf", "Oven", "Cooling Rack", "Packager"]

# ============================================================================
# QUEUING THEORY FUNCTIONS
# ============================================================================

def calculate_erlang_c(servers, traffic_intensity):
    """Calculate Erlang C formula for M/M/c queues"""
    if traffic_intensity >= servers:
        return 1.0
    rho = traffic_intensity / servers
    sum_term = 0
    for n in range(servers):
        sum_term += (traffic_intensity ** n) / factorial(n)
    numerator = (traffic_intensity ** servers) / factorial(servers)
    denominator = numerator + (1 - rho) * sum_term
    return numerator / denominator


def mmc_queue_metrics(arrival_rate, service_rate, servers):
    """Calculate M/M/c queue metrics"""
    if servers == 0:
        return 1e+6, 1e+6, 1e+6
    traffic_intensity = arrival_rate / service_rate
    utilization = traffic_intensity / servers
    if utilization >= 1:
        return 1e+6, 1e+6, utilization
    P_queue = calculate_erlang_c(servers, traffic_intensity)
    Lq = P_queue * utilization / (1 - utilization)
    Wq = Lq / arrival_rate
    return Lq, Wq, utilization


def calculate_station_queue_metrics(solution, recipes, scalings, required_rate):
    """Calculate queuing metrics for all stations in the production line.

    Each station is evaluated against the full required_rate so that ALL
    bottlenecks are visible (not masked by upstream constraints).
    System throughput = the minimum output_rate across all stations.
    """
    metrics = []

    for idx, (count, recipe) in enumerate(zip(solution, recipes.values())):
        mu_per_machine = (recipe["output_qty"] / recipe["time"]) * scalings[idx]
        total_service_rate = mu_per_machine * count

        Lq, Wq, utilization = mmc_queue_metrics(
            required_rate, total_service_rate / count, count
        )

        output_rate = min(required_rate, total_service_rate * 0.95)

        metrics.append({
            'station': idx,
            'machines': count,
            'service_rate': total_service_rate,
            'arrival_rate': required_rate,
            'Lq': Lq,
            'Wq': Wq,
            'utilization': utilization,
            'output_rate': output_rate,
        })

    return metrics


# ============================================================================
# CULTURAL ALGORITHM COMPONENTS
# ============================================================================

class BeliefSpace:
    """Cultural Algorithm Belief Space"""

    def __init__(self, n_stations, min_machines=None):
        self.n_stations = n_stations
        self.min_machines = min_machines or [1] * n_stations
        self.normative = {
            'lower': list(self.min_machines),
            'upper': [max(10, mm) for mm in self.min_machines],
            'score': [0.0] * n_stations,
        }
        self.situational = []
        self.fitness_mean = []
        self.fitness_std = []
        self.fitness_best = []

        self.station_mean = np.zeros(n_stations)
        self.station_std = np.zeros(n_stations)

        self.bottleneck_count = np.zeros(n_stations)
        self.bottleneck_util_sum = np.zeros(n_stations)

        self.Wq_sum = np.zeros(n_stations)
        self.util_sum = np.zeros(n_stations)

        self.samples = 0
        self.historical = []

    def update_normative(self, accepted_solutions):
        if not accepted_solutions:
            return
        for i in range(self.n_stations):
            values = [sol[i] for sol in accepted_solutions]
            current_lower = self.normative['lower'][i]
            avg_lower = np.percentile(values, 25)
            if avg_lower < current_lower:
                self.normative['lower'][i] = max(self.min_machines[i], int(avg_lower))
                self.normative['score'][i] += 1
            current_upper = self.normative['upper'][i]
            avg_upper = np.percentile(values, 75)
            if avg_upper > current_upper:
                self.normative['upper'][i] = min(20, int(avg_upper))
                self.normative['score'][i] += 1

    def update_situational(self, solutions):
        self.situational.extend(solutions)
        self.situational = sorted(
            self.situational, key=lambda x: x.fitness.values[0]
        )[:10]

    def update_historical(self, accepted, required_rate, recipes, scalings,
                          eval_fn=None):
        fitnesses = []
        allocations = []
        _eval = eval_fn or evaluate_with_queuing
        for genome in accepted:
            ind = creator.Individual(genome)
            fit = _eval(ind, required_rate, recipes, scalings)
            ind.fitness.values = fit
            fitnesses.append(fit[0])
            allocations.append(genome)

        self.fitness_mean.append(np.mean(fitnesses))
        self.fitness_std.append(np.std(fitnesses))
        self.fitness_best.append(min(fitnesses))

        allocations_arr = np.array(allocations)
        self.station_mean = (
            self.station_mean * self.samples + allocations_arr.mean(axis=0)
        ) / (self.samples + 1)
        self.samples += 1

    def influence(self, individual):
        if random.random() < 0.5:
            for i in range(self.n_stations):
                lower = self.normative['lower'][i]
                upper = self.normative['upper'][i]
                if individual[i] < lower:
                    individual[i] += 1
                elif individual[i] > upper:
                    individual[i] -= 1

        if self.situational and random.random() < 0.3:
            model = random.choice(self.situational)
            for i in range(self.n_stations):
                if random.random() < 0.2:
                    if individual[i] < model[i]:
                        individual[i] += 1
                    elif individual[i] > model[i]:
                        individual[i] -= 1
        return individual


def acceptance_function(population, top_percentile=0.3):
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
    n_selected = max(2, int(len(population) * top_percentile))
    accepted = sorted_pop[:n_selected]
    return [list(ind) for ind in accepted]


# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def min_max_normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def evaluate_with_queuing(individual, required_rate, recipes, scalings):
    queue_metrics = calculate_station_queue_metrics(
        individual, recipes, scalings, required_rate
    )

    total_Lq = sum(m['Lq'] for m in queue_metrics)
    max_Wq = max(m['Wq'] for m in queue_metrics)
    system_throughput = min(m['output_rate'] for m in queue_metrics)

    utilizations = [m['utilization'] for m in queue_metrics]
    bottleneck_utilization = max(utilizations)
    bottleneck_idx = utilizations.index(bottleneck_utilization)

    machine_cost = sum(individual) * 10

    throughput_ratio = system_throughput / required_rate
    if throughput_ratio >= 1.0:
        throughput_penalty = 0
    elif throughput_ratio >= 0.95:
        throughput_penalty = 50
    elif throughput_ratio >= 0.90:
        throughput_penalty = 100
    else:
        throughput_penalty = 1000

    queue_penalty = total_Lq * 5 if total_Lq < 1e+6 else 1000
    waiting_penalty = max_Wq * 100 if max_Wq < 1e+6 else 1000

    util_std = np.std(utilizations)
    if util_std < 0.1 and bottleneck_utilization < 0.85:
        balance_bonus = -50
    elif util_std < 0.2 and bottleneck_utilization < 0.9:
        balance_bonus = -20
    else:
        balance_bonus = 0

    normalized_machine_cost = min_max_normalize(machine_cost, 60, 1320)
    normalized_throughput_penalty = min_max_normalize(throughput_penalty, 0, 1000)
    normalized_queue_penalty = min_max_normalize(queue_penalty, 0, 1000)
    normalized_waiting_penalty = min_max_normalize(waiting_penalty, 0, 1000)

    fitness = (
        normalized_queue_penalty
        + normalized_waiting_penalty
        + normalized_throughput_penalty
        + normalized_machine_cost
    )

    individual.metrics = {
        'queue_metrics': queue_metrics,
        'throughput': system_throughput,
        'bottleneck': bottleneck_idx,
        'bottleneck_util': bottleneck_utilization,
    }

    return (fitness,)


def evaluate_base_ca(individual, required_rate, recipes, scalings):
    """Base CA fitness: machine cost + throughput penalty only (NO queuing penalties)."""
    queue_metrics = calculate_station_queue_metrics(
        individual, recipes, scalings, required_rate
    )

    system_throughput = min(m['output_rate'] for m in queue_metrics)
    utilizations = [m['utilization'] for m in queue_metrics]
    bottleneck_utilization = max(utilizations)
    bottleneck_idx = utilizations.index(bottleneck_utilization)

    machine_cost = sum(individual) * 10

    throughput_ratio = system_throughput / required_rate
    if throughput_ratio >= 1.0:
        throughput_penalty = 0
    elif throughput_ratio >= 0.95:
        throughput_penalty = 50
    elif throughput_ratio >= 0.90:
        throughput_penalty = 100
    else:
        throughput_penalty = 1000

    normalized_machine_cost = min_max_normalize(machine_cost, 60, 1320)
    normalized_throughput_penalty = min_max_normalize(throughput_penalty, 0, 1000)

    fitness = normalized_machine_cost + normalized_throughput_penalty

    individual.metrics = {
        'queue_metrics': queue_metrics,
        'throughput': system_throughput,
        'bottleneck': bottleneck_idx,
        'bottleneck_util': bottleneck_utilization,
    }

    return (fitness,)


# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def create_cultural_population(pop_size, n_stations, belief_space, min_machines=None):
    if min_machines is None:
        min_machines = [1] * n_stations
    population = []
    for _ in range(pop_size):
        ind = [random.randint(min_machines[i], max(10, min_machines[i])) for i in range(n_stations)]
        population.append(creator.Individual(ind))
    return population


def cultural_crossover(ind1, ind2, belief_space):
    tools.cxTwoPoint(ind1, ind2)
    # Enforce minimum machines constraint after crossover
    for i in range(len(ind1)):
        ind1[i] = max(belief_space.min_machines[i], ind1[i])
        ind2[i] = max(belief_space.min_machines[i], ind2[i])
    if random.random() < 0.5:
        belief_space.influence(ind1)
    if random.random() < 0.5:
        belief_space.influence(ind2)
    return ind1, ind2


def cultural_mutation(individual, belief_space, mutation_rate=0.5):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            current = individual[i]
            lower = belief_space.normative['lower'][i]
            upper = belief_space.normative['upper'][i]
            if current < lower:
                individual[i] = random.randint(lower, min(upper, lower + 2))
            elif current > upper:
                individual[i] = random.randint(max(lower, upper - 2), upper)
            else:
                individual[i] = max(belief_space.min_machines[i], current + random.randint(-2, 2))
    return (individual,)


# ============================================================================
# MAIN CULTURAL ALGORITHM (Generator version for GUI progress)
# ============================================================================

def _setup_deap():
    """Ensure DEAP creator classes exist."""
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def cultural_algorithm(recipes, scalings, required_rate,
                       pop_size=50, max_gen=1000, seed=1,
                       use_queuing=True, min_machines=None):
    """
    Run Cultural Algorithm. Returns a generator that yields progress dicts
    each generation, then yields the final result dict.

    use_queuing=True  -> CA+QT (with Lq/Wq penalties)
    use_queuing=False -> Base CA (machine cost + throughput only)
    """
    random.seed(seed)
    np.random.seed(seed)

    n_stations = len(recipes)
    if min_machines is None:
        min_machines = [1] * n_stations
    _setup_deap()

    eval_fn = evaluate_with_queuing if use_queuing else evaluate_base_ca

    belief_space = BeliefSpace(n_stations, min_machines=min_machines)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 10)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_int, n=n_stations)
    toolbox.register("population", create_cultural_population,
                     pop_size=pop_size, n_stations=n_stations,
                     belief_space=belief_space, min_machines=min_machines)
    toolbox.register("evaluate", eval_fn,
                     required_rate=required_rate,
                     recipes=recipes, scalings=scalings)
    toolbox.register("mate", cultural_crossover, belief_space=belief_space)
    toolbox.register("mutate", cultural_mutation, belief_space=belief_space)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population()

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    accepted = acceptance_function(pop)
    belief_space.update_normative(accepted)
    belief_space.update_situational(pop[:5])
    belief_space.update_historical(accepted, required_rate, recipes, scalings,
                                   eval_fn=eval_fn)

    hof = tools.HallOfFame(10)
    hof.update(pop)

    best_so_far = float("inf")
    stagnation_counter = 0
    patience = 50
    start_time = time.time()

    for gen in range(1, max_gen + 1):
        accepted = acceptance_function(pop)
        belief_space.update_normative(accepted)
        belief_space.update_situational(pop[:3])
        belief_space.update_historical(accepted, required_rate, recipes, scalings,
                                       eval_fn=eval_fn)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.5:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        best_fitness = min(ind.fitness.values[0] for ind in pop)

        if best_fitness < best_so_far:
            best_so_far = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        early_stop = stagnation_counter >= patience

        # Population stats for visualization
        all_fitnesses = [ind.fitness.values[0] for ind in pop]
        pop_arr = np.array([list(ind) for ind in pop])

        # Yield progress every generation
        yield {
            'type': 'progress',
            'generation': gen,
            'max_gen': max_gen,
            'best_fitness': best_fitness,
            'best_solution': list(hof[0]),
            'early_stop': early_stop,
            # Belief space state
            'belief_lower': list(belief_space.normative['lower']),
            'belief_upper': list(belief_space.normative['upper']),
            # Population diversity
            'pop_fitness_mean': float(np.mean(all_fitnesses)),
            'pop_fitness_std': float(np.std(all_fitnesses)),
            'pop_fitness_min': float(np.min(all_fitnesses)),
            'pop_fitness_max': float(np.max(all_fitnesses)),
            # Per-station population spread
            'pop_station_mean': pop_arr.mean(axis=0).tolist(),
            'pop_station_std': pop_arr.std(axis=0).tolist(),
            # Crossover/mutation counts
            'n_evaluated': len(invalid_ind),
            'n_population': len(pop),
            'stagnation': stagnation_counter,
        }

        if early_stop:
            break

    elapsed = time.time() - start_time

    # Build final results
    results = []
    for idx, ind in enumerate(hof[:10]):
        metrics = ind.metrics
        queue_metrics = metrics['queue_metrics']
        total_machines = sum(ind)
        throughput = metrics['throughput']
        throughput_ratio = throughput / required_rate

        total_time = sum(
            m['Wq'] + (1 / (m['service_rate'] / max(1, m['machines'])))
            for m in queue_metrics
        )

        results.append({
            'rank': idx + 1,
            'solution': list(ind),
            'machines': total_machines,
            'throughput': throughput,
            'throughput_ratio': throughput_ratio,
            'total_Lq': sum(m['Lq'] for m in queue_metrics),
            'max_Wq': max(m['Wq'] for m in queue_metrics),
            'total_time': total_time,
            'fitness': ind.fitness.values[0],
            'bottleneck': f"Station {metrics['bottleneck'] + 1}",
            'bottleneck_util': metrics['bottleneck_util'],
        })

    yield {
        'type': 'result',
        'results': results,
        'belief_space': belief_space,
        'hof': hof,
        'elapsed': elapsed,
        'required_rate': required_rate,
        'use_queuing': use_queuing,
    }


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

SIMULATION_HOURS = 12
TOTAL_MINUTES = SIMULATION_HOURS * 60


class ProductionSimulator:
    """Discrete-event simulator for production line"""

    def __init__(self, solution, recipes=None, scalings=None, names=None):
        self.solution = solution
        self.recipes = recipes or cookie_recipes
        self.scalings = scalings or scaling_factors
        self.names = names or station_names
        self.stations = []
        self.setup_stations()

    def setup_stations(self):
        n_stations = len(self.solution)
        for i in range(n_stations):
            recipe = self.recipes[i]
            scaling = self.scalings[i] if i < len(self.scalings) else 1
            name = self.names[i] if i < len(self.names) else f"Stage {i+1}"
            capacity_per_machine = (recipe["output_qty"] / recipe["time"]) * scaling
            self.stations.append({
                'name': name,
                'machines': self.solution[i],
                'capacity_per_machine': capacity_per_machine,
                'type': recipe.get('machine_type', f'machine_{i}'),
                'processing_time': recipe['time'],
                'batch_size': recipe["output_qty"] * scaling,
            })

    def simulate_shift(self, machine_availability=0.95, time_variation=0.10):
        station_states = []
        for station in self.stations:
            available_machines = sum(
                1 for _ in range(station['machines'])
                if random.random() <= machine_availability
            )
            if available_machines == 0 and station['machines'] > 0:
                available_machines = 1

            time_factor = 1.0 + random.uniform(-time_variation, time_variation)
            effective_capacity = (
                station['capacity_per_machine'] * available_machines / time_factor
            )

            station_states.append({
                'available_machines': available_machines,
                'effective_capacity': effective_capacity,
                'total_processed': 0,
            })

        bottleneck_capacity = min(s['effective_capacity'] for s in station_states)
        bottleneck_idx = int(np.argmin([s['effective_capacity'] for s in station_states]))
        total_cookies = bottleneck_capacity * TOTAL_MINUTES

        utilizations = []
        for s in station_states:
            util = (
                bottleneck_capacity / s['effective_capacity']
                if s['effective_capacity'] > 0 else 0
            )
            utilizations.append(util)

        return total_cookies, bottleneck_idx, utilizations


def run_monte_carlo(solution, n_simulations=1000, required_rate=None,
                    recipes=None, scalings=None, names=None):
    """Run Monte Carlo simulation and return summary data."""
    recipes = recipes or cookie_recipes
    scalings = scalings or scaling_factors
    names = names or station_names

    if required_rate is None:
        required_rate = 26577 / (12 * 60)

    target = required_rate * TOTAL_MINUTES
    sim = ProductionSimulator(solution, recipes=recipes, scalings=scalings, names=names)

    productions = []
    bottleneck_counts = defaultdict(int)

    for _ in range(n_simulations):
        total, bn_idx, _ = sim.simulate_shift()
        bn_name = names[bn_idx] if bn_idx < len(names) else f"Stage {bn_idx+1}"
        productions.append(total)
        bottleneck_counts[bn_name] += 1

    productions = np.array(productions)
    success_rate = np.mean(productions >= target) * 100

    # Compute system capacity and time to reach target
    sim_base = ProductionSimulator(solution, recipes=recipes, scalings=scalings, names=names)
    system_capacity = min(s['capacity_per_machine'] * s['machines']
                         for s in sim_base.stations)
    hours_to_target = (target / system_capacity) / 60 if system_capacity > 0 else 12
    queue_wait = sum(
        m['Wq'] for m in calculate_station_queue_metrics(
            solution, recipes, scalings, required_rate)
    )

    return {
        'productions': productions,
        'target': target,
        'mean': np.mean(productions),
        'std': np.std(productions),
        'min': np.min(productions),
        'max': np.max(productions),
        'success_rate': success_rate,
        'bottleneck_counts': dict(bottleneck_counts),
        'n_simulations': n_simulations,
        'system_capacity': system_capacity,
        'safety_margin': (system_capacity - required_rate) / required_rate * 100
                         if required_rate > 0 else 0,
        'hours_to_target': hours_to_target,
        'queue_wait': queue_wait,
    }


def run_stress_test(solution, base_required_rate, recipes=None, scalings=None, names=None):
    """Sensitivity analysis: test solution under -20% to +20% demand variation."""
    recipes = recipes or cookie_recipes
    scalings = scalings or scaling_factors
    names = names or station_names

    scenarios = [
        ("-20%", 0.80),
        ("-10%", 0.90),
        ("Base", 1.00),
        ("+10%", 1.10),
        ("+20%", 1.20),
    ]

    results = []
    for label, factor in scenarios:
        test_rate = base_required_rate * factor
        metrics = calculate_station_queue_metrics(solution, recipes, scalings, test_rate)
        utilizations = [m['utilization'] for m in metrics]
        bottleneck_util = max(utilizations)
        bottleneck_idx = utilizations.index(bottleneck_util)

        results.append({
            'scenario': label,
            'demand_factor': factor,
            'required_rate': test_rate,
            'bottleneck_station': names[bottleneck_idx] if bottleneck_idx < len(names) else f"Stage {bottleneck_idx+1}",
            'bottleneck_util': bottleneck_util,
            'total_Lq': sum(m['Lq'] for m in metrics),
            'max_Wq': max(m['Wq'] for m in metrics),
            'status': 'OK' if bottleneck_util < 0.85 else (
                'Warning' if bottleneck_util < 0.95 else 'Critical'),
        })

    return results
