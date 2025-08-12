import random
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class MutationType(Enum):
    NUMERIC = "numeric"
    BINARY = "binary"
    STRING = "string"


@dataclass
class GAConfig:
    """Configuration class for Genetic Algorithm parameters"""
    population_size: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_type: MutationType = MutationType.NUMERIC
    num_generations: int = 100
    elitism_count: int = 2
    tournament_size: int = 3
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for JSON serialization"""
        return {
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_method': self.selection_method.value,
            'crossover_method': self.crossover_method.value,
            'mutation_type': self.mutation_type.value,
            'num_generations': self.num_generations,
            'elitism_count': self.elitism_count,
            'tournament_size': self.tournament_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create configuration from dictionary"""
        return cls(
            population_size=data['population_size'],
            mutation_rate=data['mutation_rate'],
            crossover_rate=data['crossover_rate'],
            selection_method=SelectionMethod(data['selection_method']),
            crossover_method=CrossoverMethod(data['crossover_method']),
            mutation_type=MutationType(data['mutation_type']),
            num_generations=data['num_generations'],
            elitism_count=data['elitism_count'],
            tournament_size=data['tournament_size']
        )


@dataclass
class GenerationStats:
    """Statistics for a single generation"""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    std_fitness: float
    best_individual: List


class Individual:
    """Represents an individual in the population"""
    
    def __init__(self, chromosome: List, fitness: float = None):
        self.chromosome = chromosome
        self.fitness = fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness if self.fitness is not None else False
    
    def __gt__(self, other):
        return self.fitness > other.fitness if self.fitness is not None else False


class FitnessFunction(ABC):
    """Abstract base class for fitness functions"""
    
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual"""
        pass
    
    @abstractmethod
    def create_individual(self) -> Individual:
        """Create a random individual for the problem"""
        pass


class SelectionOperator:
    """Handles different selection strategies"""
    
    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)
    
    @staticmethod
    def roulette_wheel_selection(population: List[Individual]) -> Individual:
        """Roulette wheel selection"""
        # Handle negative fitness values by shifting
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in population]
        else:
            adjusted_fitness = [ind.fitness for ind in population]
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, individual in enumerate(population):
            current += adjusted_fitness[i]
            if current >= pick:
                return individual
        return population[-1]
    
    @staticmethod
    def rank_selection(population: List[Individual]) -> Individual:
        """Rank-based selection"""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        for i, individual in enumerate(sorted_pop):
            current += ranks[i]
            if current >= pick:
                return individual
        return sorted_pop[-1]


class CrossoverOperator:
    """Handles different crossover operations"""
    
    @staticmethod
    def single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if len(parent1.chromosome) <= 1:
            return Individual(parent1.chromosome[:]), Individual(parent2.chromosome[:])
        
        crossover_point = random.randint(1, len(parent1.chromosome) - 1)
        
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        
        return Individual(child1_chromosome), Individual(child2_chromosome)
    
    @staticmethod
    def two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover"""
        if len(parent1.chromosome) <= 2:
            return CrossoverOperator.single_point_crossover(parent1, parent2)
        
        point1 = random.randint(1, len(parent1.chromosome) - 2)
        point2 = random.randint(point1 + 1, len(parent1.chromosome) - 1)
        
        child1_chromosome = (parent1.chromosome[:point1] + 
                           parent2.chromosome[point1:point2] + 
                           parent1.chromosome[point2:])
        child2_chromosome = (parent2.chromosome[:point1] + 
                           parent1.chromosome[point1:point2] + 
                           parent2.chromosome[point2:])
        
        return Individual(child1_chromosome), Individual(child2_chromosome)
    
    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        child1_chromosome = []
        child2_chromosome = []
        
        for i in range(len(parent1.chromosome)):
            if random.random() < 0.5:
                child1_chromosome.append(parent1.chromosome[i])
                child2_chromosome.append(parent2.chromosome[i])
            else:
                child1_chromosome.append(parent2.chromosome[i])
                child2_chromosome.append(parent1.chromosome[i])
        
        return Individual(child1_chromosome), Individual(child2_chromosome)


class MutationOperator:
    """Handles different mutation operations"""
    
    @staticmethod
    def numeric_mutation(individual: Individual, mutation_rate: float, 
                        bounds: Tuple[float, float] = (-10, 10)) -> None:
        """Gaussian mutation for numeric chromosomes"""
        for i in range(len(individual.chromosome)):
            if random.random() < mutation_rate:
                # Gaussian mutation with standard deviation based on range
                std_dev = (bounds[1] - bounds[0]) * 0.1
                mutation = random.gauss(0, std_dev)
                individual.chromosome[i] += mutation
                # Keep within bounds
                individual.chromosome[i] = max(bounds[0], min(bounds[1], individual.chromosome[i]))
    
    @staticmethod
    def binary_mutation(individual: Individual, mutation_rate: float) -> None:
        """Bit-flip mutation for binary chromosomes"""
        for i in range(len(individual.chromosome)):
            if random.random() < mutation_rate:
                individual.chromosome[i] = 1 - individual.chromosome[i]
    
    @staticmethod
    def string_mutation(individual: Individual, mutation_rate: float, 
                       alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        """Character substitution mutation for string chromosomes"""
        for i in range(len(individual.chromosome)):
            if random.random() < mutation_rate:
                individual.chromosome[i] = random.choice(alphabet)


class GAResults:
    """Stores and manages GA execution results"""
    
    def __init__(self):
        self.generation_stats: List[GenerationStats] = []
        self.best_individual: Individual = None
        self.execution_time: float = 0
        self.config: GAConfig = None
    
    def add_generation_stats(self, stats: GenerationStats):
        """Add statistics for a generation"""
        self.generation_stats.append(stats)
    
    def get_fitness_history(self) -> Tuple[List[float], List[float]]:
        """Get best and average fitness history"""
        best_fitness = [stats.best_fitness for stats in self.generation_stats]
        avg_fitness = [stats.average_fitness for stats in self.generation_stats]
        return best_fitness, avg_fitness
    
    def save_to_json(self, filename: str):
        """Save results to JSON file"""
        data = {
            'config': self.config.to_dict() if self.config else None,
            'execution_time': self.execution_time,
            'best_individual': {
                'chromosome': self.best_individual.chromosome if self.best_individual else None,
                'fitness': self.best_individual.fitness if self.best_individual else None
            },
            'generation_stats': [{
                'generation': stats.generation,
                'best_fitness': stats.best_fitness,
                'average_fitness': stats.average_fitness,
                'worst_fitness': stats.worst_fitness,
                'std_fitness': stats.std_fitness,
                'best_individual': stats.best_individual
            } for stats in self.generation_stats]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_to_pickle(self, filename: str):
        """Save results to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_pickle(cls, filename: str):
        """Load results from pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def plot_fitness_evolution(self, save_path: Optional[str] = None):
        """Plot fitness evolution over generations"""
        best_fitness, avg_fitness = self.get_fitness_history()
        generations = list(range(len(best_fitness)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a detailed text report"""
        if not self.generation_stats:
            return "No results to report."
        
        final_stats = self.generation_stats[-1]
        initial_stats = self.generation_stats[0]
        
        report = f"""
=== Genetic Algorithm Execution Report ===

Configuration:
- Population Size: {self.config.population_size if self.config else 'N/A'}
- Generations: {len(self.generation_stats)}
- Mutation Rate: {self.config.mutation_rate if self.config else 'N/A'}
- Crossover Rate: {self.config.crossover_rate if self.config else 'N/A'}
- Selection Method: {self.config.selection_method.value if self.config else 'N/A'}
- Crossover Method: {self.config.crossover_method.value if self.config else 'N/A'}
- Mutation Type: {self.config.mutation_type.value if self.config else 'N/A'}
- Elitism Count: {self.config.elitism_count if self.config else 'N/A'}

Execution Results:
- Execution Time: {self.execution_time:.2f} seconds
- Initial Best Fitness: {initial_stats.best_fitness:.6f}
- Final Best Fitness: {final_stats.best_fitness:.6f}
- Fitness Improvement: {final_stats.best_fitness - initial_stats.best_fitness:.6f}
- Final Average Fitness: {final_stats.average_fitness:.6f}
- Final Standard Deviation: {final_stats.std_fitness:.6f}

Best Solution:
- Chromosome: {self.best_individual.chromosome if self.best_individual else 'N/A'}
- Fitness: {self.best_individual.fitness if self.best_individual else 'N/A'}

Generation Statistics (Last 5):
"""
        
        for stats in self.generation_stats[-5:]:
            report += f"Gen {stats.generation:3d}: Best={stats.best_fitness:8.4f}, "
            report += f"Avg={stats.average_fitness:8.4f}, Std={stats.std_fitness:6.4f}\n"
        
        return report


class GeneticAlgorithm:
    """Main Genetic Algorithm framework class"""
    
    def __init__(self, fitness_function: FitnessFunction, config: GAConfig = None):
        self.fitness_function = fitness_function
        self.config = config or GAConfig()
        self.population: List[Individual] = []
        self.results = GAResults()
        self.results.config = self.config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self) -> None:
        """Initialize the population with random individuals"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = self.fitness_function.create_individual()
            individual.fitness = self.fitness_function.evaluate(individual)
            self.population.append(individual)
        
        self.logger.info(f"Initialized population of size {self.config.population_size}")
    
    def select_parent(self) -> Individual:
        """Select a parent using the configured selection method"""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return SelectionOperator.tournament_selection(
                self.population, self.config.tournament_size)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return SelectionOperator.roulette_wheel_selection(self.population)
        elif self.config.selection_method == SelectionMethod.RANK:
            return SelectionOperator.rank_selection(self.population)
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover using the configured method"""
        if self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return CrossoverOperator.single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
            return CrossoverOperator.two_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
            return CrossoverOperator.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.config.crossover_method}")
    
    def mutate(self, individual: Individual) -> None:
        """Mutate an individual using the configured mutation type"""
        if self.config.mutation_type == MutationType.NUMERIC:
            MutationOperator.numeric_mutation(individual, self.config.mutation_rate)
        elif self.config.mutation_type == MutationType.BINARY:
            MutationOperator.binary_mutation(individual, self.config.mutation_rate)
        elif self.config.mutation_type == MutationType.STRING:
            MutationOperator.string_mutation(individual, self.config.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation type: {self.config.mutation_type}")
    
    def calculate_generation_stats(self, generation: int) -> GenerationStats:
        """Calculate statistics for the current generation"""
        fitnesses = [ind.fitness for ind in self.population]
        best_individual = max(self.population, key=lambda ind: ind.fitness)
        
        return GenerationStats(
            generation=generation,
            best_fitness=max(fitnesses),
            average_fitness=np.mean(fitnesses),
            worst_fitness=min(fitnesses),
            std_fitness=np.std(fitnesses),
            best_individual=best_individual.chromosome[:]
        )
    
    def create_next_generation(self) -> List[Individual]:
        """Create the next generation using selection, crossover, and mutation"""
        next_generation = []
        
        # Elitism: preserve best individuals
        if self.config.elitism_count > 0:
            elite = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            next_generation.extend(elite[:self.config.elitism_count])
        
        # Generate remaining individuals
        while len(next_generation) < self.config.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = Individual(parent1.chromosome[:])
                child2 = Individual(parent2.chromosome[:])
            
            self.mutate(child1)
            self.mutate(child2)
            
            child1.fitness = self.fitness_function.evaluate(child1)
            child2.fitness = self.fitness_function.evaluate(child2)
            
            next_generation.extend([child1, child2])
        
        # Trim to exact population size
        return next_generation[:self.config.population_size]
    
    def run(self) -> GAResults:
        """Execute the genetic algorithm"""
        import time
        start_time = time.time()
        
        self.logger.info("Starting Genetic Algorithm execution")
        self.initialize_population()
        
        for generation in range(self.config.num_generations):
            # Calculate and store statistics
            stats = self.calculate_generation_stats(generation)
            self.results.add_generation_stats(stats)
            
            # Log progress
            if generation % 10 == 0 or generation == self.config.num_generations - 1:
                self.logger.info(
                    f"Generation {generation}: Best={stats.best_fitness:.4f}, "
                    f"Avg={stats.average_fitness:.4f}")
            
            # Create next generation (except for the last generation)
            if generation < self.config.num_generations - 1:
                self.population = self.create_next_generation()
        
        # Store final results
        self.results.best_individual = max(self.population, key=lambda ind: ind.fitness)
        self.results.execution_time = time.time() - start_time
        
        self.logger.info(f"GA completed in {self.results.execution_time:.2f} seconds")
        self.logger.info(f"Best fitness: {self.results.best_individual.fitness:.6f}")
        
        return self.results
    
    def save_config(self, filename: str):
        """Save GA configuration to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filename: str) -> GAConfig:
        """Load GA configuration from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return GAConfig.from_dict(data)


# Example fitness functions for common optimization problems

class SphereFunction(FitnessFunction):
    """Sphere function: minimize sum of squares"""
    
    def __init__(self, dimensions: int = 10, bounds: Tuple[float, float] = (-10, 10)):
        self.dimensions = dimensions
        self.bounds = bounds
    
    def evaluate(self, individual: Individual) -> float:
        # Return negative sum of squares (since GA maximizes)
        return -sum(x**2 for x in individual.chromosome)
    
    def create_individual(self) -> Individual:
        chromosome = [random.uniform(self.bounds[0], self.bounds[1]) 
                     for _ in range(self.dimensions)]
        return Individual(chromosome)


class RastriginFunction(FitnessFunction):
    """Rastrigin function: multimodal test function"""
    
    def __init__(self, dimensions: int = 10, bounds: Tuple[float, float] = (-5.12, 5.12)):
        self.dimensions = dimensions
        self.bounds = bounds
    
    def evaluate(self, individual: Individual) -> float:
        A = 10
        n = len(individual.chromosome)
        result = A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in individual.chromosome)
        return -result  # Return negative (since GA maximizes)
    
    def create_individual(self) -> Individual:
        chromosome = [random.uniform(self.bounds[0], self.bounds[1]) 
                     for _ in range(self.dimensions)]
        return Individual(chromosome)


class OneMaxFunction(FitnessFunction):
    """OneMax function: maximize number of 1s in binary string"""
    
    def __init__(self, length: int = 50):
        self.length = length
    
    def evaluate(self, individual: Individual) -> float:
        return sum(individual.chromosome)
    
    def create_individual(self) -> Individual:
        chromosome = [random.randint(0, 1) for _ in range(self.length)]
        return Individual(chromosome)


# Example usage
if __name__ == "__main__":
    # Configure the GA
    config = GAConfig(
        population_size=100,
        num_generations=50,
        mutation_rate=0.02,
        crossover_rate=0.8,
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.SINGLE_POINT,
        mutation_type=MutationType.NUMERIC,
        elitism_count=5,
        tournament_size=3
    )
    
    # Create a fitness function (Sphere function)
    fitness_func = SphereFunction(dimensions=5, bounds=(-10, 10))
    
    # Initialize and run the GA
    ga = GeneticAlgorithm(fitness_func, config)
    results = ga.run()
    
    # Print report
    print(results.generate_report())
    
    # Save results
    results.save_to_json("ga_results.json")
    results.save_to_pickle("ga_results.pkl")
    
    # Plot fitness evolution
    results.plot_fitness_evolution("fitness_evolution.png")
    
    # Save configuration for future use
    ga.save_config("ga_config.json")
