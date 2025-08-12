# Genetic Algorithm Framework

A comprehensive, modular Python framework for solving optimization problems using Genetic Algorithms (GA). This framework provides flexible configuration options, multiple selection strategies, crossover methods, mutation operators, and detailed result analysis capabilities.

## Features

### Core Capabilities
- **Customizable Fitness Functions**: Abstract base class for defining problem-specific evaluation functions
- **Multiple Selection Strategies**: Tournament, roulette wheel, and rank-based selection
- **Various Crossover Methods**: Single-point, two-point, and uniform crossover
- **Flexible Mutation Operators**: Support for numeric, binary, and string chromosome representations
- **Elitism Support**: Preserve top-performing individuals across generations
- **Comprehensive Statistics**: Track and log population metrics across generations

### Configuration & Persistence
- **Full Parameter Control**: Population size, mutation rate, crossover rate, number of generations
- **Save/Load Configurations**: JSON-based configuration management
- **Result Persistence**: Save results in JSON or pickle formats
- **Detailed Reporting**: Generate comprehensive execution reports

### Visualization & Analysis
- **Fitness Evolution Plots**: Track fitness improvement over generations
- **Statistical Analysis**: Best, average, worst fitness with standard deviation
- **Performance Metrics**: Execution time tracking and improvement analysis

## Installation

1. Clone or download the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from ga_framework import GeneticAlgorithm, GAConfig, SelectionMethod, CrossoverMethod, MutationType
from ga_framework import SphereFunction

# Configure the genetic algorithm
config = GAConfig(
    population_size=100,
    num_generations=50,
    mutation_rate=0.02,
    crossover_rate=0.8,
    selection_method=SelectionMethod.TOURNAMENT,
    crossover_method=CrossoverMethod.SINGLE_POINT,
    mutation_type=MutationType.NUMERIC,
    elitism_count=5
)

# Define the optimization problem
fitness_func = SphereFunction(dimensions=5, bounds=(-10, 10))

# Run the optimization
ga = GeneticAlgorithm(fitness_func, config)
results = ga.run()

# Analyze results
print(results.generate_report())
results.plot_fitness_evolution()
```

### Custom Fitness Function

```python
from ga_framework import FitnessFunction, Individual
import random

class CustomProblem(FitnessFunction):
    def __init__(self, target_sum=100):
        self.target_sum = target_sum
    
    def evaluate(self, individual: Individual) -> float:
        # Maximize fitness when sum approaches target
        current_sum = sum(individual.chromosome)
        return -abs(current_sum - self.target_sum)
    
    def create_individual(self) -> Individual:
        # Create random individual with 10 values between 0-20
        chromosome = [random.uniform(0, 20) for _ in range(10)]
        return Individual(chromosome)

# Use custom fitness function
fitness_func = CustomProblem(target_sum=150)
ga = GeneticAlgorithm(fitness_func, config)
results = ga.run()
```

## Configuration Options

### GAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 100 | Number of individuals in population |
| `mutation_rate` | float | 0.01 | Probability of mutation per gene |
| `crossover_rate` | float | 0.8 | Probability of crossover between parents |
| `selection_method` | SelectionMethod | TOURNAMENT | Selection strategy |
| `crossover_method` | CrossoverMethod | SINGLE_POINT | Crossover technique |
| `mutation_type` | MutationType | NUMERIC | Type of mutation operator |
| `num_generations` | int | 100 | Number of evolution generations |
| `elitism_count` | int | 2 | Number of elite individuals to preserve |
| `tournament_size` | int | 3 | Tournament size for tournament selection |

### Selection Methods
- **TOURNAMENT**: Select best individual from random tournament
- **ROULETTE**: Fitness-proportionate selection with roulette wheel
- **RANK**: Rank-based selection to handle negative fitness values

### Crossover Methods
- **SINGLE_POINT**: Single crossover point dividing parents
- **TWO_POINT**: Two crossover points creating three segments
- **UNIFORM**: Gene-by-gene probabilistic crossover

### Mutation Types
- **NUMERIC**: Gaussian mutation for floating-point values
- **BINARY**: Bit-flip mutation for binary chromosomes
- **STRING**: Character substitution for string representations

## Built-in Test Functions

### Sphere Function
Classic optimization benchmark - minimize sum of squares:
```python
fitness_func = SphereFunction(dimensions=10, bounds=(-10, 10))
```

### Rastrigin Function
Multimodal test function with many local optima:
```python
fitness_func = RastriginFunction(dimensions=10, bounds=(-5.12, 5.12))
```

### OneMax Function
Binary optimization - maximize number of 1s:
```python
fitness_func = OneMaxFunction(length=50)
```

## Advanced Usage

### Configuration Management

```python
# Save configuration for reuse
config = GAConfig(population_size=200, num_generations=100)
ga = GeneticAlgorithm(fitness_func, config)
ga.save_config("my_config.json")

# Load configuration
loaded_config = GeneticAlgorithm.load_config("my_config.json")
ga = GeneticAlgorithm(fitness_func, loaded_config)
```

### Result Analysis

```python
# Run optimization
results = ga.run()

# Generate detailed report
report = results.generate_report()
print(report)

# Get fitness history for custom analysis
best_fitness, avg_fitness = results.get_fitness_history()

# Save results
results.save_to_json("results.json")
results.save_to_pickle("results.pkl")

# Load previous results
previous_results = GAResults.load_from_pickle("results.pkl")

# Create fitness evolution plot
results.plot_fitness_evolution("fitness_chart.png")
```

### Logging Configuration

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Run with detailed logging
ga = GeneticAlgorithm(fitness_func, config)
results = ga.run()  # Will show generation-by-generation progress
```

## Example Applications

### 1. Function Optimization
```python
# Minimize Rastrigin function
config = GAConfig(
    population_size=150,
    num_generations=200,
    mutation_rate=0.01,
    selection_method=SelectionMethod.TOURNAMENT
)

fitness_func = RastriginFunction(dimensions=20)
ga = GeneticAlgorithm(fitness_func, config)
results = ga.run()
```

### 2. Binary String Optimization
```python
# Maximize 1s in binary string
config = GAConfig(
    population_size=100,
    num_generations=50,
    mutation_rate=0.05,
    mutation_type=MutationType.BINARY,
    crossover_method=CrossoverMethod.UNIFORM
)

fitness_func = OneMaxFunction(length=100)
ga = GeneticAlgorithm(fitness_func, config)
results = ga.run()
```

### 3. Custom Constraint Problem
```python
class ConstrainedOptimization(FitnessFunction):
    def evaluate(self, individual: Individual) -> float:
        x, y, z = individual.chromosome
        
        # Objective: maximize x + y + z
        objective = x + y + z
        
        # Constraint: x^2 + y^2 + z^2 <= 100
        constraint_violation = max(0, x**2 + y**2 + z**2 - 100)
        
        # Penalty for constraint violation
        return objective - 1000 * constraint_violation
    
    def create_individual(self) -> Individual:
        return Individual([random.uniform(-10, 10) for _ in range(3)])
```

## Performance Tips

1. **Population Size**: Larger populations explore more but take longer
2. **Mutation Rate**: Higher rates increase exploration but may disrupt good solutions
3. **Elitism**: Preserve 2-10% of population as elite individuals
4. **Selection Pressure**: Tournament selection with size 3-7 works well for most problems
5. **Generations**: Monitor convergence - stop early if no improvement

## API Reference

### Classes

- **`GeneticAlgorithm`**: Main framework class
- **`GAConfig`**: Configuration container
- **`GAResults`**: Results storage and analysis
- **`Individual`**: Represents a chromosome with fitness
- **`FitnessFunction`**: Abstract base for fitness functions
- **`SelectionOperator`**: Selection strategies
- **`CrossoverOperator`**: Crossover methods
- **`MutationOperator`**: Mutation techniques

### Key Methods

- **`GeneticAlgorithm.run()`**: Execute the genetic algorithm
- **`GAResults.generate_report()`**: Create detailed text report
- **`GAResults.plot_fitness_evolution()`**: Visualize fitness over time
- **`GAConfig.to_dict()`/`from_dict()`**: Serialize configuration

## Contributing

This framework is designed to be easily extensible:

1. **Custom Fitness Functions**: Inherit from `FitnessFunction`
2. **New Selection Methods**: Add to `SelectionOperator`
3. **Additional Crossover**: Extend `CrossoverOperator`
4. **Custom Mutations**: Add to `MutationOperator`

## License

This project is provided as-is for educational and research purposes.

## Examples Directory Structure

```
genetic_algorithm_framework/
│
├── ga_framework.py          # Main framework code
├── requirements.txt         # Dependencies
├── README.md               # This file

```

## Getting Help

For questions about usage, implementation details, or extending the framework, please refer to:

1. The example fitness functions in the main code
2. The docstrings in each class and method
3. The configuration options detailed above

The framework is designed to be self-documenting with comprehensive docstrings and clear method signatures.
