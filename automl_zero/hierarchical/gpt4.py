import random
import numpy as np

# Define the base mathematical functions (OPs)
def add(x, y):
 return x + y

def subtract(x, y):
 return x - y

def multiply(x, y):
 return x * y

def divide(x, y):
 return x / y if y != 0 else 1

# Define the hierarchical search space
OPS = [add, subtract, multiply, divide]
META_OPS = [OPS] # Add more levels of hierarchy as needed

# Define the Gene class
class Gene:
    def __init__(self, level=0):
        self.level = level
        self.op = random.choice(META_OPS[level])

    def mutate(self):
        self.op = random.choice(META_OPS[self.level])

# Define the Individual class
class Individual:
    def __init__(self, num_genes, gene_levels):
        self.genes = [Gene(level=random.choice(gene_levels)) for _ in range(num_genes)]

    def crossover(self, other):
        crossover_point = random.randint(0, len(self.genes))
        child_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child = Individual(len(child_genes), gene_levels=[gene.level for gene in child_genes])
        child.genes = child_genes
        return child

    def mutate(self):
        gene_to_mutate = random.choice(self.genes)
        gene_to_mutate.mutate()

    def evaluate(self, x, y):
        result = x
        for gene in self.genes:
        result = gene.op(result, y)
        return result

# Define the main function
def main():
    population_size = 100
    num_genes = 5
    gene_levels = [0] # Add more levels as needed
    generations = 100

 # Initialize the population
    population = [Individual(num_genes, gene_levels) for _ in range(population_size)]

    # Main loop
    for generation in range(generations):
    # Evaluate the population
    fitness = [ind.evaluate(np.random.rand(), np.random.rand()) for ind in population]

    # Select the best individuals
    best_individuals = sorted(population, key=lambda ind: fitness[ind], reverse=True)[:population_size // 2]

    # Create the next generation
    new_population = []
    for _ in range(population_size // 2):
    parent1 = random.choice(best_individuals)
    parent2 = random.choice(best_individuals)
    child = parent1.crossover(parent2)
    child.mutate()
    new_population.append(child)

    population = new_population




if __name__ == "__main__":
    main()


import numpy as np

# Define the basic operations
op_dict = {1: np.add, 2: np.subtract, 3: np.log}

# Define a function to apply a series of operations
def apply_operations(gene, input_data):
 result = input_data
 for op_id in gene:
 if op_id in op_dict:
 result = op_dict[op_id](result, result)
 else:
 raise ValueError(f"Invalid operation ID: {op_id}")
 return result

# Define a function to apply a meta-level gene
def apply_meta_gene(meta_gene, input_data):
 result = input_data
 for gene in meta_gene:
 result = apply_operations(gene, result)
 return result

# Example usage
input_data = np.array([1, 2, 3, 4, 5])

# Define a meta-level gene representing a complex operation
meta_gene = [
 [1, 2], # np.add followed by np.subtract
 [3] # np.log
]

output = apply_meta_gene(meta_gene, input_data)
print(output)


import numpy as np

class Memory:
 def __init__(self, memory_size):
 self.memory = np.zeros(memory_size)
 self.memory_ref_dict = {}

 def add_subarray(self, key, start, end):
 self.memory_ref_dict[key] = (start, end)

 def get_subarray(self, key):
 start, end = self.memory_ref_dict[key]
 return self.memory[start:end]

class Gene:
 def __init__(self, input_keys, output_key, function):
 self.input_keys = input_keys
 self.output_key = output_key
 self.function = function

 def execute(self, memory):
 inputs = [memory.get_subarray(key) for key in self.input_keys]
 output = self.function(*inputs)
 memory.memory[memory.memory_ref_dict[self.output_key][0]:memory.memory_ref_dict[self.output_key][1]] = output

class MetaGene(Gene):
 def __init__(self, input_keys, output_key, gene_list):
 self.input_keys = input_keys
 self.output_key = output_key
 self.gene_list = gene_list

 def execute(self, memory):
 for gene in self.gene_list:
 gene.execute(memory)

# Example usage
memory = Memory(memory_size=100)
memory.add_subarray("input1", 0, 10)
memory.add_subarray("input2", 10, 20)
memory.add_subarray("output", 20, 30)

def example_function(a, b):
 return a + b

gene1 = Gene(["input1", "input2"], "output", example_function)
gene1.execute(memory)

print(memory.get_subarray("output"))