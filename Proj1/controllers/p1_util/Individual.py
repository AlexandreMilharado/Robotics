from math import sqrt
import torch
import torch.nn as nn

SIMPLE_NET_INPUT = 2
SIMPLE_NET_HIDDEN = 4
SIMPLE_NET_OUTPUT = 2

# Span of Universe
MAX_VALUE_WEIGHT = 1                                        # Maximum Value for Gene
MIN_VALUE_WEIGHT = -1                                       # Minimum Value for Gene

# Crossover
GENES_NUMBER = 6                                            # Number of Genes

class SimpleNet(torch.nn.Module):
    def __init__(self):        
        super().__init__() 
        self.FC = nn.Sequential(
            nn.Linear(SIMPLE_NET_INPUT, SIMPLE_NET_HIDDEN),
            nn.Tanh(),  
            nn.Linear(SIMPLE_NET_HIDDEN, SIMPLE_NET_HIDDEN), 
            nn.Tanh(), 
            nn.Linear(SIMPLE_NET_HIDDEN, SIMPLE_NET_OUTPUT),  
            nn.Tanh()
        )

    def forward(self, x): 
        x = self.FC(x)
        return x


class Individual:                       
    id_counter = 0

# Inits
    def __init__(self, gen_number, id, weights, fitness = 0, path = [], diversity = 0):
        """
        Class constructor.

        Parameters
        ----------
        id : int
            Unique identifier for the individual.
        weights : list
            List of weights for the individual.
        fitness : float
            Fitness value of the individual.
        path : list
            The path taken by the individual during the simulation.
        diversity : float
            The genetic diversity of the individual.

        Notes
        -----
        This method should not be called directly, use `create_individual` instead.
        """
        self.gen_number = gen_number
        self.id = id
        self.weights = weights
        self.fitness = fitness
        self.path = path
        self.diversity = diversity
        self.line_percentage = 0

# Setters
    def add_path(self, position):
        """
        Adds a new position to the path taken by the individual.

        Parameters
        ----------
        position : tuple
            The new position to add to the path.
        """
        self.path.append(position)

    def set_diversity(self, diversity):
        """
        Sets the genetic diversity of the individual.

        Parameters
        ----------
        diversity : float
            The genetic diversity of the individual.
        """
        self.diversity = diversity

    def reset_path(self):
        """
        Resets the path of the individual to an empty list.

        This method should be called after the fitness evaluation of the individual is finished.
        """
        self.path = []

    def set_line_percentage(self, line_percentage):
        self.line_percentage = line_percentage

    def reset_fitness(self):
        """
        Resets the fitness of the individual to 0.

        This method should be called after the selection process of the generation is finished.
        """
        self.fitness = 0

# Getters
    def dna_diff(self, other):
        """
        Calculates the average absolute difference between the weights of this individual and the weights of another individual.

        Parameters
        ----------
        other : Individual
            The other individual to compare the weights with.

        Returns
        -------
        float
            The average absolute difference between the weights of this individual and the weights of the other individual.
        """
        diff = 0
        for i in range(len(self.weights)):
            diff += abs(self.weights[i] - other.weights[i])
        return diff / (GENES_NUMBER * (MAX_VALUE_WEIGHT - MIN_VALUE_WEIGHT))
    
    def to_list(self):
        return [self.gen_number,
                self.id,
                self.fitness,
                self.weights,
                self.diversity,
                self.line_percentage,
                ]    

    @classmethod
    def sort_individuals(cls, individuals):
        return sorted(individuals,
                    key=lambda individual: individual.fitness,
                    reverse=True)
    
    @classmethod
    def sort_individuals_by_fitness_list(cls, individuals, fitness_l):
        sorted_fitness = sorted(fitness_l,
                                key=lambda fitness: fitness[1],
                                reverse=True)
        
        sorted_individuals = []
        for i, fitness in sorted_fitness:
            sorted_individuals.append(individuals[i])

        return sorted_individuals

# Object methods    
    def __str__(self):
        """
        Returns a string representation of the individual, showing its ID, weights and fitness.

        Returns
        -------
        str
            A string representation of the individual.
        """
        return f"{{ID: {self.id}, Weights: {self.weights}, Fitness: {self.fitness}}}"