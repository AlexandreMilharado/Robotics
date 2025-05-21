from math import sqrt

# Span of Universe
MAX_VALUE_WEIGHT = 1                                        # Maximum Value for Gene
MIN_VALUE_WEIGHT = -1                                       # Minimum Value for Gene


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
    def euclidean(self, p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def distance_from_all(self, population):
        def path_distance(path1, path2):
            return sum(self.euclidean(p1, p2) for p1, p2 in zip(path1, path2)) / len(path1) if len(path1) != 0 else 0
        
        return sum(path_distance(self.path, other.path) for other in population if other != self) / (len(population) - 1)

    # def position_diff(self):
    #     diff = 0
    #     for i in range(len(self.path) - 1):
    #         diff += self.euclidean(self.path[i + 1], self.path[i])

    #     return diff / (9.53 * len(self.path)) if len(self.path) != 0 else 0
    def position_diff(self, threshold_dispersion=0.05, ratio_threshold=2.0):
        if len(self.path) < 10:
            return False  # Not enough data

        split_index = int(len(self.path) * 0.3)
        early_path = self.path[:split_index]
        late_path = self.path[split_index:]

        def dispersion(positions):
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        early_disp = dispersion(early_path)
        late_disp = dispersion(late_path)

        # Circling if early movement was broad, then got "stuck" in a tight loop
        if early_disp > threshold_dispersion and late_disp < (early_disp / ratio_threshold):
            return True

        return False

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
        return diff / (len(self.weights) * (MAX_VALUE_WEIGHT - MIN_VALUE_WEIGHT))
    
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
    def sort_individuals_by_id(cls, individuals):
        return sorted(individuals,
                    key=lambda individual: individual.id,
                    reverse=False)
    
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