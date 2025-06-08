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
        """
        Calculates the Euclidean distance between two points.

        Parameters
        ----------
        p1 : tuple
            The first point.
        p2 : tuple
            The second point.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def distance_from_all(self, population):
        """
        Calculates the average path distance from this individual to all other individuals in the population.

        This method computes the average Euclidean distance between the path of the current individual and 
        the paths of all other individuals in the given population. It uses the `path_distance` helper function 
        to calculate the path distance between two paths, which is the sum of Euclidean distances between 
        corresponding points in the paths, divided by the length of the path.

        Parameters
        ----------
        population : list
            A list of individuals in the population.

        Returns
        -------
        float
            The average path distance from this individual to all other individuals in the population.
        """

        def path_distance(path1, path2):
            return sum(self.euclidean(p1, p2) for p1, p2 in zip(path1, path2)) / len(path1) if len(path1) != 0 else 0
        
        return sum(path_distance(self.path, other.path) for other in population if other != self) / (len(population) - 1)

    # def position_diff(self):
    #     diff = 0
    #     for i in range(len(self.path) - 1):
    #         diff += self.euclidean(self.path[i + 1], self.path[i])

    #     return diff / (9.53 * len(self.path)) if len(self.path) != 0 else 0
    def position_diff(self, threshold_dispersion=0.05, ratio_threshold=2.0):
        """
        Determines if the individual is exhibiting a circling behavior based on its path dispersion.

        This function analyzes the path taken by the individual and checks if there is a significant 
        reduction in movement dispersion, suggesting a transition from broad movement to being stuck 
        in a tight loop, which is indicative of circling behavior.

        Parameters
        ----------
        threshold_dispersion : float, optional
            The minimum dispersion in the early path to consider it "broad" (default is 0.05).
        ratio_threshold : float, optional
            The factor by which the early path dispersion must exceed the late path dispersion to 
            classify the behavior as circling (default is 2.0).

        Returns
        -------
        bool
            True if the individual is circling, False otherwise.
        """

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
        """
        Convert the individual to a list of its attributes.

        Returns
        -------
        list
            A list containing the individual's generation number, id, fitness, weights, diversity, and line percentage.
        """
        return [self.gen_number,
                self.id,
                self.fitness,
                self.weights,
                self.diversity,
                self.line_percentage,
                ]    

    @classmethod
    def sort_individuals(cls, individuals):
        """
        Sorts a list of individuals by their fitness in descending order.

        Parameters
        ----------
        individuals : list
            A list of individuals to be sorted.

        Returns
        -------
        list
            The sorted list of individuals.
        """
        return sorted(individuals,
                    key=lambda individual: individual.fitness,
                    reverse=True)
    
    @classmethod
    def sort_individuals_by_id(cls, individuals):
        """
        Sorts a list of individuals by their ID in ascending order.

        Parameters
        ----------
        individuals : list
            A list of individuals to be sorted.

        Returns
        -------
        list
            The sorted list of individuals.
        """
        return sorted(individuals,
                    key=lambda individual: individual.id,
                    reverse=False)
    
    @classmethod
    def sort_individuals_by_fitness_list(cls, individuals, fitness_l):
        """
        Sorts a list of individuals based on their fitness scores in descending order.

        This method takes a list of individuals and a corresponding list of fitness scores,
        sorts the fitness scores in descending order, and reorders the individuals accordingly.

        Parameters
        ----------
        individuals : list
            A list of individuals to be sorted.
        fitness_l : list
            A list of tuples where each tuple contains the index of the individual and its fitness score.

        Returns
        -------
        list
            A list of individuals sorted by their fitness scores in descending order.
        """

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