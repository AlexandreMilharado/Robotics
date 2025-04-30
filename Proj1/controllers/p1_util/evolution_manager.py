import ast
from math import sqrt
import os
import pickle
import numpy as np
import pandas as pd
from controller import Supervisor
from p1_util.robot_class import Agent
from functools import partial

# Random
np.random.seed(105946)                                      # Seed for Reproducibility

# Span of Universe
MAX_VALUE_WEIGHT = 1                                        # Maximum Value for Gene
MIN_VALUE_WEIGHT = -1                                       # Minimum Value for Gene

# Crossover
GENES_NUMBER = 6                                            # Number of Genes
CROSSOVER_ARITHMETIC_MAX = 1.5                              # Max Value for Alpha
CROSSOVER_ARITHMETIC_MIN = -0.5                             # Min Value for Alpha

# Mutation
MUTATION_VARIANCE = MAX_VALUE_WEIGHT/2                      # Variance for Mutation

# Fitness Reward
REWARD_PATH = 0.1                                           # Reward for Path Difference
PENALTY_EXPLORE = 0.5                                       # Penalty for not exploring enough
PENALTY_DNA = 0.15                                          # Penalty for DNA Difference

# Individuals History
INDIVIDUALS_HISTORY_PATH = "../p1_util/evolutionary.csv"    # Path for Individuals History
BEST_INDIVIDUAL_PATH = "../p1_util/best_individual.pkl"     # Path for Best Individual


def random_orientation():                                       # USELESS
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):                 # USELESS
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return (x, y, z)

class Individual:                       
    id_counter = 0
    weights_history = []

# Inits
    def __init__(self, id, weights, fitness = 0, path = [], diversity = 0):
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
        self.id = id
        self.weights = weights
        self.fitness = fitness
        self.path = path
        self.diversity = diversity

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

    def reset_fitness(self):
        """
        Resets the fitness of the individual to 0.

        This method should be called after the selection process of the generation is finished.
        """
        self.fitness = 0

# Getters
    def euclidean(self, p1, p2):
        """
        Calculates the Euclidean distance between two points in 2D space.

        Parameters
        ----------
        p1 : tuple
            A tuple containing the x and y coordinates of the first point.
        p2 : tuple
            A tuple containing the x and y coordinates of the second point.

        Returns
        -------
        float
            The Euclidean distance between the two points.
        """
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def distance_from_all(self, population):
        """
        Calculates the average Euclidean distance between the path of this individual and the paths of all other individuals in the population.

        Parameters
        ----------
        population : list
            List of all individuals in the population.

        Returns
        -------
        float
            Average Euclidean distance between the path of this individual and the paths of all other individuals in the population.
        """
        def path_distance(path1, path2):
            return sum(self.euclidean(p1, p2) for p1, p2 in zip(path1, path2)) / len(path1)
        
        return sum(path_distance(self.path, other.path) for other in population if other != self) / (len(population) - 1)

    def position_diff(self):
        """
        Calculates the average Euclidean distance between consecutive points in the path of this individual.

        Returns
        -------
        float
            Average Euclidean distance between consecutive points in the path of this individual.
        """
        diff = 0
        for i in range(len(self.path) - 1):
            diff += self.euclidean(self.path[i + 1], self.path[i])

        return diff / (9.53 * len(self.path))

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
    
# Class methods
    @classmethod
    def set_id_counter(cls, number):
        """
        Sets the ID counter for the class to the specified number.

        Parameters
        ----------
        number : int
            The new value for the ID counter.

        Notes
        -----
        This method should only be called once during the loading of population.
        """
        cls.id_counter = number

    @classmethod
    def create_individual(cls, weights):
        """
        Creates a new individual with the given weights.

        Parameters
        ----------
        weights : list
            List of weights for the individual.

        Returns
        -------
        Individual
            A new individual with the given weights, or None if the weights are repeated.
        """
        ws = ""
        for w in weights:
            ws += str(w)
        if ws in cls.weights_history:
            return None

        cls.weights_history.append(ws)
        cls.id_counter += 1
        return Individual(cls.id_counter, weights)

# Object methods
    def __eq__(self, other):
        """
        Compares two individuals by their ID.

        Parameters
        ----------
        other : Individual
            The other individual to compare with.

        Returns
        -------
        bool
            True if the IDs of the two individuals are the same, False otherwise.
        """
        return self.id == other.id
    
    def __str__(self):
        """
        Returns a string representation of the individual, showing its ID, weights and fitness.

        Returns
        -------
        str
            A string representation of the individual.
        """
        return f"{{ID: {self.id}, Weights: {self.weights}, Fitness: {self.fitness}}}"
        


class Evolution_Manager:
    def __init__(self, timestep_multiplier):
        """
        Class Contructor.

        Parameters
        ----------
        timestep_multiplier : float
            The multiplier for the Webots timestep.

        Attributes
        ----------
        supervisor : Supervisor
            The Webots supervisor.
        timestep : int
            The timestep for the simulation.
        agent : Agent
            The agent object for getting sensors and motors.
        history : list
            A list of lists containing the generation number, ID, fitness, weights, and diversity of each individual.

        Notes
        -----
        To load the best individual and testing it in Webots, no further steps are needed.
        To continue or start training `load_train_params` must be called, before calling `train_all`.
        """
        self.supervisor : Supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep() * timestep_multiplier)
        self.agent : Agent = Agent(self.supervisor, self.timestep)
        self.history = []

    def load_train_params(self, generation_converge_stop, population_size, selection_number, parents_number, mutation_rate, mutation_alter_rate, evaluation_time):
        """
        Initializes the training parameters for the evolutionary algorithm.

        Parameters
        ----------
        generation_converge_stop : int
            Number of generations to run before stopping the training.
        population_size : int
            Number of individuals in each generation.
        selection_number : int
            Number of individuals selected to keep for the next generation.
        parents_number : int
            Number of parents used for crossover to produce new individuals.
        mutation_rate : float
            Probability of mutation occurring in an individual.
        mutation_alter_rate : float
            Rate at which mutation alters the genes of an individual.
        evaluation_time : int
            Time allocated for evaluating each individual in the simulation.
        
        Notes
        -----
        Must be called before calling `train_all`.
        """
        self.individuals = [self._generate_individual(GENES_NUMBER) for _ in range(population_size)]
        self.generation_converge_stop = generation_converge_stop
        self.evaluation_time = evaluation_time
        self.selection_number = selection_number
        self.parents_number = parents_number
        self.mutation_rate = mutation_rate
        self.mutation_alter_rate = mutation_alter_rate
        self.mutation_individual = selection_number/population_size
        self.generation_start_number = 0

    def reset(self, rotation, translation):
        """
        Resets the simulation environment by setting the agent's rotation and translation
        to the specified values and resetting the simulation physics.

        Parameters
        ----------
        rotation : list
            A list of 4 floats representing the rotation of the robot in the form [x, y, z, angle].
        translation : list
            A list of 3 floats representing the translation of the robot in the form [x, y, z].

        Notes
        -----
        This function steps the simulation forward twice after resetting the physics to ensure
        the changes take effect.
        """
        self.agent.reset(rotation, translation)
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)
        self.supervisor.step(self.timestep)

    def _update_generation(self, gen_number, individuals):
        """
        Updates the history of the evolution manager with the details of the current generation.

        Parameters
        ----------
        gen_number : int
            The generation number being updated.
        individuals : list
            List of individuals whose data will be recorded in the history. Each individual
            is expected to have attributes: id, fitness, weights, and diversity.

        Notes
        -----
        This function records the generation number, ID, fitness, weights, and diversity
        of each individual in the history for later analysis or storage.
        """
        for individual in individuals:
            self.history.append([gen_number, individual.id, individual.fitness, individual.weights, individual.diversity])


    def save_training(self):
        """
        Saves the history of the evolution manager to a CSV file.

        Notes
        -----
        The history is saved to a file at the path defined by the
        `INDIVIDUALS_HISTORY_PATH` constant. The file is created if it does not
        exist, and the header is added if it does not exist already. The history
        is saved in the form of a pandas DataFrame, with columns for the
        generation number, ID, fitness, weights, and diversity of each
        individual.
        """
        file_path = INDIVIDUALS_HISTORY_PATH
        header = not os.path.exists(file_path)
        pd.DataFrame(data=self.history, columns=["Gen Number", "ID", "Fitness", "Weights", "Diversity"]).to_csv(file_path, mode='a', header=header, index=False)

    def save_best_individual(self):
        """
        Saves the best individual from the current generation to a pickle file.

        Notes
        -----
        The best individual is saved to a file at the path defined by the
        `BEST_INDIVIDUAL_PATH` constant. The file is overwritten if it already
        exists. The best individual is determined by sorting the individuals
        by their fitness in descending order and picking the first one.
        """
        with open(BEST_INDIVIDUAL_PATH, 'wb') as f:
            pickle.dump(self.sort_individuals(self.individuals)[0], f)

    def load_training(self):
        """
        Loads the training data from the individuals history CSV file and updates the 
        evolution manager's state accordingly.

        Notes
        -----
        The function reads the CSV file specified by the `INDIVIDUALS_HISTORY_PATH` 
        constant to retrieve the saved history of individuals. It updates the list of 
        individuals with the last entries in the CSV corresponding to the number of 
        individuals currently in the manager. It also sets the ID counter for the 
        `Individual` class to the next available ID and updates the generation start 
        number for the continuing training process.
        """
        df = pd.read_csv(INDIVIDUALS_HISTORY_PATH)
        id_counter = df["ID"].max() + 1
        gen_number = df["Gen Number"].max() + 1
        rows = df.iloc[-len(self.individuals):]

        individuals_rows = rows.apply(lambda row: Individual(int(row["ID"]), ast.literal_eval(row["Weights"]), float(row["Fitness"]), float(row["Diversity"])), axis=1)
        self.individuals = individuals_rows.tolist()

        Individual.set_id_counter(id_counter)
        self.generation_start_number = gen_number
        

    def load_best_individual(cls):
        """
        Loads the best individual from the pickle file saved by the `save_best_individual` method.

        Returns
        -------
        Individual
            The best individual saved by the `save_best_individual` method.

        Notes
        -----
        The path to the pickle file is defined by the `BEST_INDIVIDUAL_PATH` constant.
        """
        with open(BEST_INDIVIDUAL_PATH, 'rb') as f:
            return pickle.load(f)
        
    def run_individual(self, individual):
        """
        Runs the given individual in the simulation.

        Parameters
        ----------
        individual : Individual
            The individual to run in the simulation.

        Notes
        -----
        The function will run the individual in the simulation until the agent collides with an obstacle.
        The agent is controlled by the weights of the given individual.
        """
        while not self.agent.collided():
            # Read Sensors
            sensors_inputs = self._read_sensors()

            # Control Motors
            self._run_step(individual.weights, sensors_inputs)

    def _generate_individual(self, size):
        """
        Generates a new individual with random weights.

        Parameters
        ----------
        size : int
            The number of weights to generate for the individual.

        Returns
        -------
        Individual
            A new individual with random weights within the defined bounds.
        """
        def generate_weights(size):
            return [np.random.uniform(MIN_VALUE_WEIGHT, MAX_VALUE_WEIGHT) for _ in range(size)]
        
        individual = Individual.create_individual(generate_weights(size))
        while individual is None:
            individual = Individual.create_individual(generate_weights(size))
        return individual

    def _run_step(self, weights, sensors_inputs):
        """
        Runs a step in the simulation with the given weights and sensor inputs.

        Parameters
        ----------
        weights : list
            The weights to use to control the motors.
        sensors_inputs : list
            The sensor inputs to use to control the motors.
        """
        p_1_e, p_2_e, p_3_e, p_1_d, p_2_d, p_3_d = weights
        s_e, s_d = sensors_inputs

        left_speed =  s_e * p_1_e + s_d * p_2_e + p_3_e
        right_speed = s_e * p_1_d + s_d * p_2_d + p_3_d

        self.agent.set_velocity_left_motor(left_speed, sensors_inputs)
        self.agent.set_velocity_right_motor(right_speed, sensors_inputs)
        
        self.supervisor.step(self.timestep)

    def _read_sensors(self):
        """
        Reads the values of the left and right ground sensors and returns a tuple containing a boolean for each sensor indicating whether the robot is not on a black line.

        Returns
        -------
        tuple
            A tuple containing two booleans indicating whether the robot is not on a black line for the left and right ground sensors respectively.
        """
        (left_sensor_value, right_sensor_value) = self.agent.get_ground_sensors_values()
        return (self.agent.is_not_on_black_line(left_sensor_value),
                self.agent.is_not_on_black_line(right_sensor_value))

    def _run_train_simulation(self, individual):
        """
        Runs the training simulation for the given individual.

        Parameters
        ----------
        individual : Individual
            The individual to run in the simulation.

        Notes
        -----
        The function will update the fitness of the individual based on the performance in the simulation.
        """
        def penalty_fitness_based_on_collision(fitness):
            return fitness / (limit_timestep - timesteps + 1) 

        def update_fitness(fitness):
            penalize = np.sign(angular_velocity) != np.sign(self.agent.get_angular_velocity()) and self.agent.left_motor != self.agent.right_motor
            return (fitness
                    + int((not sensors_inputs[0]) + (not sensors_inputs[1])) * self.agent.get_average_velocity()
                    - penalize * self.agent.get_max_velocity())
        
    
        fitness = individual.fitness
        timesteps = 0
        limit_timestep = int((self.evaluation_time * 1000) / self.timestep + 0.5)
        angular_velocity = self.agent.get_angular_velocity()

        while (timesteps < limit_timestep and
                not self.agent.collided()):
            
            # Read Sensors
            sensors_inputs  = self._read_sensors()

            # Calculate Fitness
            fitness = update_fitness(fitness)

            # Control Motors
            self._run_step(individual.weights, sensors_inputs)

            # Update States
            angular_velocity = self.agent.get_angular_velocity()
            individual.add_path(self.supervisor.getSelf().getPosition()[:2])
            timesteps += 1

        individual.fitness = penalty_fitness_based_on_collision(fitness)
    

    def train_one_individual(self, individual, rotation, translation):
        """
        Trains one individual in the simulation.

        Parameters
        ----------
        individual : Individual
            The individual to train in the simulation.
        rotation : tuple
            The rotation of the robot in the simulation.
        translation : tuple
            The translation of the robot in the simulation.
        """
        failed = True

        while failed:
            self.reset(rotation, translation)
            self._run_train_simulation(individual)
            failed = len(individual.path) == 0

    def normalise_fitness(self):
        """
        Normalizes the fitness of the individuals in the population.

        The normalization is done by dividing the fitness of each individual
        by the maximum possible fitness that can be obtained in the simulation
        (i.e. the length of the path and the time taken to traverse it).
        """
        dem = 2 * 9.53 * int((self.evaluation_time * 1000) / self.timestep + 0.5)
        for individual in self.individuals:
            individual.fitness = individual.fitness / dem

    def reward_fitness_based_on_path(self, sorted_individuals):
        """
        Rewards the fitness of the individuals in the population based on their path difference.

        The reward is calculated as the maximum distance of the individual from all other individuals
        divided by the maximum possible distance (i.e. the length of the path and the time taken to
        traverse it). The reward is then multiplied by the maximum possible fitness (i.e. the length
        of the path and the time taken to traverse it) to get the final reward.

        The reward is proportional to the distance of the individual from all other individuals and
        inversely proportional to the maximum possible distance.

        Parameters
        ----------
        sorted_individuals : list
            A list of `Individual` objects sorted by their fitness in descending order.

        Returns
        -------
        rewards : list
            A list of rewards for the individuals in the population.
        """
        distances = []
        for individual in sorted_individuals:
            distances.append(individual.distance_from_all(sorted_individuals))

        max_value = max(distances)
        distances = [distance / max_value for distance in distances]

        rewards = []
        for i, individual in enumerate(sorted_individuals):
            rewards.append(REWARD_PATH * distances[i])
        return rewards
    
    def penalty_fitness_based_on_exploration(self, sorted_individuals, penalties):
        """
        Penalizes the fitness of the individuals in the population based on their path difference.

        The penalty is calculated as the maximum distance of the individual from all other individuals
        divided by the maximum possible distance (i.e. the length of the path and the time taken to
        traverse it). The penalty is then multiplied by the maximum possible fitness (i.e. the length
        of the path and the time taken to traverse it) to get the final penalty.

        The penalty is proportional to the distance of the individual from all other individuals and
        inversely proportional to the maximum possible distance.

        Parameters
        ----------
        sorted_individuals : list
            A list of `Individual` objects sorted by their fitness in descending order.
        penalties : list
            A list of penalties for the individuals in the population.

        Returns
        -------
        penalties : list
            The list of penalties for the individuals in the population.
        """
        for i, individual in enumerate(sorted_individuals):
            penalties[i] -= PENALTY_EXPLORE * individual.fitness * individual.position_diff()
        return penalties

    def penalty_fitness_based_on_dna(self, sorted_individuals, penalties):
        """
        Penalizes the fitness of the individuals in the population based on their genetic difference.

        The penalty is calculated as the genetic difference between the individual and all other individuals
        multiplied by the maximum possible fitness (i.e. the length of the path and the time taken to
        traverse it). The penalty is then multiplied by the maximum possible fitness (i.e. the length
        of the path and the time taken to traverse it) to get the final penalty.

        The penalty is proportional to the genetic difference of the individual from all other individuals
        and inversely proportional to the maximum possible distance.

        Parameters
        ----------
        sorted_individuals : list
            A list of `Individual` objects sorted by their fitness in descending order.
        penalties : list
            A list of penalties for the individuals in the population.

        Returns
        -------
        penalties : list
            The list of penalties for the individuals in the population.
        """
        for i in range(len(sorted_individuals) - 1):
            for j in range(i+1, len(sorted_individuals)):
                penalties[j] -= PENALTY_DNA * sorted_individuals[j].fitness * (1 - sorted_individuals[i].dna_diff(sorted_individuals[j]))
        return penalties
    
    def set_diversity(self, individuals):
        """
        Calculates and sets the genetic diversity for each individual in a population.

        Parameters
        ----------
        individuals : list
            A list of `Individual` objects for which the genetic diversity is to be calculated and set.
        """
        gene_matrix = np.array([individual.weights for individual in individuals])
        diversity = np.mean(np.std(gene_matrix, axis=0))
        for individual in individuals:
            individual.set_diversity(diversity)

    def train_one_generation(self, gen_number):
        """
        Trains one generation of individuals in the population.

        This method trains all individuals in the population by running the simulation with the
        current weights and updating the fitness of each individual. The fitness is then normalized
        and penalized based on the genetic difference and exploration. The best individuals are
        selected and used for crossover to produce new children. The children are then mutated to
        introduce genetic variation. The performance of the best individual is logged and the
        generation is updated.

        Parameters
        ----------
        gen_number : int
            The current generation number.

        Returns
        -------
        survivors : list
            A list of `Individual` objects which are the survivors of the selection process.
        """
        # Reset Paths
        for individual in self.individuals:
            individual.reset_path()

        # Train Individuals Center Map
        for individual in self.individuals:
            rotation, translation = self.agent.get_rotation_translation()
            self.train_one_individual(individual, rotation, translation)

        # Normalize Fitness
        self.normalise_fitness()

        # Penalize Fitness
        individuals = self.sort_individuals(self.individuals)
        rewards = self.reward_fitness_based_on_path(individuals)
        rewards = self.penalty_fitness_based_on_exploration(individuals, rewards)
        rewards = self.penalty_fitness_based_on_dna(individuals, rewards)

        for i in range(len(individuals)):
            individuals[i].fitness += rewards[i]

        # Sort Best Individuals
        sorted_individuals = self.sort_individuals(individuals)

        # Set Diversity for individuals
        self.set_diversity(sorted_individuals)

        # Selection
        survivors = sorted_individuals[:self.selection_number]
    
        # Crossover
        children = self.crossover(partial(self.sus, survivors),
                                self.arithmetic_crossover,
                                len(sorted_individuals) - self.selection_number)
        
        # Mutation
        self.mutation(self.mutate_alter_value, children)

        # Performance Log
        print(f"---GEN {gen_number}---")
        print(f"BEST INDIVIDUAL: {sorted_individuals[0]}")

        # Updating Individuals
        survivors.extend(children)
        self._update_generation(gen_number, sorted_individuals)

        return survivors
        

    def train_all(self):
        """
        Trains the population for a specified number of generations.

        This method starts training the population from the current generation number
        and continues until the specified number of generations has been reached.
        The best individual is saved to a pickle file after the training is finished.
        """

        generation_number = self.generation_start_number
        generation_stop = self.generation_converge_stop + self.generation_start_number
        while generation_number < generation_stop:
            survivors = self.train_one_generation(generation_number)
            
            # Updating new Individuals
            self.individuals = survivors

            generation_number += 1
        self.save_training()
        self.save_best_individual()


    def sort_individuals(self, individuals):
        """
        Sorts a list of individuals by their fitness in descending order.

        Parameters
        ----------
        individuals : list
            A list of `Individual` objects to be sorted.

        Returns
        -------
        list
            A list of `Individual` objects sorted by their fitness in descending order.
        """
        return sorted(individuals, key=lambda individual: individual.fitness, reverse=True)

    def sus(self, sorted_individuals):
        """
        Selects parents from a sorted list of individuals using stochastic universal sampling.

        This method calculates selection probabilities for each individual based on their
        ranking in the sorted list. It then selects parents by generating random numbers and
        comparing them to the cumulative selection probabilities.

        Parameters
        ----------
        sorted_individuals : list
            A list of `Individual` objects sorted by their fitness in descending order.

        Returns
        -------
        list
            A list of selected `Individual` objects to be used as parents.
        """
        dem = 0.5 * len(sorted_individuals) * (len(sorted_individuals) + 1)
        probs = np.arange(len(sorted_individuals), 0, -1) / dem
        probs = np.cumulative_sum(np.array(probs))
        parents = []
        for _ in range(self.parents_number):
            random_number = np.random.rand(1)
            for i, prob in enumerate(probs):
                if random_number < prob:
                    parents.append(sorted_individuals[i])

        return parents
    
    def arithmetic_crossover(self, parent1, parent2):
        """
        Performs an arithmetic crossover between two parents to generate a child.

        This method generates a random alpha value in the range [CROSSOVER_ARITHMETIC_MIN,
        CROSSOVER_ARITHMETIC_MAX] and calculates the corresponding beta value as 1 - alpha.
        The weights of the child are then calculated as a weighted sum of the weights of the
        two parents, where the weights are multiplied by alpha and beta respectively. The
        resulting weights are then clipped to the range [MIN_VALUE_WEIGHT, MAX_VALUE_WEIGHT].

        Parameters
        ----------
        parent1 : Individual
            The first parent individual.
        parent2 : Individual
            The second parent individual.

        Returns
        -------
        Individual
            A new individual with weights generated by the arithmetic crossover.
        """
        alpha = np.random.uniform(CROSSOVER_ARITHMETIC_MIN, CROSSOVER_ARITHMETIC_MAX)
        beta = 1 - alpha
        weights = []
        for i in range(GENES_NUMBER):
            weights.append(max(min(parent1.weights[i] * alpha + parent2.weights[i] * beta, MAX_VALUE_WEIGHT), MIN_VALUE_WEIGHT))
        return Individual.create_individual(weights) 

    def crossover(self, selection_parents, crossover_aux, children_number):
        """
        Generates children by performing crossover on selected parents.

        This function takes a selection method to choose parents and a crossover
        method to generate children. It iterates over the number of children to
        be created, applies the crossover method to pairs of parents, and appends
        the resulting children to a list.

        Parameters
        ----------
        selection_parents : callable
            A function that selects and returns a list of parent individuals.
        crossover_aux : callable
            A function that takes two parents and returns a child individual.
        children_number : int
            The number of children to generate.

        Returns
        -------
        list
            A list of `Individual` objects representing the generated children.
        """
        children = []

        parents = selection_parents()

        for i in range(children_number):
            child = crossover_aux(parents[i], parents[i+1])
            children.append(child)

        return children

    def individual_random(self, individual):
        """
        Generates a new individual with weights randomly generated from a normal distribution.

        If the provided individual is None, this function generates a new individual with
        weights randomly generated from a normal distribution with variance MUTATION_VARIANCE.
        It ensures that the weights are within the valid range [MIN_VALUE_WEIGHT, MAX_VALUE_WEIGHT].

        Parameters
        ----------
        individual : Individual
            The individual to be used as a base for generating the new individual.

        Returns
        -------
        Individual
            A new individual with weights randomly generated from a normal distribution.
        """
        while individual is None:
            weights = []
            for _ in range(GENES_NUMBER):
                random_value = np.random.normal(0, MUTATION_VARIANCE)
                while random_value < MIN_VALUE_WEIGHT or random_value > MAX_VALUE_WEIGHT:
                    random_value = np.random.normal(0, MUTATION_VARIANCE)
                weights.append(random_value) 
            individual = Individual.create_individual(weights)
        return individual

    def mutate_alter_value(self, child):
        """
        Mutates a child by altering some of its weights.

        This method iterates over the weights of the child and randomly alters
        some of them. The probability of alteration is given by the
        `mutation_alter_rate` attribute. The new value is randomly generated
        from a normal distribution with variance `MUTATION_VARIANCE` and mean
        equal to the current weight. The method ensures that the new weight is
        within the valid range [MIN_VALUE_WEIGHT, MAX_VALUE_WEIGHT].

        Parameters
        ----------
        child : Individual
            The individual to be mutated.

        Returns
        -------
        Individual
            The mutated individual.
        """
        for i, w in enumerate(child.weights):
            if np.random.rand() < self.mutation_alter_rate:
                random_value = np.random.normal(w, MUTATION_VARIANCE)
                while random_value < MIN_VALUE_WEIGHT or random_value > MAX_VALUE_WEIGHT:
                    random_value = np.random.normal(w, MUTATION_VARIANCE)
                child.weights[i] = random_value
        return child
         

    def mutation(self, mutation_aux, children):
        """
        Applies mutation to a list of child individuals.

        This method iterates over a list of child individuals and applies a mutation
        function to each child with a probability defined by the `mutation_rate`.
        If a child becomes None after mutation, a new individual with random weights
        is generated to replace it. This ensures that all children remain valid individuals.

        Parameters
        ----------
        mutation_aux : callable
            A function that takes an individual and returns a mutated individual.
        children : list
            A list of `Individual` objects to be mutated.
        """
        for i in range(len(children)):
            if np.random.rand() < self.mutation_rate and children[i] is not None:
                children[i] = mutation_aux(children[i])
            if children[i] is None:
                children[i] = self.individual_random(children[i])