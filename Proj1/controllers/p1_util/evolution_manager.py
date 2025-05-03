import ast
import numpy as np
import pandas as pd
from p1_util.Individual import Individual
from p1_util.History_Individuals import History_Individuals

from p1_util.robot_class import Agent
from functools import partial


np.random.seed(105946)                                                  # Seed for Reproducibility

BRAITENBERG = { "MAX_VALUE_WEIGHT": 1,                                  # Maximum Value for Gene
                "MIN_VALUE_WEIGHT" : -1,                                # Minimum Value for Gene
                "GENES_NUMBER" : 6,                                     # Number of Genes
                "PENALTY_DNA": 0.15,                                    # Penalty for DNA Difference
                "REWARD_DNA_GEN": 0.15,                                 # Reward for DNA Difference
                "CROSSOVER_ARITHMETIC_MAX": 1.5,                        # Max Value for Alpha
                "CROSSOVER_ARITHMETIC_MIN": -0.5,                       # Min Value for Alpha
                "MUTATION_VARIANCE": 0.5,                               # Variance for Mutation
               }


NETWORKS_SIMPLE = { "MAX_VALUE_WEIGHT": 1,                               # Maximum Value for Gene
                    "MIN_VALUE_WEIGHT" : -1,                             # Minimum Value for Gene
                    "GENES_NUMBER" : 6,                                  # Number of Genes
                    "PENALTY_DNA": 0.15,                                 # Penalty for DNA Difference
                    "CROSSOVER_ARITHMETIC_MAX": 1.5,                     # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": -0.5,                    # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,                            # Variance for Mutation
                 }

NETWORKS_COMPLEX = {"MAX_VALUE_WEIGHT": 1,                               # Maximum Value for Gene
                    "MIN_VALUE_WEIGHT" : -1,                             # Minimum Value for Gene
                    "GENES_NUMBER" : 6,                                  # Number of Genes
                    "PENALTY_DNA": 0.15,                                 # Penalty for DNA Difference
                    "CROSSOVER_ARITHMETIC_MAX": 1.5,                     # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": -0.5,                    # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,                            # Variance for Mutation
                    }

class Evolution_Manager:
    def __init__(self, SENSOR_TYPE, TIMESTEP_MULTIPLIER, INDIVIDUALS_HISTORY_PATH, BEST_INDIVIDUAL_PATH):
        self.agent : Agent = Agent(SENSOR_TYPE, TIMESTEP_MULTIPLIER)
        self.history = History_Individuals(INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH,
                                        BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH)

    def load_train_params(self, INDIVIDUAL_TYPE, population_size, generation_limit, evaluation_time, selection_number, mutation_rate, mutation_alter_rate):
        if INDIVIDUAL_TYPE == "BRAITENBERG":
            self.CONSTANT = BRAITENBERG
        elif INDIVIDUAL_TYPE == "NETWORKS_SIMPLE":
            self.CONSTANT = NETWORKS_SIMPLE
        else:
            self.CONSTANT = NETWORKS_COMPLEX

        self.generation_start_number = 0
        self.generation_limit = generation_limit

        self.evaluation_time = evaluation_time

        self.selection_number = selection_number

        self.mutation_rate = mutation_rate
        self.mutation_alter_rate = mutation_alter_rate

        self.current_gen_individuals = [self._generate_individual() for _ in range(population_size)]
        pass

    def load_training(self):
        df = pd.read_csv(self.history.INDIVIDUALS_HISTORY_PATH)
        rows = df.iloc[-len(self.current_gen_individuals):]

        individuals_rows = rows.apply(lambda row: Individual(int(row["Gen Number"]),
                                                             int(row["ID"]),
                                                             ast.literal_eval(row["Weights"])), axis=1)
        self.current_gen_individuals = individuals_rows.tolist()

        self.generation_start_number = df["Gen Number"].max() + 1

    def reset(self):
        self.agent.reset()

    def _generate_individual(self):
        def generate_weights(size):
            return [np.random.uniform(self.CONSTANT["MIN_VALUE_WEIGHT"], self.CONSTANT["MAX_VALUE_WEIGHT"]) for _ in range(size)]
        
        individual = self.history.create_individual(self.generation_start_number, generate_weights(self.CONSTANT["GENES_NUMBER"]))
        while individual is None:
            individual = Individual.create_individual(self.generation_start_number, generate_weights(self.CONSTANT["GENES_NUMBER"]))
        return individual
    
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
        parents_number = (len(self.current_gen_individuals) - self.selection_number) * 2
        for _ in range(parents_number):
            random_number = np.random.rand(1)
            for i, prob in enumerate(probs):
                if random_number < prob:
                    parents.append(sorted_individuals[i])

        return parents
    
    def arithmetic_crossover(self, parent1, parent2):
        alpha = np.random.uniform(self.CONSTANT["CROSSOVER_ARITHMETIC_MIN"], self.CONSTANT["CROSSOVER_ARITHMETIC_MAX"])
        beta = 1 - alpha
        weights = []
        for i in range(self.CONSTANT["GENES_NUMBER"]):
            weights.append(max(min(parent1.weights[i] * alpha + parent2.weights[i] * beta,
                                   self.CONSTANT["MAX_VALUE_WEIGHT"]),
                                self.CONSTANT["MIN_VALUE_WEIGHT"]))
        return weights

    def crossover(self, selection_parents, crossover_aux, children_number):
        children_weights = []

        parents = selection_parents()

        for i in range(children_number):
            child_weights = crossover_aux(parents[i], parents[i+1])
            children_weights.append(child_weights)

        return children_weights

    def mutate_alter_value(self, child_weights):
        for i, w in enumerate(child_weights):
            if np.random.rand() < self.mutation_alter_rate:
                random_value = np.random.normal(w, self.CONSTANT["MUTATION_VARIANCE"])

                while random_value < self.CONSTANT["MIN_VALUE_WEIGHT"] or random_value > self.CONSTANT["MAX_VALUE_WEIGHT"]:
                    random_value = np.random.normal(w, self.CONSTANT["MUTATION_VARIANCE"])
            
                child_weights[i] = random_value

        return child_weights
         

    def mutation(self, mutation_aux, children_weights):
        for i in range(len(children_weights)):
            if np.random.rand() < self.mutation_rate:
                children_weights[i] = mutation_aux(children_weights[i])

    def removing_duplicates(self, gen_number, individual_weights):
        individual = self.history.create_individual(gen_number, individual_weights)
        while individual is None:
            weights = []

            for _ in range(self.CONSTANT["GENES_NUMBER"]):
                random_value = np.random.normal(0, self.CONSTANT["MUTATION_VARIANCE"])

                while random_value < self.CONSTANT["MIN_VALUE_WEIGHT"] or random_value > self.CONSTANT["MAX_VALUE_WEIGHT"]:
                    random_value = np.random.normal(0, self.CONSTANT["MUTATION_VARIANCE"])
    
                weights.append(random_value)

            individual = self.history.create_individual(gen_number, individual_weights)

        return individual
    

    def start_fitness(self):
        return self.agent.get_max_velocity()

    def reward_fitness_on_black_line(self):
        sensors_readings =  self.agent.read_sensors()
        return self.agent.get_average_velocity() * sum([int(not reading) for reading in sensors_readings])

    def penalty_fitness_based_on_collision(self, fitness, limit_timestep, timesteps):
        return fitness / (limit_timestep - timesteps + 1) 
    
    def penalty_wiggly_movement(self, angular_velocity):
        return -self.agent.get_max_velocity() * (np.sign(angular_velocity) != np.sign(self.agent.get_angular_velocity()) and
                                                    self.agent.left_motor != self.agent.right_motor)
                
    def penalty_fitness_based_on_best_dna(self, sorted_individuals, penalties):
        for i in range(len(sorted_individuals) - 1):
            for j in range(i+1, len(sorted_individuals)):
                index, fitness = penalties[j]
                fitness -= self.CONSTANT["PENALTY_DNA"] * fitness * (1 - sorted_individuals[i].dna_diff(sorted_individuals[j]))
                penalties[j] = (index, fitness)

        return penalties
    
    def reward_fitness_based_on_generational_dna(self, individuals, fitness):
        def calculate_mean_weights():
            mean_weigths = [0 for _ in range(len(individuals[0].weights))]
            for individual in individuals:
                for i in range(len(individuals[0].weights)):
                    mean_weigths[i] += individual.weights[i]

            for i in range(len(mean_weigths)):
                mean_weigths[i] /= len(individuals[0].weights)

            return mean_weigths
        
        mean_weigths = calculate_mean_weights()
        max_difference = (self.CONSTANT["MAX_VALUE_WEIGHT"] - self.CONSTANT["MIN_VALUE_WEIGHT"]) * len(mean_weigths)
        
        for i, individual in enumerate(individuals):
            acc = 0
            for j in range(len(mean_weigths)):
                acc += abs(mean_weigths[j] - individual.weights[j])

            (index, value) = fitness[i]
            value += self.CONSTANT["REWARD_DNA_GEN"] * (acc / max_difference)
            fitness[i] = (index, value)

        return fitness

    def set_diversity(self, individuals):
        gene_matrix = np.array([individual.weights for individual in individuals])
        diversity = np.mean(np.std(gene_matrix, axis=0))
        for individual in individuals:
            individual.set_diversity(diversity)

    def set_line_percentage(self, individuals, line_percentage):
        for individual in individuals:
            individual.set_line_percentage(line_percentage)

    def train_individual(self, individual):
        def update_fitness(fitness, angular_velocity):
            return (fitness +
                    self.reward_fitness_on_black_line() +
                    self.penalty_wiggly_movement(angular_velocity))
        
        def normalise_fitness(fitness):
            return fitness / MAX_FITNESS_VALUE

        def update_stats(stats):
            stats["line_percentage"] += int(self.agent.is_on_black_line_map())

        angular_velocity = self.agent.get_angular_velocity()
        limit_timestep = int((self.evaluation_time * 1000) / self.agent.timestep + 0.5)
        MAX_FITNESS_VALUE = 2 * 9.53 * limit_timestep
        fitness = self.start_fitness()
        stats = {"line_percentage": 0}
        timesteps = 0 

        while (timesteps < limit_timestep and
                not self.agent.collided()):

            # Run 1 Step
            self.agent.run_individual(individual)

            # Fitness Calculation
            fitness = update_fitness(fitness, angular_velocity)

            # Updating States
            angular_velocity = self.agent.get_angular_velocity()
            individual.add_path(self.agent.supervisor.getSelf().getPosition()[:2])
            update_stats(stats)

            timesteps += 1
        # Fitness Calculation
        fitness = self.penalty_fitness_based_on_collision(fitness, limit_timestep, timesteps)
        fitness = normalise_fitness(fitness)

        return {"FITNESS": fitness, "LINE_PERCENTAGE": stats["line_percentage"]/limit_timestep}

    def train_one_individual(self, individual):
        failed = True
        result = {}
        while failed:
            self.agent.reset()
            result = self.train_individual(individual)
            failed = len(individual.path) == 0
        return result

    def train_one_generation(self, gen_number):
        # Reset Paths
        for individual in self.current_gen_individuals:
            individual.reset_path()

        # Train Individuals Center Map
        fitness = []
        line_percentage = []
        for i, individual in enumerate(self.current_gen_individuals):
            result = self.train_one_individual(individual)
            fitness.append((i, result["FITNESS"]))
            line_percentage.append((i, result["LINE_PERCENTAGE"]))

        # Fitness Calculation
        individuals = Individual.sort_individuals_by_fitness_list(self.current_gen_individuals, fitness)
        fitness = self.penalty_fitness_based_on_best_dna(individuals, fitness)
        fitness = self.reward_fitness_based_on_generational_dna(individuals, fitness)

        # Setting Individuals Fitness
        for i in range(len(individuals)):
            individuals[i].fitness = fitness[i][1]
        
        # Setting Line Percentage
        self.set_line_percentage(individuals, line_percentage)

        # Sort Best Individuals
        self.current_gen_individuals = Individual.sort_individuals(individuals)

        # Selection
        survivors = self.current_gen_individuals[:self.selection_number]

        # Crossover
        children_weights = self.crossover(partial(self.sus, self.current_gen_individuals),
                                self.arithmetic_crossover,
                                len(self.current_gen_individuals) - self.selection_number)
        
        # Mutation
        self.mutation(self.mutate_alter_value, children_weights)

        # Removing Duplicates
        children = []
        for weights in children_weights:
            children.append(self.removing_duplicates(gen_number, weights)) 

        # Performance Log
        print(f"---GEN {gen_number}---")
        print(f"BEST INDIVIDUAL: {self.current_gen_individuals[0]}")

        # Updating Individuals
        survivors.extend(children)
        self.set_diversity(self.current_gen_individuals)

        # Add Individuals to History
        self.history.add_all(self.current_gen_individuals)


    def train_all(self):
        generation_number = self.generation_start_number
        generation_stop = self.generation_limit + self.generation_start_number

        while generation_number < generation_stop:
            self.train_one_generation(generation_number)
            
            generation_number += 1
    
        self.history.save_history()
        self.history.save_best_individual()