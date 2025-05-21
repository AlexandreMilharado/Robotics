import ast
import numpy as np
import pandas as pd
from p1_util.Individual import Individual
from p1_util.History_Individuals import History_Individuals

from p1_util.robot_class import Agent, SimulationEndedError
from functools import partial


np.random.seed(2025)                                                  # Seed for Reproducibility

BRAITENBERG = { "MAX_VALUE_WEIGHT": 1,                                  # Maximum Value for Gene
                "MIN_VALUE_WEIGHT" : -1,                                # Minimum Value for Gene
                "GENES_NUMBER" : 6,                                     # Number of Genes
                "REWARD_BLACK_LINE": 1,                                 # Reward on Black Line
                "REWARD_BLACK_SPEED": 0,                                # Reward on Black Line
                "REWARD_DIFFERENT_PATH": 0,                             # Reward on difference between paths
                "PENALTY_DIFFERENT_POSITION": 0.9,                      # Penalty on same positions taken by Robot
                "PENALTY_WIGGLY_MOV": 0,                                # Penalty on Wiggle Movement
                "PENALTY_DNA": 0.15,                                    # Penalty for DNA Difference
                "PENALTY_COLISION": 0,                                  # Penalty for colling
                "MAX_COLISION": 1,                                      # Max Colision before stop
                "CROSSOVER_ARITHMETIC_MAX": 1.5,                        # Max Value for Alpha
                "CROSSOVER_ARITHMETIC_MIN": -0.5,                       # Min Value for Alpha
                "MUTATION_VARIANCE": 0.5,                               # Variance for Mutation
               }


NETWORKS_SIMPLE = { "MAX_VALUE_WEIGHT": 1,                              # Maximum Value for Gene
                    "MIN_VALUE_WEIGHT" : -1,                            # Minimum Value for Gene
                    "GENES_NUMBER" : 22,                                # Number of Genes
                    "REWARD_BLACK_LINE": 1,                             # Reward on Black Line
                    "REWARD_BLACK_SPEED": 0,                            # Reward on Black Line
                    "REWARD_DIFFERENT_PATH": 0,                         # Reward on difference between paths
                    "PENALTY_DIFFERENT_POSITION": 0.9,                  # Penalty on same positions taken by Robot
                    "PENALTY_WIGGLY_MOV": 0,                            # Penalty on Wiggle Movement
                    "PENALTY_DNA": 0.15,                                # Penalty for DNA Difference
                    "PENALTY_COLISION": 0,                              # Penalty for colling
                    "MAX_COLISION": 1,                                  # Max Colision before stop
                    "CROSSOVER_ARITHMETIC_MAX": 1.5,                    # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": -0.5,                   # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,                           # Variance for Mutation
                }

NETWORKS_COMPLEX = {"MAX_VALUE_WEIGHT": 1,                              # Maximum Value for Gene
                    "MIN_VALUE_WEIGHT" : -1,                            # Minimum Value for Gene
                    "GENES_NUMBER" : 112,                               # Number of Genes
                    "REWARD_BLACK_LINE": 1,                             # Reward on Black Line
                    "REWARD_BLACK_SPEED": 0,                            # Reward on Black Line
                    "REWARD_DIFFERENT_PATH": 0,                         # Reward on difference between paths
                    "PENALTY_DIFFERENT_POSITION": 0.9,                  # Penalty on same positions taken by Robot
                    "PENALTY_WIGGLY_MOV": 0,                            # Penalty on Wiggle Movement
                    "PENALTY_DNA": 0.15,                                # Penalty for DNA Difference
                    "PENALTY_STOP_FRONT": 0.05,                         # Penalty for not moving when enconter object
                    "PENALTY_COLISION": 0.4,                            # Penalty for colling
                    "MAX_COLISION": 10,                                 # Max Colision before stop
                    "CROSSOVER_ARITHMETIC_MAX": 1,                      # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": 1,                      # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,  
                }

class Evolution_Manager():

# Init and Loading
    def __init__(self, INDIVIDUAL_TYPE, TIMESTEP_MULTIPLIER, INDIVIDUALS_HISTORY_PATH, BEST_INDIVIDUAL_PATH):
        self.agent : Agent = Agent(INDIVIDUAL_TYPE, TIMESTEP_MULTIPLIER)
        self.history = History_Individuals(INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH,
                                        BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH,
                                        INDIVIDUAL_TYPE = INDIVIDUAL_TYPE)
        match INDIVIDUAL_TYPE:
            case "BRAITENBERG":
                self.CONSTANT = BRAITENBERG
            case "NETWORKS_SIMPLE":
                self.CONSTANT = NETWORKS_SIMPLE
            case _: 
                self.CONSTANT = NETWORKS_COMPLEX

    def load_train_params(self, population_size, generation_limit, evaluation_time, selection_number, mutation_rate, mutation_alter_rate):
        self.generation_start_number = 0
        self.generation_limit = generation_limit

        self.evaluation_time = evaluation_time

        self.selection_number = selection_number

        self.mutation_rate = mutation_rate
        self.mutation_alter_rate = mutation_alter_rate

        self.current_gen_individuals = [self._generate_individual() for _ in range(population_size)]

    def load_training(self):
        try:
            df = pd.read_csv(self.history.INDIVIDUALS_HISTORY_PATH)
            rows = df.iloc[-len(self.current_gen_individuals):]

            individuals_rows = rows.apply(lambda row: self.history.create_individual_class( int(row["Gen Number"]),
                                                                                            int(row["ID"]),
                                                                                            ast.literal_eval(row["Weights"])), axis=1)
            self.current_gen_individuals = individuals_rows.tolist()

            self.generation_start_number = df["Gen Number"].max() + 1

            for individual in self.current_gen_individuals:
                individual.gen_number = self.generation_start_number

            self.history.load_history()
        except FileNotFoundError:
            print("No file found --- Creating a new one")

    def load_best_individual_on_last_gen(self):
        return self.history.load_best_individual()

    def reset(self):
        self.agent.reset()

        for _ in range(3):
            if self.agent.supervisor.step(self.agent.timestep) == -1:
                raise SimulationEndedError("Step failed after reset")
            
    def run_individual(self, individual):
        while self.agent.supervisor.step(self.agent.timestep) != -1 and not self.agent.collided():
            # Run 1 Step
            self.agent.run_individual(individual)

# Evolucionary Functions
    def _generate_individual(self):
        def generate_weights(size):
            return [np.random.uniform(self.CONSTANT["MIN_VALUE_WEIGHT"], self.CONSTANT["MAX_VALUE_WEIGHT"]) for _ in range(size)]
        
        individual = self.history.create_individual(self.generation_start_number, generate_weights(self.CONSTANT["GENES_NUMBER"]))
        while individual is None:
            individual = self.history.create_individual(self.generation_start_number, generate_weights(self.CONSTANT["GENES_NUMBER"]))

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
    
# Fitness Functions
    def start_fitness(self):
        return abs(self.agent.get_max_velocity())

    def reward_fitness_on_black_line(self):
        sensors_readings =  self.agent._get_ground_sensors_values()
        return self.CONSTANT["REWARD_BLACK_SPEED"] * abs(self.agent.get_average_velocity()) * sum([int(not reading) for reading in sensors_readings])
    
    def reward_fitness_on_straight_line(self):
        return self.CONSTANT["REWARD_STRAIGHT_LINE"] * (abs(self.agent.left_motor.getVelocity() - self.agent.right_motor.getVelocity()) < 0.2)

    def penalty_fitness_based_on_collision(self, fitness, limit_timestep, timesteps):
        return fitness / (limit_timestep - timesteps + 1) 
    
    def penalty_wiggly_movement(self, angular_velocity):
        return -self.CONSTANT["PENALTY_WIGGLY_MOV"] * abs(self.agent.get_max_velocity()) * (np.sign(angular_velocity) != np.sign(self.agent.get_angular_velocity()) and
                                                    self.agent.left_motor != self.agent.right_motor)
    def penalty_front_object(self):
        values = [value / 4301 for value in self.agent.get_frontal_sensors_values()]
        result = sum(values) / len(values)
        return -self.CONSTANT["PENALTY_STOP_FRONT"] * abs(self.agent.get_average_velocity()) * result

    def penalty_obstacle_colision(self, fitness, n):
        return fitness - (n/self.CONSTANT["MAX_COLISION"]) * self.CONSTANT["PENALTY_COLISION"]
                
    def penalty_fitness_based_on_best_dna(self, sorted_individuals, penalties):
        for i in range(len(sorted_individuals) - 1):
            for j in range(i+1, len(sorted_individuals)):
                index, fitness = penalties[j]
                fitness -= self.CONSTANT["PENALTY_DNA"] * fitness * (1 - sorted_individuals[i].dna_diff(sorted_individuals[j]))
                penalties[j] = (index, fitness)

        return penalties
    
    def reward_fitness_on_different_pathing_from(self, sorted_individuals, rewards):
        distances = []
        for individual in sorted_individuals:
            distances.append(individual.distance_from_all(sorted_individuals))

        max_value = max(distances)
        distances = [distance / max_value for distance in distances]

        for i in range(len(sorted_individuals)):
            index, fitness = rewards[i]
            fitness += self.CONSTANT["REWARD_DIFFERENT_PATH"] * distances[i]
            rewards[i] = (index, fitness)

        return rewards
    
    def penalty_fitness_on_same_position(self, sorted_individuals, penalties):
        for i, individual in enumerate(sorted_individuals):
            index, fitness = penalties[i]
            fitness -= self.CONSTANT["PENALTY_DIFFERENT_POSITION"] * fitness * individual.position_diff()
            penalties[i] = (index, fitness)

        return penalties

    def reward_fitness_black_line(self, sorted_individuals, rewards, line_percentage):
        for i, individual in enumerate(sorted_individuals):
            index, fitness = rewards[i]
            fitness += self.CONSTANT["REWARD_BLACK_LINE"] * line_percentage[index][1]
            rewards[i] = (index, fitness)

        return rewards
 

# Stats World
    def set_diversity(self, individuals):
        gene_matrix = np.array([individual.weights for individual in individuals])
        diversity = np.mean(np.std(gene_matrix, axis=0))
        for individual in individuals:
            individual.set_diversity(diversity)

    def set_line_percentage(self, individuals, line_percentage):
        for i, individual in enumerate(individuals):
            individual.set_line_percentage(line_percentage[i][1])

# Training Details
    def train_individual(self, individual):
        def update_fitness(fitness, angular_velocity):
            return (fitness +
                    self.reward_fitness_on_black_line() +
                    self.penalty_front_object() +
                    self.penalty_wiggly_movement(angular_velocity))
        
        def normalise_fitness(fitness):
            return (fitness) / MAX_FITNESS_VALUE

        def update_stats(stats):
            stats["line_touches"] += int(self.agent.is_on_black_line_map())

        angular_velocity = self.agent.get_angular_velocity()
        limit_timestep = int((self.evaluation_time * 1000) / self.agent.timestep + 0.5)
        MAX_FITNESS_VALUE = 2 * 9.53 * limit_timestep
        # MAX_FITNESS_VALUE =  limit_timestep
        fitness = self.start_fitness()
        stats = {"line_touches": 0}
        timesteps = 0 
        n = 0
        while self.agent.supervisor.step(self.agent.timestep) != -1 and timesteps < limit_timestep and n < 10:
            # Run 1 Step
            self.agent.run_individual(individual)

            # Fitness Calculation
            fitness = update_fitness(fitness, angular_velocity)

            # Updating States
            angular_velocity = self.agent.get_angular_velocity()
            individual.add_path(self.agent.supervisor.getSelf().getPosition()[:2])
            update_stats(stats)
            timesteps += 1

            if self.agent.collided():
                n += 1
                self.reset()
        
        if timesteps < limit_timestep and not self.agent.collided():
            raise SimulationEndedError()
        
        # Fitness Calculation
        fitness = self.penalty_fitness_based_on_collision(fitness, limit_timestep, timesteps)
        fitness = normalise_fitness(fitness)
        fitness = self.penalty_obstacle_colision(fitness, n)

        return {"FITNESS": fitness, "LINE_PERCENTAGE": stats["line_touches"]/limit_timestep}

    def train_one_individual(self, individual):
        failed = True
        result = {"FITNESS": 0, "LINE_PERCENTAGE": 0}
        tries = 0
        while failed and tries < 3:
            try:
                self.reset()
                result_try = self.train_individual(individual)
                result["FITNESS"] += result_try["FITNESS"]
                result["LINE_PERCENTAGE"] += result_try["LINE_PERCENTAGE"]
                failed = False
                tries +=1
            except SimulationEndedError:
                print("Restarting training")

        result["FITNESS"] /= tries
        result["LINE_PERCENTAGE"] /= tries
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
        print("sort_individuals_by_fitness_list")
        fitness = self.penalty_fitness_based_on_best_dna(individuals, fitness)
        print("penalty_fitness_based_on_best_dna")
        fitness = self.reward_fitness_on_different_pathing_from(individuals, fitness)
        print("reward_fitness_on_different_pathing_from")
        fitness = self.penalty_fitness_on_same_position(individuals, fitness)
        print("penalty_fitness_on_same_position")
        fitness = self.reward_fitness_black_line(individuals, fitness, line_percentage)
        print("Fitness Ended")

        # Setting Individuals Fitness
        for i in range(len(individuals)):
            individuals[i].fitness = fitness[i][1]
            
        
        # Setting Generation Stats and adding to history
        self.set_line_percentage(individuals, line_percentage)
        self.set_diversity(individuals)
        self.history.add_all(individuals)
        print("SETTING Ended")

        # Sort Best Individuals
        self.current_gen_individuals = Individual.sort_individuals(individuals)
        print("Sort Ended")

        # Selection
        survivors = self.current_gen_individuals[:self.selection_number]
        print("Selection Ended")

        # Crossover
        children_weights = self.crossover(partial(self.sus, self.current_gen_individuals),
                                self.arithmetic_crossover,
                                len(self.current_gen_individuals) - self.selection_number)
        print("Crossover Ended")
        

        # Mutation
        self.mutation(self.mutate_alter_value, children_weights)
        print("Mutation Ended")

        # Performance Log
        print(f"---GEN {gen_number}---")
        print(f"BEST INDIVIDUAL: {self.current_gen_individuals[0]}")

        # Updating Generation Number
        for survivor in survivors:
            survivor.gen_number = gen_number + 1
        
        # Removing Duplicates
        children = []
        for weights in children_weights:
            children.append(self.removing_duplicates(gen_number + 1, weights))
        

        # Updating Individuals
        survivors.extend(children)
        self.current_gen_individuals = survivors
        

    def train_all(self):
        generation_number = self.generation_start_number
        generation_stop = self.generation_limit + self.generation_start_number

        while generation_number < generation_stop:
            self.train_one_generation(generation_number)
            
            generation_number += 1
    
            self.history.save_history()
            self.history.save_best_individual()