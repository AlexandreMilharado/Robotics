import ast
from math import sqrt
import os
import pickle
import numpy as np
import pandas as pd
from controller import Supervisor
from p1_util.robot_class import Agent
from functools import partial


np.random.seed(105946)
MIN_VALUE_WEIGHT = -1
MAX_VALUE_WEIGHT = 1

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

    def __init__(self, id, weights, fitness = 0, path = []):
        self.id = id
        self.weights = weights
        self.fitness = fitness
        self.path = path

    def add_path(self, position):
        self.path.append(position)

    def reset_path(self):
        self.path = []
        
    def distance_from_all(self, population):
        def euclidean(p1, p2):
            return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        def path_distance(path1, path2):
            return sum(euclidean(p1, p2) for p1, p2 in zip(path1, path2)) / len(path1)
        
        return sum(path_distance(self.path, other.path) for other in population if other != self) / (len(population) - 1)

    @classmethod
    def set_id_counter(cls, number):
        cls.id_counter = number

    @classmethod
    def create_individual(cls, weights):
        cls.id_counter += 1
        return Individual(cls.id_counter, weights)

    def __eq__(self, other):
        return self.id == other.id
    
    def __str__(self):
        return f"{{ID: {self.id}, Weights: {self.weights}, Fitness: {self.fitness}}}"
        


class Evolution_Manager:
    def __init__(self, timestep_multiplier):
        self.supervisor : Supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep() * timestep_multiplier)
        self.agent : Agent = Agent(self.supervisor, self.timestep)
        self.history = []

    def load_train_params(self, generation_converge_stop, population_size, selection_number, parents_number, mutation_rate, mutation_alter_rate, blind_markov_assumption, evaluation_time):
        self.individuals = [Individual.create_individual(self.generate_weights(6)) for _ in range(population_size)]
        self.generation_converge_stop = generation_converge_stop
        self.evaluation_time = evaluation_time
        self.selection_number = selection_number
        self.parents_number = parents_number
        self.mutation_rate = mutation_rate
        self.mutation_alter_rate = mutation_alter_rate
        self.blind_markov_assumption = blind_markov_assumption
        self.generation_start_number = 0


    def reset_state(self):
        self.agent.reset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)
        self.supervisor.step(self.timestep)

    def update_generation(self, gen_number, individuals):
        for individual in individuals:
            self.history.append([gen_number, individual.id, individual.fitness, individual.weights])


    def save_training(self):
        file_path = "../p1_util/evolutionary.csv"
        header = not os.path.exists(file_path)
        pd.DataFrame(data=self.history, columns=["Gen Number", "ID", "Fitness", "Weights"]).to_csv(file_path, mode='a', header=header, index=False)

    def save_best_individual(self):
        with open('../p1_util/best_individual.pkl', 'wb') as f:
            pickle.dump(self.sort_individuals()[0], f)

    def load_training(self):
        df = pd.read_csv('../p1_util/evolutionary.csv')
        id_counter = df["ID"].max()
        gen_number = df["Gen Number"].max()
        rows = df.iloc[-len(self.individuals):]

        individuals_rows = rows.apply(lambda row: Individual(int(row["ID"]), ast.literal_eval(row["Weights"]), float(row["Fitness"])), axis=1)
        self.individuals = individuals_rows.tolist()

        Individual.set_id_counter(id_counter)
        self.generation_start_number = gen_number
        

    def load_best_individual(cls):
        with open('../p1_util/best_individual.pkl', 'rb') as f:
            return pickle.load(f)
        
    def run_individual(self, individual):
        while not self.agent.collided():
            # Read Sensors
            sensors_inputs = self._read_sensors()

            # Control Motors
            self.run_step(individual.weights, sensors_inputs)


    def generate_weights(self, size):
        return [np.random.uniform(MIN_VALUE_WEIGHT, MAX_VALUE_WEIGHT) for _ in range(size)]

    def run_step(self, weights, sensors_inputs):
        p_1_e, p_2_e, p_3_e, p_1_d, p_2_d, p_3_d = weights
        s_e, s_d = sensors_inputs

        left_speed =  s_e * p_1_e + s_d * p_2_e + p_3_e
        right_speed = s_e * p_1_d + s_d * p_2_d + p_3_d

        self.agent.set_velocity_left_motor(left_speed, sensors_inputs)
        self.agent.set_velocity_right_motor(right_speed, sensors_inputs)
        
        self.supervisor.step(self.timestep)

    def _read_sensors(self):
        (left_sensor_value, right_sensor_value) = self.agent.get_ground_sensors_values()
        return (self.agent.is_not_on_black_line(left_sensor_value),
                self.agent.is_not_on_black_line(right_sensor_value))

    def _run_train_simulation(self, individual):
            def update_fitness_collision(fitness):
                return fitness / (limit_timestep - timesteps + 1) 

            def update_fitness(fitness):
                penalize = np.sign(angular_velocity) != np.sign(self.agent.get_angular_velocity()) and self.agent.left_motor != self.agent.right_motor
                return (fitness
                        + int((not sensors_inputs[0]) + (not sensors_inputs[1])) * self.agent.get_average_velocity()
                        - penalize * self.agent.get_max_velocity())
        
            self.reset_state()
            fitness = 0
            timesteps = 0
            limit_timestep = (self.evaluation_time * 1000) / self.timestep
            angular_velocity = self.agent.get_angular_velocity()

            while (timesteps < limit_timestep and
                    not self.agent.collided()):
                
                # Read Sensors
                sensors_inputs  = self._read_sensors()

                # Calculate Fitness
                fitness = update_fitness(fitness)

                # Control Motors
                self.run_step(individual.weights, sensors_inputs)

                # Update States
                angular_velocity = self.agent.get_angular_velocity()
                individual.add_path(self.supervisor.getSelf().getPosition()[:2])
                timesteps += 1

            individual.fitness = update_fitness_collision(fitness)
    
    def train_one_individual(self, individual):
        failed = True

        while failed:
            self._run_train_simulation(individual)
            failed = len(individual.path) == 0


    def update_fitness_based_on_exploration(self):
        distances = []
        for individual in self.individuals:
            distances.append(individual.distance_from_all(self.individuals))

        max_value = max(distances)
        distances = [distance / max_value for distance in distances]

        for i, individual in enumerate(self.individuals):
            individual.fitness += distances[i]

    def normalise_fitness(self):
        dem = 2 * 9.53 * ((self.evaluation_time * 1000) / self.timestep) + 1
        for individual in self.individuals:
            individual.fitness = individual.fitness / dem

    def train_one_generation(self, gen_number):
        # Reset Paths
        for individual in self.individuals:
            individual.reset_path()

        # Train Individuals
        for individual in self.individuals:
            self.train_one_individual(individual)

        # Update Fitness
        self.update_fitness_based_on_exploration()
        self.normalise_fitness()

        # Sort Best Individuals
        sorted_individuals = self.sort_individuals()

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
        self.update_generation(gen_number, sorted_individuals)
        survivors.extend(children)

        return survivors
        

    def train_all(self):
        generation_number = self.generation_start_number
        generation_stop = self.generation_converge_stop + self.generation_start_number
        while generation_number < generation_stop:
            survivors = self.train_one_generation(generation_number)
            
            # Updating new Individuals
            self.individuals = survivors

            generation_number += 1
        self.save_training()
        self.save_best_individual()


    def sort_individuals(self):
        return sorted(self.individuals, key=lambda individual: individual.fitness, reverse=True)

    def sus(self, sorted_individuals):
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
        alpha = np.random.uniform(-0.5, 1.5)
        beta = 1 - alpha
        weights = []
        for i in range(6):
            weights.append(parent1.weights[i] * alpha + parent2.weights[i] * beta)
        return Individual.create_individual(weights) 

    def crossover(self, selection_parents, crossover_aux, children_number):
        children = []

        parents = selection_parents()

        for i in range(children_number):
            child = crossover_aux(parents[i], parents[i+1])
            children.append(child)

        return children

    def mutate_alter_value(self, child):
        for i, w in enumerate(child.weights):
            if np.random.rand() < self.mutation_alter_rate:
                random_value = np.random.normal(w, 0.5)
                while random_value < MIN_VALUE_WEIGHT or random_value > MAX_VALUE_WEIGHT:
                    random_value = np.random.normal(w, 0.5)
                child.weights[i] = random_value

    def mutation(self, mutation_aux, children):
        for child in children:
            if np.random.rand() < self.mutation_rate:
                mutation_aux(child)