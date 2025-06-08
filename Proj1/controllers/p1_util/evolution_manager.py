import ast
import numpy as np
import pandas as pd
from p1_util.Individual import Individual
from p1_util.History_Individuals import History_Individuals

from p1_util.robot_class import Agent, SimulationEndedError
from functools import partial


np.random.seed(2025)                                                    # Seed for Reproducibility

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
                    "GENES_NUMBER" : 38,                                # Number of Genes
                    "REWARD_BLACK_LINE": 0,                             # Reward on Black Line
                    "REWARD_BLACK_SPEED": 0,                            # Reward on Black Line
                    "REWARD_DIFFERENT_PATH": 0,                         # Reward on difference between paths
                    "REWARD_ALIVE": 1,                                  # Reward for being alive
                    "PENALTY_DIFFERENT_POSITION": 0.9,                  # Penalty on same positions taken by Robot
                    "PENALTY_WIGGLY_MOV": 0,                            # Penalty on Wiggle Movement
                    "PENALTY_DNA": 0.15,                                # Penalty for DNA Difference
                    "PENALTY_STOP_FRONT": 0.15,                         # Penalty for not moving when enconter object
                    "PENALTY_COLISION": 0.4,                            # Penalty for colling
                    "MAX_COLISION": 5,                                  # Max Colision before stop
                    "CROSSOVER_ARITHMETIC_MAX": 1.5,                    # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": -0.5,                   # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,                           # Variance for Mutation
                }

NETWORKS_COMPLEX = {"MAX_VALUE_WEIGHT": 1,                              # Maximum Value for Gene
                    "MIN_VALUE_WEIGHT" : -1,                            # Minimum Value for Gene
                    "GENES_NUMBER" : 42,                                # Number of Genes
                    "REWARD_BLACK_LINE": 1,                             # Reward on Black Line
                    "REWARD_BLACK_SPEED": 0,                            # Reward on Black Line
                    "REWARD_DIFFERENT_PATH": 0,                         # Reward on difference between paths
                    "REWARD_ALIVE": 0,                                  # Reward for being alive
                    "PENALTY_DIFFERENT_POSITION": 0.9,                  # Penalty on same positions taken by Robot
                    "PENALTY_WIGGLY_MOV": 0,                            # Penalty on Wiggle Movement
                    "PENALTY_DNA": 0,                                   # Penalty for DNA Difference
                    "PENALTY_STOP_FRONT": 0,                            # Penalty for not moving when enconter object
                    "PENALTY_FUTURE_COLISION": 0.5,                     # Penalty if current policy will colide
                    "PENALTY_COLISION": 0,                              # Penalty for colling
                    "MAX_COLISION": 1,                                  # Max Colision before stop
                    "CROSSOVER_ARITHMETIC_MAX": 1.5,                    # Max Value for Alpha
                    "CROSSOVER_ARITHMETIC_MIN": -0.5,                   # Min Value for Alpha
                    "MUTATION_VARIANCE": 0.5,  
                }

class Evolution_Manager():

# Init and Loading
    def __init__(self, INDIVIDUAL_TYPE, TIMESTEP_MULTIPLIER, INDIVIDUALS_HISTORY_PATH, BEST_INDIVIDUAL_PATH):
        """
        Initialize the Evolution Manager
        
        Parameters
        ----------
        INDIVIDUAL_TYPE : str
            Type of Individual to use, can be "BRAITENBERG", "NETWORKS_SIMPLE", or "NETWORKS_COMPLEX"
        TIMESTEP_MULTIPLIER : int
            Multiplier for the Webots timestep
        INDIVIDUALS_HISTORY_PATH : str
            Path to save the history of the individuals
        BEST_INDIVIDUAL_PATH : str
            Path to save the best individual
        """       
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
        """
        Load the parameters for the evolutionary algorithm
        
        Parameters
        ----------
        population_size : int
            Number of individuals in the population
        generation_limit : int
            Number of generations to run
        evaluation_time : int
            Time to evaluate each individual
        selection_number : int
            Number of individuals to select for the next generation
        mutation_rate : float
            Rate of mutation
        mutation_alter_rate : float
            Rate of mutation of the mutation rate
        
        """

        self.generation_start_number = 0
        self.generation_limit = generation_limit

        self.evaluation_time = evaluation_time

        self.selection_number = selection_number

        self.mutation_rate = mutation_rate
        self.mutation_alter_rate = mutation_alter_rate

        self.current_gen_individuals = [self._generate_individual() for _ in range(population_size)]

    def load_training(self):
        """
        Load the training data from the history file
        
        This function loads the last generation from the history file and sets the current generation number to the last generation number + 1.
        If the history file does not exist, it creates a new one and starts from generation 0.
        """
        
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
        """
        Load the best individual from the last saved generation.

        This function retrieves the best individual from the history
        of saved individuals, using the path specified in the 
        BEST_INDIVIDUAL_PATH attribute of the history object.

        Returns
        -------
        Individual
            The best individual from the last saved generation.
        """

        return self.history.load_best_individual()

    def reset(self):
        """
        Resets the robot and the simulation.
        
        This function resets the robot's state to the default state and runs the simulation for 3 steps to ensure that the
        robot is properly reset.
        """
        
        self.agent.reset()

        for _ in range(3):
            if self.agent.supervisor.step(self.agent.timestep) == -1:
                raise SimulationEndedError("Step failed after reset")
            
    def run_individual(self, individual):
        """
        Runs an individual in the simulation until the simulation ends or the individual collides.
        
        Parameters
        ----------
        individual : Individual
            The individual to run in the simulation.
        """
        
        while self.agent.supervisor.step(self.agent.timestep) != -1 and not self.agent.collided():
            # Run 1 Step
            self.agent.run_individual(individual)

# Evolucionary Functions
    def _generate_individual(self):
        """
        Generates a new individual for the population.
        
        This function generates a new individual by creating a new set of weights, and then
        creates a new individual using the history object. If the individual already exists,
        the function generates a new set of weights and tries again until a new individual
        is created.
        
        Returns
        -------
        Individual
            A new individual with the generated weights.
        """
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
        """
        Performs arithmetic crossover on two parent individuals to generate a new set of weights.

        This method blends the weights of two parent individuals using a randomly generated
        alpha value. The resulting weights are bounded by the defined minimum and maximum values.

        Parameters
        ----------
        parent1 : Individual
            The first parent individual contributing to the crossover.
        parent2 : Individual
            The second parent individual contributing to the crossover.

        Returns
        -------
        list
            A list of weights representing the offspring generated through arithmetic crossover.
        """

        alpha = np.random.uniform(self.CONSTANT["CROSSOVER_ARITHMETIC_MIN"], self.CONSTANT["CROSSOVER_ARITHMETIC_MAX"])
        beta = 1 - alpha
        weights = []
        for i in range(self.CONSTANT["GENES_NUMBER"]):
            weights.append(max(min(parent1.weights[i] * alpha + parent2.weights[i] * beta,
                                   self.CONSTANT["MAX_VALUE_WEIGHT"]),
                                self.CONSTANT["MIN_VALUE_WEIGHT"]))

        return weights

    def crossover(self, selection_parents, crossover_aux, children_number):
        """
        Generates offspring by performing crossover on selected parents.

        This method uses a selection function to choose parents and a crossover 
        auxiliary function to blend their genetic information, producing a specified 
        number of children with new sets of weights.

        Parameters
        ----------
        selection_parents : callable
            A function that selects and returns a list of parent individuals for crossover.
        crossover_aux : callable
            A function that performs the crossover operation on two parents to produce a child.
        children_number : int
            The number of children to generate through crossover.

        Returns
        -------
        list
            A list of weights representing the offspring generated through crossover.
        """

        children_weights = []

        parents = selection_parents()

        for i in range(children_number):
            child_weights = crossover_aux(parents[i], parents[i+1])
            children_weights.append(child_weights)

        return children_weights

    def mutate_alter_value(self, child_weights):
        """
        Randomly alters a value in a set of weights to another value following a normal distribution.

        This method iterates through a list of weights and randomly selects a value to be altered.
        The new value is generated by sampling from a normal distribution with mean equal to the
        current value and standard deviation equal to a constant defined in the configuration.
        If the new value is outside the valid range of values, it is re-sampled until a valid value
        is obtained.

        Parameters
        ----------
        child_weights : list
            The list of weights to be mutated.

        Returns
        -------
        list
            The list of weights after mutation.
        """
        for i, w in enumerate(child_weights):
            if np.random.rand() < self.mutation_alter_rate:
                random_value = np.random.normal(w, self.CONSTANT["MUTATION_VARIANCE"])

                while random_value < self.CONSTANT["MIN_VALUE_WEIGHT"] or random_value > self.CONSTANT["MAX_VALUE_WEIGHT"]:
                    random_value = np.random.normal(w, self.CONSTANT["MUTATION_VARIANCE"])
            
                child_weights[i] = random_value

        return child_weights
         

    def mutation(self, mutation_aux, children_weights):
        """
        Applies mutation to a list of weights using a mutation function.

        This method iterates through a list of weights and randomly selects a weight to be mutated.
        The mutation function is called with the selected weight and the result is assigned back to the
        list of weights.

        Parameters
        ----------
        mutation_aux : callable
            A function that takes a weight and returns a mutated weight.
        children_weights : list
            A list of weights to be mutated.

        Returns
        -------
        list
            The list of weights after mutation.
        """
        for i in range(len(children_weights)):
            if np.random.rand() < self.mutation_rate:
                children_weights[i] = mutation_aux(children_weights[i])

    def removing_duplicates(self, gen_number, individual_weights):
        """
        Removes duplicates of an individual in the history.

        This method takes a generation number and a set of weights and creates a new individual in the history.
        If the individual already exists in the history, it generates a new set of weights and tries again until a new individual
        is created.

        Parameters
        ----------
        gen_number : int
            The generation number of the individual.
        individual_weights : list
            A list of weights representing the individual.

        Returns
        -------
        Individual
            The newly created individual.
        """

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
        """
        Returns the initial fitness value of an individual, which is the absolute value of the maximum velocity of the robot.
        
        Returns
        -------
        float
            The initial fitness value of an individual.
        """
        return abs(self.agent.get_max_velocity())

    def reward_fitness_on_black_line(self):
        """
        Returns the reward for the fitness of an individual based on its speed when on black line.
        
        The reward is calculated as the absolute value of the average velocity of the robot
        multiplied by a constant and the sum of the two ground sensors, which are both
        inverted so that the reward is higher when the robot is on the black line.
        
        Returns
        -------
        float
            The reward for the fitness of an individual.
        """
        
        sensors_readings =  self.agent._get_ground_sensors_values()
        return self.CONSTANT["REWARD_BLACK_SPEED"] * abs(self.agent.get_average_velocity()) * sum([int(not reading) for reading in sensors_readings])
    
    def reward_fitness_on_straight_line(self):
        """
        Returns the reward for maintaining straight line movement.

        This method calculates a reward based on the difference in velocity
        between the left and right motors. If the difference is below a certain
        threshold, it is considered to be moving in a straight line, and a reward 
        is given.

        Returns
        -------
        float
            The reward for maintaining straight line movement.
        """

        return self.CONSTANT["REWARD_STRAIGHT_LINE"] * (abs(self.agent.left_motor.getVelocity() - self.agent.right_motor.getVelocity()) < 0.2)
    
    def penalty_wiggly_movement(self, angular_velocity):
        """
        Returns the penalty for wiggly movement.

        This method calculates a penalty based on the difference in angular velocity
        between the left and right motors. If the difference is not zero, it is considered
        to be wiggly movement, and a penalty is given.

        Parameters
        ----------
        angular_velocity : float
            The angular velocity of the robot.

        Returns
        -------
        float
            The penalty for wiggly movement.
        """
        return -self.CONSTANT["PENALTY_WIGGLY_MOV"] * abs(self.agent.get_max_velocity()) * (np.sign(angular_velocity) != np.sign(self.agent.get_angular_velocity()) and
                                                    self.agent.left_motor != self.agent.right_motor)
    def penalty_front_object(self):
        """
        Returns the penalty for objects in front of the robot.

        This method calculates a penalty based on the average of the frontal sensors' values.
        If the average is close to 1, it means there is an object in front of the robot, and a penalty
        is given.

        Returns
        -------
        float
            The penalty for objects in front of the robot.
        """
        values = [value / 4301 for value in self.agent.get_frontal_sensors_values()]
        result = sum(values) / len(values)
        return -self.CONSTANT["PENALTY_STOP_FRONT"] * abs(self.agent.get_average_velocity()) * result
    
    def penalty_fitness_collide(self):
        """
        Returns the penalty for the robot colliding with an obstacle in its path.

        This method calculates a penalty based on whether the robot will collide with an obstacle
        in the next position. If the robot will collide, a penalty is given.

        Returns
        -------
        float
            The penalty for the robot colliding with an obstacle in its path.
        """
        past_collide = self.agent.will_next_position_collide()
        self.agent.update_sensors_past()
        new_collide = self.agent.will_next_position_collide()
        return -self.CONSTANT["PENALTY_FUTURE_COLISION"] * (past_collide and new_collide) * abs(self.agent.get_average_velocity())

    def penalty_obstacle_colision(self, fitness, n):
        """
        Returns the penalty for the robot colliding with an obstacle in its path.

        This method calculates a penalty based on the number of times the robot has collided
        with an obstacle. If the robot has collided more than the maximum number of
        collisions, the penalty is the maximum penalty. Otherwise, the penalty is a percentage
        of the maximum penalty based on the number of collisions.

        Parameters
        ----------
        fitness : float
            The fitness of the individual.
        n : int
            The number of times the robot has collided with an obstacle.

        Returns
        -------
        float
            The penalty for the robot colliding with an obstacle in its path.
        """
        return fitness - (n/self.CONSTANT["MAX_COLISION"]) * self.CONSTANT["PENALTY_COLISION"]
                
    def penalty_fitness_based_on_best_dna(self, sorted_individuals, penalties):
        """
        Returns the penalties for the given sorted individuals based on the difference between the current DNA and the best DNA.

        This method calculates a penalty based on the difference between the current DNA and the best DNA.
        The penalty is a percentage of the current fitness, given by the constant "PENALTY_DNA".

        Parameters
        ----------
        sorted_individuals : list
            A list of sorted individuals.
        penalties : list
            A list of tuples, where the first element is the index of the individual and the second element is the penalty.

        Returns
        -------
        list
            A list of tuples, where the first element is the index of the individual and the second element is the penalty.
        """
        for i in range(len(sorted_individuals) - 1):
            for j in range(i+1, len(sorted_individuals)):
                index, fitness = penalties[j]
                fitness -= self.CONSTANT["PENALTY_DNA"] * fitness * (1 - sorted_individuals[i].dna_diff(sorted_individuals[j]))
                penalties[j] = (index, fitness)

        return penalties
    
    def reward_fitness_on_different_pathing_from(self, sorted_individuals, rewards):
        """
        Returns the rewards for the given sorted individuals based on the distance from all other individuals.

        This method calculates a reward based on the distance from all other individuals. The reward is a percentage of the current fitness, given by the constant "REWARD_DIFFERENT_PATH".

        Parameters
        ----------
        sorted_individuals : list
            A list of sorted individuals.
        rewards : list
            A list of tuples, where the first element is the index of the individual and the second element is the reward.

        Returns
        -------
        list
            A list of tuples, where the first element is the index of the individual and the second element is the reward.
        """
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
        """
        Returns the penalties for the given sorted individuals based on the position difference from all other individuals.

        This method calculates a penalty based on the position difference from all other individuals. The penalty is a percentage of the current fitness, given by the constant "PENALTY_DIFFERENT_POSITION".

        Parameters
        ----------
        sorted_individuals : list
            A list of sorted individuals.
        penalties : list
            A list of tuples, where the first element is the index of the individual and the second element is the penalty.

        Returns
        -------
        list
            A list of tuples, where the first element is the index of the individual and the second element is the penalty.
        """
        for i, individual in enumerate(sorted_individuals):
            index, fitness = penalties[i]
            fitness -= self.CONSTANT["PENALTY_DIFFERENT_POSITION"] * fitness * individual.position_diff()
            penalties[i] = (index, fitness)

        return penalties

    def reward_fitness_black_line(self, sorted_individuals, rewards, line_percentage):
        """
        Returns the rewards for the given sorted individuals based on the black line percentage.

        This method adds a reward to the fitness of each individual based on the percentage of the black line that the individual has driven on. The reward is a percentage of the current fitness, given by the constant "REWARD_BLACK_LINE".

        Parameters
        ----------
        sorted_individuals : list
            A list of sorted individuals.
        rewards : list
            A list of tuples, where the first element is the index of the individual and the second element is the reward.
        line_percentage : list
            A list of tuples, where the first element is the index of the individual and the second element is the percentage of the black line that the individual has driven on.

        Returns
        -------
        list
            A list of tuples, where the first element is the index of the individual and the second element is the reward.
        """
        for i, individual in enumerate(sorted_individuals):
            index, fitness = rewards[i]
            fitness += self.CONSTANT["REWARD_BLACK_LINE"] * line_percentage[index][1]
            rewards[i] = (index, fitness)

        return rewards
    
    def reward_fitness_alive(self, fitness):
        """
        Returns the reward for the fitness of an individual based on its survival time.
        
        The reward is the fitness of the individual multiplied by the constant "REWARD_ALIVE".
        
        Parameters
        ----------
        fitness : float
            The fitness of the individual.
        
        Returns
        -------
        float
            The reward for the fitness of an individual.
        """
        
        return fitness * self.CONSTANT["REWARD_ALIVE"]


# Stats World
    def set_diversity(self, individuals):
        """
        Sets the diversity of the individuals in the population.

        This method calculates the diversity of the population by calculating the standard deviation of the weights of all individuals
        and then takes the mean of the standard deviations. The diversity is then set for each individual in the population.

        Parameters
        ----------
        individuals : list
            A list of individuals in the population.
        """
        gene_matrix = np.array([individual.weights for individual in individuals])
        diversity = np.mean(np.std(gene_matrix, axis=0))
        for individual in individuals:
            individual.set_diversity(diversity)

    def set_line_percentage(self, individuals, line_percentage):
        """
        Sets the percentage of the black line that each individual in the population has driven on.
        
        Parameters
        ----------
        individuals : list
            A list of individuals in the population.
        line_percentage : list
            A list of tuples, where the first element is the index of the individual and the second element is the percentage of the black line that the individual has driven on.
        """
        
        for i, individual in enumerate(individuals):
            individual.set_line_percentage(line_percentage[i][1])

# Training Details
    def train_individual(self, individual):
        """
        Trains an individual for a specified evaluation time and calculates fitness.

        This function iteratively simulates the behavior of an individual in the environment, updating its fitness
        based on various reward and penalty functions. The individual's path and performance on the black line
        are tracked, and the fitness is normalized at the end of the evaluation.

        Parameters
        ----------
        individual : Individual
            The individual to be trained.

        Returns
        -------
        dict
            A dictionary with keys "FITNESS" and "LINE_PERCENTAGE" representing the fitness score of the individual
            and the percentage of the black line the individual has driven on, respectively.
        """

        def update_fitness(fitness, angular_velocity):
            """
            Updates the fitness of an individual based on various reward and penalty factors.

            This method calculates the new fitness score by adding the existing fitness to the sum of
            rewards and penalties from different criteria, such as the individual's performance on the 
            black line, proximity to frontal objects, potential collisions, and wiggly movement.

            Parameters
            ----------
            fitness : float
                The current fitness score of the individual.
            angular_velocity : float
                The angular velocity of the robot used to assess wiggly movement penalties.

            Returns
            -------
            float
                The updated fitness score.
            """

            return (fitness +
                    self.reward_fitness_on_black_line() +
                    self.penalty_front_object() +
                    self.penalty_fitness_collide() +
                    self.penalty_wiggly_movement(angular_velocity))
        
        def normalise_fitness(fitness):
            """
            Normalizes the fitness value to a value between 0 and 1.

            Parameters
            ----------
            fitness : float
                The fitness value to be normalized.

            Returns
            -------
            float
                The normalized fitness value.
            """
            return (fitness) / MAX_FITNESS_VALUE

        def update_stats(stats):
            """
            Updates the statistics dictionary with the count of times the agent has touched the black line.

            This method increments the "line_touches" value in the provided stats dictionary by checking
            if the agent is currently on a black line in the map.

            Parameters
            ----------
            stats : dict
                A dictionary containing various statistics about the agent's performance. The key "line_touches"
                should be present in this dictionary, and its value will be incremented.

            """

            stats["line_touches"] += int(self.agent.is_on_black_line_map())

        angular_velocity = self.agent.get_angular_velocity()
        limit_timestep = int((self.evaluation_time * 1000) / self.agent.timestep + 0.5)
        MAX_FITNESS_VALUE = ((self.CONSTANT["REWARD_BLACK_LINE"] != 0 or self.CONSTANT["REWARD_BLACK_SPEED"] != 0) * 2 * 9.53 * limit_timestep +
                             (self.CONSTANT["REWARD_ALIVE"] != 0) * limit_timestep)
        fitness = self.start_fitness()
        stats = {"line_touches": 0}
        timesteps = 0 
        n = 0
        while self.agent.supervisor.step(self.agent.timestep) != -1 and timesteps < limit_timestep and n < self.CONSTANT["MAX_COLISION"]:
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
        
        if timesteps < limit_timestep and not self.agent.collided() and n != self.CONSTANT["MAX_COLISION"]:
            raise SimulationEndedError()
        
        # Fitness Calculation
        fitness = normalise_fitness(fitness)
        fitness = self.penalty_obstacle_colision(fitness, n)
        fitness = self.reward_fitness_alive(fitness)

        return {"FITNESS": fitness, "LINE_PERCENTAGE": stats["line_touches"]/limit_timestep}

    def train_one_individual(self, individual):
        """
        Train one individual and return its fitness and line percentage.

        Parameters
        ----------
        individual : Individual
            The individual to be trained.

        Returns
        -------
        result : dict
            A dictionary containing the fitness and line percentage of the individual.

        Notes
        -----
        If the simulation ends before the evaluation time is finished, the method
        will restart the simulation until the evaluation time is finished or
        the maximum number of tries is reached.
        """
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
        """
        Trains one generation of individuals in the evolutionary algorithm.

        This method performs the training process for a single generation of individuals. It includes resetting
        paths, training each individual, calculating and updating fitness, setting generation statistics,
        performing selection, crossover, and mutation to evolve the population to the next generation.

        Parameters
        ----------
        gen_number : int
            The current generation number.

        Notes
        -----
        The function logs various stages of the training process for debugging and monitoring purposes.
        """

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
        """
        Trains all generations in the evolutionary algorithm.

        This method performs the training process for all generations. It includes resetting
        paths, training each individual, calculating and updating fitness, setting generation statistics,
        performing selection, crossover, and mutation to evolve the population to the next generation.

        Notes
        -----
        The function logs various stages of the training process for debugging and monitoring purposes.
        """
        generation_number = self.generation_start_number
        generation_stop = self.generation_limit + self.generation_start_number

        while generation_number < generation_stop:
            self.train_one_generation(generation_number)
            
            generation_number += 1
            self.history.save_history()
            # self.history.save_best_individual()