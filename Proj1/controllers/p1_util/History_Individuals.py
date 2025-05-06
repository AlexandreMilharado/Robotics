
import ast
import os
import pickle
import copy
import pandas as pd

from p1_util.Individual import Individual


class History_Individuals:
    def __init__(self, INDIVIDUALS_HISTORY_PATH, BEST_INDIVIDUAL_PATH):
        self.individuals_to_save = []
        self.weights_history = []

        self.id_counter = 0

        self.INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH
        self.BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH


    def create_individual(self, gen_number, weights):
        def _add_weights(weights):
            self.weights_history.append(weights)

        def _exists(weights):
            return weights in self.weights_history
        
        if _exists(weights):
            return None
        
        _add_weights(weights)
        self.id_counter += 1
        return Individual(gen_number, self.id_counter, weights)


    def add(self, individual):
        self.individuals_to_save.append(copy.deepcopy(individual))

    def add_all(self, individuals):
        sorted_individuals = Individual.sort_individuals_by_id(individuals)
        for individual in sorted_individuals:
            self.add(individual)


    def save_history(self):
        def _convert_history_to_save_format():
            return [individual.to_list() for individual in self.individuals_to_save]
        
        def get_index_next_gen(individuals, max_gen):
            for i in range(len(individuals)):
                if individuals[i][0] == max_gen:
                    return i
            return None

        file_path = self.INDIVIDUALS_HISTORY_PATH 
        header = not os.path.exists(file_path)
        data = _convert_history_to_save_format()

        if not header:
            df = pd.read_csv(self.INDIVIDUALS_HISTORY_PATH)
            data = data[get_index_next_gen(data, df["Gen Number"].max() + 1):]
        pd.DataFrame(data=data,
                    columns=["Gen Number", "ID", "Fitness", "Weights", "Diversity", "Black Line Percentage"]
                    ).to_csv(file_path, mode='a', header=header, index=False)
        
    def load_history(self):
        def update_history_weights(row):
            self.weights_history.append(ast.literal_eval(row["Weights"]))
        
        df = pd.read_csv(self.INDIVIDUALS_HISTORY_PATH)

        self.id_counter = df["ID"].max() + 1
        df.apply(update_history_weights, axis=1)


    def save_best_individual(self):
        with open(self.BEST_INDIVIDUAL_PATH, 'wb') as f:
            pickle.dump(Individual.sort_individuals(self.individuals_to_save)[0], f)

    def load_best_individual(self):
        with open(self.BEST_INDIVIDUAL_PATH, 'rb') as f:
            return pickle.load(f)