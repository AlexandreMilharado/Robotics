
import ast
import os
import pickle
import copy
import pandas as pd

from p1_util.Individual import Individual
from p1_util.Networks import ComplexNet, SimpleNet


class History_Individuals:
    def __init__(self, INDIVIDUALS_HISTORY_PATH, BEST_INDIVIDUAL_PATH, INDIVIDUAL_TYPE):
        """
        Initialize the History_Individuals class

        Parameters
        ----------
        INDIVIDUALS_HISTORY_PATH : str
            Path to save the history of the individuals
        BEST_INDIVIDUAL_PATH : str
            Path to save the best individual
        INDIVIDUAL_TYPE : str
            Type of Individual to use, can be "BRAITENBERG", "NETWORKS_SIMPLE", or "NETWORKS_COMPLEX"

        Notes
        -----
        This method initializes the History_Individuals class and sets the correct create_individual method according to the INDIVIDUAL_TYPE.
        """
        self.individuals_to_save = []
        self.weights_history = []

        self.id_counter = 0

        self.INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH
        self.BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH
        match INDIVIDUAL_TYPE:
            case "BRAITENBERG":
                self.create_individual_class = self._create_individual_raw
            case "NETWORKS_SIMPLE":
                self.create_individual_class = self._create_individual_simple_net
            case _:
                self.create_individual_class = self._create_individual_complex_net

# Create Individual
    def _create_individual_raw(self, gen_number, id_counter, weights):
        """
        Creates an Individual object from given parameters.

        Parameters
        ----------
        gen_number : int
            Generation number of the individual
        id_counter : int
            Unique identifier for the individual
        weights : list
            List of weights that define the individual

        Returns
        -------
        Individual
            The created Individual object
        """
        return Individual(gen_number, id_counter, weights)
    
    def _create_individual_simple_net(self, gen_number, id_counter, weights):
        """
        Creates a SimpleNet object from given parameters.

        Parameters
        ----------
        gen_number : int
            Generation number of the individual.
        id_counter : int
            Unique identifier for the individual.
        weights : list
            List of weights that define the neural network of the individual.

        Returns
        -------
        SimpleNet
            The created SimpleNet object.
        """

        return SimpleNet(gen_number, id_counter, weights)
    
    def _create_individual_complex_net(self, gen_number, id_counter, weights):
        """
        Creates a ComplexNet object from given parameters.

        Parameters
        ----------
        gen_number : int
            Generation number of the individual.
        id_counter : int
            Unique identifier for the individual.
        weights : list
            List of weights that define the neural network of the individual.

        Returns
        -------
        ComplexNet
            The created ComplexNet object.
        """
        return ComplexNet(gen_number, id_counter, weights) 

    def create_individual(self, gen_number, weights):
        """
        Creates a new individual with the given weights if it doesn't already exist.

        Parameters
        ----------
        gen_number : int
            Generation number of the individual.
        weights : list
            List of weights that define the neural network of the individual.

        Returns
        -------
        Individual
            The created individual if it doesn't already exist, None otherwise.
        """
        def _add_weights(weights):
            self.weights_history.append(weights)

        def _exists(weights):
            return weights in self.weights_history
        
        if _exists(weights):
            return None
        
        _add_weights(weights)
        self.id_counter += 1
        return self.create_individual_class(gen_number, self.id_counter, weights)

# Updating History
    def add(self, individual):
        """
        Adds a new individual to the history.

        Parameters
        ----------
        individual : Individual
            The individual to be added to the history.

        Notes
        -----
        This method uses a deep copy of the individual to ensure that the history isn't modified by
        external factors.
        """
        self.individuals_to_save.append(copy.deepcopy(individual))

    def add_all(self, individuals):
        """
        Adds multiple individuals to the history.

        Parameters
        ----------
        individuals : list
            List of individuals to be added to the history.

        Notes
        -----
        This method sorts the individuals by their id before adding them to the history.
        """

        sorted_individuals = Individual.sort_individuals_by_id(individuals)
        for individual in sorted_individuals:
            self.add(individual)


# Saves & Loads
    def save_history(self):
        """
        Saves the history of individuals to a CSV file.

        This method saves the history of individuals to a CSV file specified by `INDIVIDUALS_HISTORY_PATH`.
        If the file does not exist, it creates a new one and writes the header. If the file exists, it appends
        the new individuals to the file. The individuals are sorted by their id before being written to the file.

        Notes
        -----
        This method uses a deep copy of the individuals to ensure that the history isn't modified by
        external factors.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
        """
        Load the history of individuals from a CSV file.

        This method reads the individuals' history from a CSV file specified by `INDIVIDUALS_HISTORY_PATH`.
        It updates the `weights_history` with the weights of each individual and sets the `id_counter`
        to the maximum ID found in the file plus one.

        Notes
        -----
        The method assumes that the CSV file contains a column "Weights" that stores the weights as
        string representations of lists, and a column "ID" that contains integer identifiers.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        def update_history_weights(row):
            self.weights_history.append(ast.literal_eval(row["Weights"]))
        
        df = pd.read_csv(self.INDIVIDUALS_HISTORY_PATH)

        self.id_counter = df["ID"].max() + 1
        df.apply(update_history_weights, axis=1)


    def save_best_individual(self):
        """
        Save the best individual to a pickle file.

        This method saves the best individual (highest fitness) in the `individuals_to_save` list to a pickle file
        specified by `BEST_INDIVIDUAL_PATH`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with open(self.BEST_INDIVIDUAL_PATH, 'wb') as f:
            pickle.dump(Individual.sort_individuals(self.individuals_to_save)[0], f)

    def load_best_individual(self):
        """
        Load the best individual from a pickle file.

        This method loads the best individual (highest fitness) from a pickle file specified by
        `BEST_INDIVIDUAL_PATH`.

        Parameters
        ----------
        None

        Returns
        -------
        Individual
            The best individual loaded from the pickle file.
        """
        with open(self.BEST_INDIVIDUAL_PATH, 'rb') as f:
            return pickle.load(f)