import sys
import os


# Add the parent 'controllers' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from p1_util.evolution_manager import Evolution_Manager


# Simulation parameters
    # Webots
TIME_STEP_MULTIPLIER = 6.4                      # Webots timestep
EVALUATION_TIME = 100                           # Simulated seconds per individual

    # Evolutionary
        # Model
INDIVIDUAL_TYPE = "NETWORKS_COMPLEX"            # "BRAITENBERG" | "NETWORKS_SIMPLE" | "NETWORKS_COMPLEX"

        # Init
GENERATION_LIMIT = 500                          # N Generations
POPULATION_SIZE = 20                            # N Individuals per Generation

        # Selection
SELECTION_NUMBER = 4                             # N Individuals to keep

        # Mutation
MUTATION_RATE = 1                               # N * MUTATION_RATE in N New Individuals
MUTATION_ALTER = 30/112                            # N * MUTATION_ALTER in N Genes
    
# Saving and Loading
INDIVIDUALS_HISTORY_PATH = "../p1_util/evolutionary.csv"    # Path for Individuals History
BEST_INDIVIDUAL_PATH = "../p1_util/best_individual.pkl"     # Path for Best Individual

def main():

    # Run the evolutionary algorithm
    controller = Evolution_Manager( INDIVIDUAL_TYPE = INDIVIDUAL_TYPE,
                                    TIMESTEP_MULTIPLIER = TIME_STEP_MULTIPLIER,
                                    INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH,
                                    BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH)
    
    controller.load_train_params(   population_size = POPULATION_SIZE,
                                    generation_limit = GENERATION_LIMIT,
                                    evaluation_time = EVALUATION_TIME,
                                    selection_number = SELECTION_NUMBER,
                                    mutation_rate = MUTATION_RATE,
                                    mutation_alter_rate = MUTATION_ALTER)
    
    controller.load_training()
    

    controller.train_all()
    controller.reset()

if __name__ == "__main__":
    main()