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
        # Init
GENERATIONS_CONVERGE_STOP = 200                 # N Generations
POPULATION_SIZE = 20                            # N Individuals per Generation

        # Selection
PARENTS_KEEP = 4                                # N Individuals to keep

        # Mutation
MUTATION_RATE = 1                               # N * MUTATION_RATE in N New Individuals
MUTATION_ALTER = 0.17                           # N * MUTATION_ALTER in N Genes
    
def main():

    # Run the evolutionary algorithm
    controller = Evolution_Manager(timestep_multiplier = TIME_STEP_MULTIPLIER)
    controller.load_train_params(generation_converge_stop = GENERATIONS_CONVERGE_STOP,
                                   population_size = POPULATION_SIZE,
                                   selection_number = PARENTS_KEEP,
                                   parents_number = (POPULATION_SIZE - PARENTS_KEEP) * 2,
                                   mutation_rate = MUTATION_RATE,
                                   mutation_alter_rate = MUTATION_ALTER,
                                   evaluation_time = EVALUATION_TIME)
    
    controller.load_training()
    

    controller.train_all()
    controller.reset_center()

if __name__ == "__main__":
    main()