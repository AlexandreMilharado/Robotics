import sys
import os

# Add the parent 'controllers' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from p1_util.Evolution_Manager import Evolution_Manager
from p1_util.Individual import Individual


# Simulation parameters
    # Webots
TIME_STEP_MULTIPLIER = 6.4                      # Webots timestep
EVALUATION_TIME = 100                           # Simulated seconds per individual

    # Evolutionary
        # Model
SENSOR_TYPE = "SIMPLE"
INDIVIDUAL_TYPE = "BRAITENBERG"

# Saving and Loading
INDIVIDUALS_HISTORY_PATH = "../p1_util/evolutionary.csv"    # Path for Individuals History
BEST_INDIVIDUAL_PATH = "../p1_util/best_individual.pkl"     # Path for Best Individual

# Main evolutionary loop
def main():

    # Run the evolutionary algorithm
    controller = Evolution_Manager( SENSOR_TYPE = SENSOR_TYPE,
                                    TIMESTEP_MULTIPLIER = TIME_STEP_MULTIPLIER,
                                    INDIVIDUALS_HISTORY_PATH = INDIVIDUALS_HISTORY_PATH,
                                    BEST_INDIVIDUAL_PATH = BEST_INDIVIDUAL_PATH)


    individual = controller.load_best_individual()
    # individual = Individual(1,[0.9862291581572298, -0.24582283053137227, 0.6494684887575346, -0.2674412457506431, 0.982340210870061, 0.9195890514638527])

    controller.run_individual(individual)

if __name__ == "__main__":
    main()