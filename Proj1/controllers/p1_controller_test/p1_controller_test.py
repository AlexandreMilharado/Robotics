import sys
import os

# Add the parent 'controllers' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from p1_util.evolution_manager import Evolution_Manager, Individual


# Simulation parameters
    # Webots
TIME_STEP_MULTIPLIER = 5


# Main evolutionary loop
def main():

    # Run the evolutionary algorithm
    controller = Evolution_Manager(timestep_multiplier = TIME_STEP_MULTIPLIER)

    individual = controller.load_best_individual()
    controller.run_individual(individual)

if __name__ == "__main__":
    main()