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
    # individual = Individual(1,[0.9862291581572298, -0.24582283053137227, 0.6494684887575346, -0.2674412457506431, 0.982340210870061, 0.9195890514638527])
    # {ID: 845, Weights: [0.9934010776087957, -0.2612430660718584, 0.11023021885718766, 0.3123250821246174, 0.38145968918147233, 0.25697293717577774], Fitness: 0}
    controller.run_individual(individual)

if __name__ == "__main__":
    main()