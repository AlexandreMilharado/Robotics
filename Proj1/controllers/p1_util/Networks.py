from functools import partial
import math
from p1_util.Individual import Individual
import torch
import torch.nn as nn
import torch.nn.functional as F


SIMPLE_LAYER_SIZE = [{"input": 2,"ouput": 4, "n": 0}, {"input": 4, "ouput" : 2, "n": 2}]
SIMPLE_LAYER_BEST_WEIGHTS = [-0.9484698409854921, 0.6598759107773684, -0.5095500921534255, -0.8614541022863407, 0.7987480117135004, 0.5916484153637585, -0.30942366969565094, -0.24405961604694743, -0.9012619357772516, 0.730737112752867, -0.028453911870992263, -1.0, -0.8881204985511026, -0.6769560352517451, 0.1136556914717895, 0.2887521061208912, -0.05115651031046836, -0.9905591879847275, 1.0, -0.5318621298911128, 0.9941904945064516, 0.21732206108884689]

SIMPLE_LAYER_SIZE_OBS = [{"input": 3,"ouput": 6, "n": 0}, {"input": 6,"ouput": 2, "n": 2}]
SIMPLE_LAYER_BEST_WEIGHTS_OBS = [-0.0526471728176896, -0.20077866217521, 0.04983578304795727, 0.3494852758285263, 0.24430211757107018, -1, 0.41671250059024556, -0.24167325866663972, 0.3823491668922202, 0.0002329031120035241, -0.30958487111745014, -0.5173113761430044, -0.284152922487296, -0.19127484660888877, -0.034734209042819736, -0.2568700631053471, -0.3201014202565545, -0.5942535149101883, 0, 0, 0, 0, 0, 0, 0.30598950450208456, 0.6112512430628934, 1, 0.4211465340897641, 0.18327776457077516, -0.16962996094364352, -0.10478612626560901, 0.02724055064698192, 0.020640144762629868, -0.4480762546689421, -0.44598265282949195, -0.5500768427168481, 0, 0]
# SIMPLE_LAYER_BEST_WEIGHTS_OBS = [0.9219143870275907, 0.5823733425510516, 0.2208237861285511, -0.6660462985514353, 0.49820971175648127, 0.44231179973788576, -1, -0.4536519851485566, 0.29088355047309405, -0.18496189528197285, 0.42824779911349, 0.566098305880921, -0.11617089394456706, 0.023673695683293042, 0.22956230429950558, -0.7092148114503845, 0.004994992466273315, 0.3358431682990998, 0, 0, 0, 0, 0, 0, -0.9764445762200997, 0.04209191991602208, -0.4492406055007087, 0.4686624849687699, 0.6715468309242497, -0.47637466117902094, -0.9808903069133581, -0.48859007308001495, 0.6721687603062454, -0.11143596241348892, -0.7169811954891323, 0.349218349914197, 0, 0]
COMPLEX_LAYER_SIZE_MERGER = [{"input": 5,"ouput": 5, "n": 0}, {"input": 5,"ouput": 2, "n": 2}]

class ParentNet(torch.nn.Module, Individual):
    def __init__(self, gen_number, id, weights):        
        """
        Initialize the ParentNet
        
        Parameters
        ----------
        gen_number : int
            Generation number
        id : int
            ID of the individual
        weights : list
            Weights of the individual
        """
        nn.Module.__init__(self)
        Individual.__init__(self, gen_number, id, weights)
    
    @classmethod
    def set_weights(cls, FC, weights, layers_sizes):
        """
        Set the weights and biases of a fully connected network.

        This method updates the weights and biases of the given neural network layers
        based on the provided weights and layer sizes. It iterates through each layer,
        extracts the corresponding weights and biases, and applies them to the network.

        Parameters
        ----------
        FC : torch.nn.Module
            The fully connected network whose weights and biases are to be set.
        weights : list
            A list of weights to be assigned to the network layers.
        layers_sizes : list of dict
            A list of dictionaries where each dictionary contains the 'input', 'ouput',
            and 'n' keys, representing the size and index of each layer.

        Returns
        -------
        torch.nn.Module
            The fully connected network with updated weights and biases.
        """

        with torch.no_grad():
            index = 0
            for layer_size in layers_sizes:
                connections = []
                for _ in range(layer_size["ouput"]):
                    row = []
                    for _ in range(layer_size["input"]):
                        row.append(weights[index])
                        index += 1
                    connections.append(row)
                bias = []
                for _ in range(layer_size["ouput"]):
                    bias.append(weights[index])
                    index += 1
    
                FC[layer_size["n"]].weight.copy_(torch.tensor(connections))
                FC[layer_size["n"]].bias.copy_(torch.tensor(bias))
        return FC

class SimpleNet(ParentNet):
    def __init__(self, gen_number, id, weights, layers_sizes = SIMPLE_LAYER_SIZE):        
        """
        Initialize the SimpleNet
        
        Parameters
        ----------
        gen_number : int
            Generation number
        id : int
            ID of the individual
        weights : list
            Weights of the individual
        layers_sizes : list of dict
            A list of dictionaries where each dictionary contains the 'input', 'ouput',
            and 'n' keys, representing the size and index of each layer.
        """
        super().__init__(gen_number, id, weights)
        self.weights
        self.layers_sizes = layers_sizes
        self.FC = nn.Sequential(
            nn.Linear(layers_sizes[0]["input"], layers_sizes[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),  
            nn.Tanh()
        )
        
        self.FC  = ParentNet.set_weights(self.FC, weights, layers_sizes)
        
    def forward(self, x):
        """
        Perform a forward pass through the network.

        This method takes an input tensor, processes it through the fully connected
        neural network (FC), and returns the result as a list. The input is converted
        to a tensor of floats before being passed through the network.

        Parameters
        ----------
        x : list
            A list of numerical values representing the input to the network.

        Returns
        -------
        list
            A list of numerical values representing the output of the network after
            processing the input.
        """

        x = torch.tensor([float(value) for value in x])
        x = self.FC(x)
        return x.detach().numpy().tolist()



class ComplexNet(ParentNet):
    def __init__(self, gen_number, id, weights, layers_sizes = COMPLEX_LAYER_SIZE_MERGER):        
        """
        Initialize the ComplexNet

        Parameters
        ----------
        gen_number : int
            Generation number
        id : int
            ID of the individual
        weights : list
            Weights of the individual
        layers_sizes : list of dict
            A list of dictionaries where each dictionary contains the 'input', 'ouput',
            and 'n' keys, representing the size and index of each layer.
        """

        super().__init__(gen_number, id, weights)

        self.layers_sizes = layers_sizes

        black_line_net = nn.Sequential(
            nn.Linear(SIMPLE_LAYER_SIZE[0]["input"], SIMPLE_LAYER_SIZE[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(SIMPLE_LAYER_SIZE[1]["input"], SIMPLE_LAYER_SIZE[1]["ouput"]),  
            nn.Tanh()
        )
        black_line_net = self.load_weights_black_line(black_line_net)

        obstacles_net = nn.Sequential(
            nn.Linear(SIMPLE_LAYER_SIZE_OBS[0]["input"], SIMPLE_LAYER_SIZE_OBS[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(SIMPLE_LAYER_SIZE_OBS[1]["input"], SIMPLE_LAYER_SIZE_OBS[1]["ouput"]),  
            nn.Tanh(),
        )
        obstacles_net = self.load_weights_obstacles(obstacles_net)

        self.experts = nn.ModuleList([black_line_net,
                                      obstacles_net ])
        self.inputs = [lambda l: self.get_elements_to_last(l, -2),
                       lambda l: self.get_elements_from_begin(l, -2)]
        self.router = nn.Sequential(
            nn.Linear(layers_sizes[0]["input"], layers_sizes[0]["ouput"]),
            nn.Tanh(),
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),
            nn.Softmax(dim=-1)
        )
        self.set_weights(weights)



    def load_weights_black_line(self, black_line_net):
        """
        Load and set the weights for the black line neural network.

        This method applies pre-defined optimal weights to the black line network
        using the ParentNet's set_weights method.

        Parameters
        ----------
        black_line_net : torch.nn.Module
            The neural network for which the weights are being set.

        Returns
        -------
        torch.nn.Module
            The neural network with updated weights.
        """

        return ParentNet.set_weights(black_line_net, SIMPLE_LAYER_BEST_WEIGHTS, SIMPLE_LAYER_SIZE)

    def load_weights_obstacles(self, obstacles_net):
        """
        Load and set the weights for the obstacles neural network.

        This method applies pre-defined optimal weights to the obstacles network
        using the ParentNet's set_weights method.

        Parameters
        ----------
        obstacles_net : torch.nn.Module
            The neural network for which the weights are being set.

        Returns
        -------
        torch.nn.Module
            The neural network with updated weights.
        """
        return ParentNet.set_weights(obstacles_net, SIMPLE_LAYER_BEST_WEIGHTS_OBS, SIMPLE_LAYER_SIZE_OBS)
    
    def get_elements(self, list, init, end):
        """
        Get a subset of a list between two indices.

        Parameters
        ----------
        list : list
            The list from which to extract the elements.
        init : int
            The starting index of the subset.
        end : int
            The ending index of the subset.

        Returns
        -------
        list
            A list containing the elements in the range [init, end) of the original list.
        """
        return list[init:end]
    
    def get_elements_to_last(self, list, init):
        """
        Get a subset of a list from a given index to the end of the list.

        Parameters
        ----------
        list : list
            The list from which to extract the elements.
        init : int
            The starting index of the subset.

        Returns
        -------
        list
            A list containing the elements in the range [init, end] of the original list.
        """
        return list[init:]
    
    def get_elements_from_begin(self, list, end):
        """
        Get a subset of a list from the beginning to a given index.

        Parameters
        ----------
        list : list
            The list from which to extract the elements.
        end : int
            The ending index of the subset.

        Returns
        -------
        list
            A list containing the elements in the range [0, end) of the original list.
        """
        return list[:end]

    def set_weights(self, weights):
        """
        Set the weights and biases of the router network.

        This method updates the weights and biases of the router network based on the given weights.

        Parameters
        ----------
        weights : list
            A list of weights to be assigned to the router network layers.

        Returns
        -------
        None
        """
        def get_connections(layer_size):
            """
            Retrieve connection weights for a given layer size.

            This function extracts the connection weights from the weights list for a specified 
            layer based on its input and output sizes. It constructs a matrix of weights, where 
            each row represents the weights connecting the inputs to a specific output. It uses 
            a counter to keep track of the position in the weights list and increments it for 
            each weight value extracted.

            Parameters
            ----------
            layer_size : dict
                A dictionary containing the 'input' and 'ouput' keys, representing the size of 
                the layer's input and output, respectively.

            Returns
            -------
            list
                A list of lists representing the weight matrix for the specified layer, where 
                each inner list corresponds to the weights connecting inputs to a single output.
            """

            connections = []
            for _ in range(layer_size["ouput"]):
                row = []
                for _ in range(layer_size["input"]):
                    # temp.append(weights[index["ws_counter"]])
                    row.append(weights[index["ws_counter"]])
                    index["ws_counter"] += 1
                connections.append(row)

            return connections
        
        def get_bias(layer_size):
            """
            Retrieve the bias values for a given layer size.

            This function extracts the bias values from the weights list for a specified 
            layer based on its output size. It uses a counter to keep track of the position 
            in the weights list and increments it for each bias value extracted.

            Parameters
            ----------
            layer_size : dict
                A dictionary containing the 'ouput' key, representing the size of the layer's output.

            Returns
            -------
            list
                A list containing the bias values for the specified layer.
            """

            bias = []
            for _ in range(layer_size["ouput"]):
                # temp.append(0)
                # bias.append(0)
                bias.append(weights[index["ws_counter"]])
                index["ws_counter"] += 1

            return bias
        
        with torch.no_grad():
            index = {"ws_counter" : 0}
            temp = []
            for layer_size_index in range(len(self.layers_sizes)):
                layer_size = self.layers_sizes[layer_size_index]
                connections = get_connections(layer_size)
                bias = get_bias(layer_size)
    
                self.router[layer_size["n"]].weight.copy_(torch.tensor(connections))
                self.router[layer_size["n"]].bias.copy_(torch.tensor(bias))
            
            # self.weights = temp
    def forward(self, x):
        """
        Perform a forward pass through the network.

        This method takes an input tensor, processes it through the fully connected
        neural network (FC), and returns the result as a list. The input is converted
        to a tensor of floats before being passed through the network.

        Parameters
        ----------
        x : list
            A list of numerical values representing the input to the network.

        Returns
        -------
        list
            A list of numerical values representing the output of the network after
            processing the input.
        """
        normalised = [float(value > 0) for value in x[:3]] 
        normalised += x[3:]
        idx = torch.argmax(self.router(torch.tensor(normalised))).item()
        x = torch.tensor([float(value) for value in x])
        velocities = self.experts[idx](self.inputs[idx](x))

        # return idx
        return velocities.detach().numpy().tolist()