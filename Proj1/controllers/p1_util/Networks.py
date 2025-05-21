import math
from p1_util.Individual import Individual
import torch
import torch.nn as nn
import torch.nn.functional as F


SIMPLE_LAYER_SIZE = [{"input": 2,"ouput": 4, "n": 0}, {"input": 4, "ouput" : 2, "n": 2}]

COMPLEX_LAYER_SIZE = [{"input": 5,"ouput": 5, "n": 0}, {"input": 5,"ouput": 10, "n": 2}]
COMPLEX_LAYER_SIZE_EXTRA = [{"input": 10, "ouput" : 1, "n": 0}]
COMPLEX_LAYER_SIZE_EXTRA2 = [{"input": 10, "ouput" : 1, "n": 0}]

class ParentNet(torch.nn.Module, Individual):
    def __init__(self, gen_number, id, weights, layers_sizes):        
        nn.Module.__init__(self)
        Individual.__init__(self, gen_number, id, weights)

        self.layers_sizes = layers_sizes
        self.FC = nn.Sequential(
            nn.Linear(layers_sizes[0]["input"], layers_sizes[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),  
            nn.Tanh()
        )


    def forward(self, x):
        x = torch.tensor([float(value) for value in x])
        x = self.FC(x)
        return x.detach().numpy().tolist()
    
    def set_FC(self, FC):
        self.FC = FC
    
    def set_weights(self, weights):
        with torch.no_grad():
            index = 0
            for layer_size in self.layers_sizes:
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
    
                self.FC[layer_size["n"]].weight.copy_(torch.tensor(connections))
                self.FC[layer_size["n"]].bias.copy_(torch.tensor(bias))


class SimpleNet(ParentNet):
    def __init__(self, gen_number, id, weights, layers_sizes = SIMPLE_LAYER_SIZE):        
        super().__init__(gen_number, id, weights, layers_sizes)

        self.layers_sizes = layers_sizes
        FC = nn.Sequential(
            nn.Linear(layers_sizes[0]["input"], layers_sizes[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),  
            nn.Tanh()
        )
        
        self.set_FC(FC)
        self.set_weights(weights)


class ComplexNet(ParentNet):
    def __init__(self, gen_number, id, weights, layers_sizes = COMPLEX_LAYER_SIZE):        
        super().__init__(gen_number, id, weights, layers_sizes)

        self.layers_sizes = layers_sizes
        FC = nn.Sequential(
            nn.Linear(layers_sizes[0]["input"], layers_sizes[0]["ouput"]),
            nn.Tanh(), 
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),  
            nn.Tanh(),
        )

        self.vel_head = nn.Linear(COMPLEX_LAYER_SIZE_EXTRA[0]["input"], COMPLEX_LAYER_SIZE_EXTRA[0]["ouput"])
        self.angle_head = nn.Linear(COMPLEX_LAYER_SIZE_EXTRA2[0]["input"], COMPLEX_LAYER_SIZE_EXTRA2[0]["ouput"])

        self.set_FC(FC)
        self.set_weights(weights)

    def set_weights(self, weights):
        def get_connections(layer_size):
            connections = []
            for _ in range(layer_size["ouput"]):
                row = []
                for _ in range(layer_size["input"]):
                    row.append(weights[index["ws_counter"]])
                    index["ws_counter"] += 1
                connections.append(row)

            return connections
        
        def get_bias(layer_size):
            bias = []
            for _ in range(layer_size["ouput"]):
                bias.append(weights[index["ws_counter"]])
                index["ws_counter"] += 1

            return bias
        
        with torch.no_grad():
            index = {"ws_counter" : 0}
            for layer_size_index in range(len(self.layers_sizes)):
                layer_size = self.layers_sizes[layer_size_index]
                connections = get_connections(layer_size)
                bias = get_bias(layer_size)
    
                self.FC[layer_size["n"]].weight.copy_(torch.tensor(connections))
                self.FC[layer_size["n"]].bias.copy_(torch.tensor(bias))
            
            connections = get_connections(COMPLEX_LAYER_SIZE_EXTRA[0])
            bias = get_bias(COMPLEX_LAYER_SIZE_EXTRA[0])
            self.vel_head.weight.copy_(torch.tensor(connections))
            self.vel_head.bias.copy_(torch.tensor(bias))

            connections = get_connections(COMPLEX_LAYER_SIZE_EXTRA2[0])
            bias = get_bias(COMPLEX_LAYER_SIZE_EXTRA2[0])
            self.angle_head.weight.copy_(torch.tensor(connections))
            self.angle_head.bias.copy_(torch.tensor(bias))


    def forward(self, x):
        x = torch.tensor([float(value) for value in x])

        features = self.FC(x)
        angle = (math.pi * torch.tanh(self.angle_head(features)))

        # probs = F.softmax(angle, dim=-1)          
        # angle_idx = torch.argmax(probs).item()     

        # vel_input = torch.cat([features, probs], dim=-1)
        # velocity = torch.tanh(self.vel_head(vel_input)).item()
        velocity = torch.tanh(self.vel_head(features)).item()

        return [angle, velocity]