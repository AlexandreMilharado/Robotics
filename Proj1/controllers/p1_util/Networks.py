from p1_util.Individual import Individual
import torch
import torch.nn as nn


SIMPLE_LAYER_SIZE = [{"input": 2,"ouput": 4, "n": 0}, {"input": 4, "ouput" : 2, "n": 2}]

COMPLEX_LAYER_SIZE = [{"input": 5,"ouput": 12, "n": 0}, {"input": 12,"ouput": 24, "n": 2}, {"input": 24, "ouput" : 2, "n": 4}]


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
            nn.GELU(), 
            nn.Linear(layers_sizes[1]["input"], layers_sizes[1]["ouput"]),  
            nn.GELU(),
            nn.Linear(layers_sizes[2]["input"], layers_sizes[2]["ouput"]),  
            nn.Tanh(),
            # nn.Conv2d(layers_sizes[2]["input"], layers_sizes[2]["ouput"], kernel_size=1)
        )

        self.set_FC(FC)
        self.set_weights(weights)
