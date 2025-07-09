import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        x = self.hidden(x)  # Pass through hidden layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)  # Pass through output layer
        return x
