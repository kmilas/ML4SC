import numpy as np
import wandb
import torch
from torch import nn
from sklearn.model_selection import train_test_split

# Load data
out = np.load('../../training-val-test-data.npz')
th = out['th']
u= out['u']

split_train = int(0.8 * th.shape[0])
split_test = int(0.9 * th.shape[0])

# Data preparation
def create_IO_data(u, y, na, nb):
    X, Y = [], []
    for k in range(max(na, nb), len(y)):
        X.append(np.concatenate([u[k-nb:k], y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

# Activation map
activation_functions = {
    'tanh': torch.tanh,
    'leaky_relu': torch.nn.functional.leaky_relu
}

# Define model
class Network(nn.Module):
    def __init__(self, n_in, n_hidden_nodes, n_hidden_layers=1, activation='leaky_relu'):
        super(Network, self).__init__()
        self.activation = activation_functions[activation]
        layers = [nn.Linear(n_in, n_hidden_nodes).double()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes).double())
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(n_hidden_nodes, 1).double()

    def forward(self, u):
        x = u
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        y = self.output_layer(x)[:, 0]
        return y

# Training function (required for sweeps)
def train():
    wandb.init(project="5SC28-NN-gridsearch")
    config = wandb.config

    # Create data
    #X, Y = create_IO_data(u_train, th_train, config.na, config.nb)
    # 80-10-10 Split
    #X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.5, random_state=42)
    

    u_train,  th_train  = u[:split_train],  th[:split_train]
    u_valid,  th_valid  = u[split_train:split_test], th[split_train:split_test]
    u_test,   th_test   = u[split_test:],     th[split_test:]

    X_train, Y_train = create_IO_data(u_train, th_train, config.na, config.nb)
    X_val, Y_val = create_IO_data(u_valid, th_valid, config.na, config.nb)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(
        n_in=config.na + config.nb,
        n_hidden_nodes=config.hidden_neurons,
        n_hidden_layers=config.hidden_layers,
        activation=config.activation
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    Xtrain_tensor = torch.tensor(X_train).to(device)
    Ytrain_tensor = torch.tensor(Y_train).to(device)
    Xval_tensor = torch.tensor(X_val).to(device)
    Yval_tensor = torch.tensor(Y_val).to(device)

    for epoch in range(config.epochs):
        model.train()
        for i in range(0, len(Xtrain_tensor), config.batch_size):
            optimizer.zero_grad()
            y_pred = model(Xtrain_tensor[i:i+config.batch_size])
            loss = torch.mean((y_pred - Ytrain_tensor[i:i+config.batch_size]) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_tensor)
            val_loss = torch.mean((val_pred - Yval_tensor) ** 2).item()
            rmse = torch.sqrt(torch.mean((val_pred - Yval_tensor) ** 2)).item()
            val_nrms = val_loss / torch.std(Yval_tensor).item()
            train_nrms = loss.item() / torch.std(Ytrain_tensor).item()

        wandb.log({
            "epoch": epoch,
            "train_loss": loss.item(),
            "train_NRMS": train_nrms,
            "val_loss": val_loss,
            "val_NRMS": val_nrms,
            "rmse": rmse
        })

    # Save model
    torch.save(
        model.state_dict(),
        f"model_na{config.na}_nb{config.nb}_hl{config.hidden_layers}_hn{config.hidden_neurons}_{config.activation}.pt"
    )
    wandb.finish()

# Entry point
if __name__ == "__main__":
    train()
