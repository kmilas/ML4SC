import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np

# NOELSTM Model
class NOELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False, dropout=0.0):
        super(NOELSTM, self).__init__()
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0).double()

        self.fc = nn.Linear(hidden_size * direction_factor, output_size).double()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Create Sequences for NOE
def create_sequences(theta, omega, u, seq_len):
    X, Y = [], []
    for i in range(len(theta) - seq_len):
        x_seq = np.stack([theta[i:i+seq_len], omega[i:i+seq_len], u[i:i+seq_len]], axis=1)
        y_target = [theta[i+seq_len], omega[i+seq_len]]
        X.append(x_seq)
        Y.append(y_target)
    return np.array(X), np.array(Y)

# Main Training Function
def train():
    wandb.init()
    config = wandb.config

    run_name = (
        f"hs{config.hidden_size}_lr{config.learning_rate}_nl{config.num_layers}_"
        f"bs{config.batch_size}_sl{config.sequence_length}_ep{config.epochs}_"
        f"bi{int(config.bidirectional)}_do{config.dropout}"
    )

    # Set the WandB run name
    wandb.run.name = run_name
    wandb.run.save()

    # Load preprocessed data
    theta = np.load("theta.npy")  # shape: (T,)
    omega = np.load("omega.npy")  # shape: (T,)
    u = np.load("u.npy")          # shape: (T,)

    # Normalize
    theta_mean, theta_std = theta.mean(), theta.std()
    omega_mean, omega_std = omega.mean(), omega.std()
    u_mean, u_std = u.mean(), u.std()

    theta = (theta - theta_mean) / theta_std
    omega = (omega - omega_mean) / omega_std
    u = (u - u_mean) / u_std

    train_NRMS_list = []
    val_NRMS_list = []

    # Create sequences
    X, Y = create_sequences(theta, omega, u, config.sequence_length)

    # Convert to tensors
    X_tensor = torch.tensor(X).double()
    Y_tensor = torch.tensor(Y).double()

    # Split: 80% train, 10% val, 10% test
    total = len(X_tensor)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    X_train, Y_train = X_tensor[:train_end], Y_tensor[:train_end]
    X_val, Y_val = X_tensor[train_end:val_end], Y_tensor[train_end:val_end]
    X_test, Y_test = X_tensor[val_end:], Y_tensor[val_end:]

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Initialize model
    model = NOELSTM(input_size=3,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    output_size=2,
                    bidirectional=config.bidirectional,
                    dropout=config.dropout if hasattr(config, 'dropout') else 0.0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), config.batch_size):
            indices = permutation[i:i + config.batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            train_loss = loss_fn(train_pred, Y_train)
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val)

            train_nrms = train_loss.item() / torch.std(Y_train).item()
            val_nrms = val_loss.item() / torch.std(Y_val).item()
            rmse = torch.sqrt(val_loss).item()

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
                "train_NRMS": train_nrms,
                "val_NRMS": val_nrms,
                "rmse": rmse
            })

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), f"best_model_{wandb.run.name}.pt")

            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # Create a descriptive filename from config
    param_str = (
        f"hs{config.hidden_size}_lr{config.learning_rate}_nl{config.num_layers}_"
        f"bs{config.batch_size}_sl{config.sequence_length}_ep{config.epochs}_"
        f"bi{int(config.bidirectional)}_do{config.dropout}"
    )

    # Save model
    model_path = f"model_{param_str}.pt"
    torch.save(model.state_dict(), model_path)

# Script Entry Point
if __name__ == "__main__":
    train()
