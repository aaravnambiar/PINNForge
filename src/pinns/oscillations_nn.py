import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from oscillations import oscillator
import torch.nn as nn
import torch.optim as optim

t_final=10.0
seed=42
n_data=40 
t_train_max=3.0
w0=2.0 
d = 0.2
noise_std=0.05


def generate_data(d=d, w0=w0):

    # Exact solution
    t_exact = np.linspace(0, t_final, 500)
    u_exact = oscillator(d, w0, t_exact)

    # Training observations
    rng = np.random.default_rng(seed)
    t_data = np.sort(rng.uniform(0, t_train_max, n_data))
    u_data = oscillator(d, w0, t_data)

    # Add measurement noise
    u_data_noisy = u_data + noise_std * rng.standard_normal(n_data)

    # Convert to PyTorch tensors
    t_data_tensor = torch.tensor(
        t_data.reshape(-1, 1), dtype=torch.float32
    )

    u_data_tensor = torch.tensor(
        u_data_noisy.reshape(-1, 1), dtype=torch.float32
    )

    t_test_tensor = torch.tensor(
        t_exact.reshape(-1, 1), dtype=torch.float32
    )

    return (
        t_data, u_data_noisy, t_exact, u_exact,
        t_data_tensor, u_data_tensor, t_test_tensor
    )

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SimpleNN(nn.Module):
    def __init__(self, hidden_size=10, n_layers=3,
                 activation=Sine):
        super().__init__()

        layers = [nn.Linear(1, hidden_size)]

        for _ in range(n_layers):
            layers.append(activation())
            layers.append(nn.Linear(hidden_size, hidden_size))

        layers.append(activation())
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)



def train_standard_nn(model, t_data, u_data, epochs=500, lr=1e-3):
    """Train a standard neural network on data only"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    pbar = tqdm(range(epochs), desc="Training Standard NN")
    for epoch in pbar:
        optimizer.zero_grad()
        # Forward pass
        u_pred = model(t_data)
        # Data loss only
        loss = criterion(u_pred, u_data)
        # Backward pass
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Update progress bar with loss information every 100 epochs
        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

    return losses

# Create and train the standard neural network
def run_standard_nn(t_data_tensor, u_data_tensor, t_test_tensor):
    model = SimpleNN()
    losses = train_standard_nn(
        model, t_data_tensor, u_data_tensor
    )

    model.eval()
    with torch.no_grad():
        predictions = model(t_test_tensor).numpy().flatten()

    return model, predictions, losses

if __name__ == "__main__":
    # Generate data
    (
        t_data, u_data_noisy, t_exact, u_exact,
        t_data_tensor, u_data_tensor, t_test_tensor
    ) = generate_data()

    # Run the standard neural network
    standard_model, u_pred_standard, losses_standard = run_standard_nn(
        t_data_tensor, u_data_tensor, t_test_tensor
    )
    plt.figure(figsize=(12, 8))

    # Plot 1: Full comparison
    plt.subplot(2, 1, 1)
    plt.plot(t_exact, u_exact, 'k-', linewidth=3, label='True solution', alpha=0.8)
    plt.plot(t_exact, u_pred_standard, 'b--', linewidth=2, label='Standard NN prediction')
    plt.scatter(t_data, u_data_noisy, color='red', s=80, zorder=5, label='Training data')
    plt.xlabel('Time t')
    plt.ylabel('Displacement u(t)')
    plt.title('Standard Neural Network: Data-Only Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Error analysis
    plt.subplot(2, 1, 2)
    error = np.abs(u_pred_standard - u_exact)
    plt.plot(t_exact, error, 'r-', linewidth=2, label='Absolute error')
    plt.scatter(t_data, np.zeros_like(t_data), color='red', s=80, zorder=5,label='Training data locations')
    plt.xlabel('Time t')
    plt.ylabel('|Error|')
    plt.title('Prediction Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # Compute metrics
    mse = np.mean((u_pred_standard - u_exact)**2)
    max_error = np.max(np.abs(u_pred_standard - u_exact))

    print(f" Standard NN Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max Error: {max_error:.6f}")