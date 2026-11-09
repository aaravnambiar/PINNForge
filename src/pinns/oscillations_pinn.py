
import torch
import numpy as np
from oscillations import oscillator
from oscillations_nn import (SimpleNN , generate_data, train_standard_nn)
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# def compute_derivatives_demo():
#     """Demonstrate automatic differentiation"""
#     # Create a simple test case: u(t) = sin(t)
#     t = torch.tensor([0.5], requires_grad=True)  # Enable gradient computation
#     u = torch.sin(t)  # u = sin(t)

#     print("Function: u(t) = sin(t)")
#     print(f"At t = {t.item():.2f}:")
#     print(f"  u = {u.item():.6f}")

#     # First derivative: du/dt
#     du_dt = torch.autograd.grad(u, t, create_graph=True)[0]
#     print(f"  du/dt = {du_dt.item():.6f} (exact: cos({t.item():.2f}) = {np.cos(t.item()):.6f})")

#     # Second derivative: d²u/dt²
#     d2u_dt2 = torch.autograd.grad(du_dt, t, create_graph=True)[0]
#     print(f"  d²u/dt² = {d2u_dt2.item():.6f} (exact: -sin({t.item():.2f}) = {-np.sin(t.item()):.6f})")

#     print("Automatic differentiation gives exact derivatives!")

# compute_derivatives_demo()

d = 0.2
w0 = 2.0

(
    t_data, u_data_noisy, t_exact, u_exact,
    t_data_tensor, u_data_tensor, t_test_tensor
) = generate_data(d=d, w0=w0)



def physics_loss(model, t_colloc, m, c, k):
    """
    Compute the physics loss for the damped harmonic oscillator

    ODE: m * d²u/dt² + c * du/dt + k * u = 0
    """
    # Ensure gradients are enabled for input
    t_colloc = t_colloc.clone().detach().requires_grad_(True)

    # Forward pass: compute u(t)
    u = model(t_colloc)

    # First derivative: du/dt
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_colloc,
        grad_outputs=torch.ones_like(u),
        create_graph=True,  # Allow higher-order derivatives
        retain_graph=True
    )[0]

    # Second derivative: d²u/dt²
    d2u_dt2 = torch.autograd.grad(
        outputs=du_dt,
        inputs=t_colloc,
        grad_outputs=torch.ones_like(du_dt),
        create_graph=True,
        retain_graph=True
    )[0]

    # Physics residual: R = m * d²u/dt² + c * du/dt + k * u
    residual = m * d2u_dt2 + c * du_dt + k * u

    # Mean squared residual
    physics_loss = torch.mean(residual**2)

    return physics_loss

def compute_physics_residual(model, t, m, c, k):

    t = t.clone().detach().requires_grad_(True)

    u = model(t)

    du_dt = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    d2u_dt2 = torch.autograd.grad(
        du_dt, t,
        grad_outputs=torch.ones_like(du_dt),
        create_graph=True
    )[0]

    residual = m*d2u_dt2 + c*du_dt + k*u

    return residual


def train_pinn(model, t_data, u_data, t_colloc, m, c, k,
               epochs=10000, lr=1e-3, lambda_physics = 1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Storage for loss history
    data_losses = []
    physics_losses = []
    total_losses = []

    print(f"Training PINN with λ = {lambda_physics}")
    print(f"Data points: {len(t_data)}, Collocation points: {len(t_colloc)}")

    pbar = tqdm(range(epochs), desc="Training PINN")
    for epoch in pbar:
        optimizer.zero_grad()

        # Data loss: how well do we fit the measurements?
        u_pred_data = model(t_data)
        loss_data = criterion(u_pred_data, u_data)

        # Physics loss: how well do we satisfy the ODE?
        loss_physics = physics_loss(model, t_colloc, m, c, k)

        # Total loss: balance data fitting and physics
        total_loss = loss_data + lambda_physics * loss_physics

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Store losses
        data_losses.append(loss_data.item())
        physics_losses.append(loss_physics.item())
        total_losses.append(total_loss.item())

        if (epoch + 1) % 2000 == 0:
            pbar.set_postfix({'Loss': f'{loss_data.item():.6f}',
                              'Physics': f'{loss_physics.item():.6f}',
                              'Total': f'{total_loss.item():.6f}'})

    return data_losses, physics_losses, total_losses

def refine_lbfgs(model, t_data, u_data, t_colloc,
                 m, c, k, lambda_physics=1.0):

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=500,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()

        u_pred = model(t_data)
        loss_data = torch.mean((u_pred - u_data)**2)

        loss_phy = physics_loss(model, t_colloc, m, c, k)

        loss = loss_data + lambda_physics * loss_phy
        loss.backward()

        return loss

    optimizer.step(closure)


# Physics parameters matching the analytical oscillator
m = 1.0
c = 2 * d
k = w0**2

# Physics enforced over the full domain
n_colloc = 200
t_colloc = torch.linspace(0, 10, n_colloc).reshape(-1, 1)

# Physics loss weighting
lambda_physics = 1

# Initialize PINN

torch.manual_seed(42)

standard_nn = SimpleNN()
pinn_model = SimpleNN()

# Give both models identical initial weights
pinn_model.load_state_dict(standard_nn.state_dict())

losses_standard = train_standard_nn(
    standard_nn,
    t_data_tensor,
    u_data_tensor,
    epochs=500
)

standard_nn.eval()
with torch.no_grad():
    u_pred_standard = (
        standard_nn(t_test_tensor).numpy().flatten()
    )

# Stage 1: Adam training
data_losses, physics_losses, total_losses = train_pinn(
    pinn_model,
    t_data_tensor,
    u_data_tensor,
    t_colloc,
    m=m,
    c=c,
    k=k,
    epochs=1000,
    lambda_physics=lambda_physics
)

print("Physics loss after Adam:",
      physics_loss(pinn_model, t_colloc, m, c, k).item())

# Stage 2: L-BFGS refinement
refine_lbfgs(
    pinn_model,
    t_data_tensor,
    u_data_tensor,
    t_colloc,
    m, c, k,
    lambda_physics=lambda_physics
)

print("Physics loss after L-BFGS:",
      physics_loss(pinn_model, t_colloc, m, c, k).item())

# Make predictions AFTER both training stages
pinn_model.eval()

# Make predictions
with torch.no_grad():
    u_pred_pinn = pinn_model(t_test_tensor).numpy().flatten()


# Generate time points across the full domain
t_residual = torch.linspace(0, 10, 500).reshape(-1, 1)

# Compute residuals
residual = compute_physics_residual(
    pinn_model, t_residual, m, c, k
)

# Convert to NumPy
residual_np = residual.detach().numpy().flatten()
t_np = t_residual.detach().numpy().flatten()

# Plot
plt.figure(figsize=(10, 5))

plt.plot(t_np, residual_np, label="Physics residual")

plt.axhline(0, color="black", linestyle="--",
            label="Ideal residual = 0")

plt.axvline(3, color="red", linestyle=":",
            label="End of training data")

plt.xlabel("Time t")
plt.ylabel("Residual R(t)")
plt.title("PINN Physics Residual vs Time")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()



print(f"PINN Training Complete!")
print(f"Final data loss: {data_losses[-1]:.6f}")
print(f"Final physics loss: {physics_losses[-1]:.6f}")

# Compare Standard NN vs PINN
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Solutions comparison
ax = axes[0, 0]
ax.plot(t_exact, u_exact, 'k-', linewidth=3, label='True solution', alpha=0.8)
ax.plot(t_exact, u_pred_standard, 'b--', linewidth=2, label='Standard NN', alpha=0.8)
ax.plot(t_exact, u_pred_pinn, 'r-', linewidth=2, label='PINN', alpha=0.8)
ax.scatter(t_data, u_data_noisy, color='red', s=80, zorder=5, label='Training data')
ax.set_xlabel('Time t')
ax.set_ylabel('Displacement u(t)')
ax.set_title('Solution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Error comparison
ax = axes[0, 1]
error_standard = np.abs(u_pred_standard - u_exact)
error_pinn = np.abs(u_pred_pinn - u_exact)
ax.plot(t_exact, error_standard, 'b-', linewidth=2, label='Standard NN error')
ax.plot(t_exact, error_pinn, 'r-', linewidth=2, label='PINN error')
ax.set_xlabel('Time t')
ax.set_ylabel('Absolute Error')
ax.set_title('Error Comparison')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Loss evolution for Standard NN
ax = axes[1, 0]
ax.plot(losses_standard, 'b-', linewidth=2, label='Data loss only')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Standard NN Training')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Loss evolution for PINN
ax = axes[1, 1]
ax.plot(data_losses, 'b-', linewidth=2, label='Data loss', alpha=0.8)
ax.plot(physics_losses, 'g-', linewidth=2, label='Physics loss', alpha=0.8)
ax.plot(total_losses, 'r-', linewidth=2, label='Total loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('PINN Training (Multiple Loss Components)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Quantitative comparison
mse_standard = np.mean((u_pred_standard - u_exact)**2)
mse_pinn = np.mean((u_pred_pinn - u_exact)**2)
max_error_standard = np.max(np.abs(u_pred_standard - u_exact))
max_error_pinn = np.max(np.abs(u_pred_pinn - u_exact))

print(" QUANTITATIVE COMPARISON")
print("="*50)
print(f"{'Metric':<20} {'Standard NN':<15} {'PINN':<15} {'Improvement':<15}")
print("-"*65)
print(f"{'MSE':<20} {mse_standard:<15.6f} {mse_pinn:<15.6f} {mse_standard/mse_pinn:<15.1f}x")
print(f"{'Max Error':<20} {max_error_standard:<15.6f} {max_error_pinn:<15.6f} {max_error_standard/max_error_pinn:<15.1f}x")

print(f" PINN is {mse_standard/mse_pinn:.0f}x more accurate!")

t_inside = torch.linspace(0, 3, 100).reshape(-1, 1)
t_outside = torch.linspace(3, 10, 200).reshape(-1, 1)

loss_inside = physics_loss(pinn_model, t_inside, m, c, k)
loss_outside = physics_loss(pinn_model, t_outside, m, c, k)
print("Physics loss [0,3]:", loss_inside.item())
print("Physics loss [3,10]:", loss_outside.item())