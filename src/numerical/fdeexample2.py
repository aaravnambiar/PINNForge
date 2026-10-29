import matplotlib.pyplot as plt
import numpy as np

# Physical parameters
L = 1.0
c = 1.0
t_final = 2.0
A = 100.0

# Points
Nx = 51
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Time points
r = 0.5
dt = r * dx / c
Nt = int(np.ceil(t_final / dt))
dt = t_final / Nt
r = c * dt / dx

u = A * np.sin(np.pi * x / L)

v = np.zeros(Nx)

u[0] = 0.0
u[-1] = 0.0

u_0 = u.copy()
u_1 = np.zeros(Nx)
u_1[1:-1] = u_0[1:-1] + r**2 *0.5* (
        u_0[2:] - 2*u_0[1:-1] + u_0[:-2]
    )
u_1[0] = 0.0
u_1[-1] = 0.0


history = [u_0]
history.append(u_1)
times = [0.0]

# Time stepping
for n in range(1, Nt):

    u_new = np.zeros(Nx)

    # Apply the wave equation
    u_new[1:-1] = (
        2 * u_1[1:-1]
        - u_0[1:-1]
        + r**2 * (
            u_1[2:]
            - 2 * u_1[1:-1]
            + u_1[:-2]
        )
    )

    # Boundary conditions
    u_new[0] = 0.0
    u_new[-1] = 0.0

    # Store calculated displacement
    history.append(u_new.copy())
    times.append((n + 1) * dt)

    # Shift time levels forward
    u_0 = u_1.copy()
    u_1 = u_new.copy()

    # Compare numerical and analytical solution at final time
u_exact = A * np.sin(np.pi * x / L) * np.cos(
    np.pi * c * t_final / L
)

error = np.abs(history[-1] - u_exact)

print("Maximum absolute error:", np.max(error))
print("Mean absolute error:", np.mean(error))






# Plot displacement at different time steps

plt.figure(figsize=(10, 6))

# Select 5 equally spaced time indices
indices = np.linspace(0, len(history) - 1, 5, dtype=int)

for i in indices:
    plt.plot(x, history[i], label=f"t = {times[i-1]:.2f} s")

plt.xlabel("Position along string (m)")
plt.ylabel("Displacement")
plt.title("1D Wave Equation using FDM")
plt.legend()
plt.grid(True)
plt.show()
