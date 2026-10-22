
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
L = 1.0
alpha = 0.01
t_final = 2.0

# Points
Nx = 51
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Time points
r = 0.4
dt = r * dx**2 / alpha
Nt = int(np.ceil(t_final / dt))
dt = t_final / Nt
r = alpha * dt / dx**2

# Initial condition boundary
T = 100 * np.sin(np.pi * x)
T[0] = T[-1] = 0.0

#history so that we can copy/place same time instant values
history = [T.copy()]
times = [0.0]

# Time stepping
for n in range(Nt):
    T_new = T.copy()
    # solving the differential equation
    T_new[1:-1] = T[1:-1] + r * (
        T[2:] - 2*T[1:-1] + T[:-2]
    )

    T_new[0] = 0.0
    T_new[-1] = 0.0

    T = T_new
    history.append(T.copy())
    times.append((n + 1) * dt)

history = np.array(history)
times = np.array(times)

# Analytical solution
T_exact = 100 * np.exp(
    -alpha * np.pi**2 * t_final
) * np.sin(np.pi * x)

#plotting the FDM vs actual answer
plt.plot(x, history[0], label="Initial")
plt.plot(x, history[-1], label="FDM")
plt.plot(x, T_exact, "--", label="Exact")

plt.xlabel("Position (m)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

rmse = np.sqrt(np.mean((history[-1] - T_exact)**2))
print(f"RMSE: {rmse:.6f}")
