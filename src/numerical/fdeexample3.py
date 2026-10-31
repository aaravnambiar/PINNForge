import matplotlib.pyplot as plt
import numpy as np

# Physical parameters
L = 1.0
v=0.01/np.pi
t_final = 1.0


# Points
Nx = 201
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]

Nt = 5000

dt = t_final / Nt
u =  -np.sin(np.pi * x / L)
u[0] = 0.0
u[-1] = 0.0

r = v * dt / dx**2
C = dt / dx  # assuming max |u| = 1 initially

history = [u.copy()]
times = [0.0]

# Time stepping
for n in range(Nt):
    u_new = u.copy()

    u_new[1:-1] =(  u[1:-1] 
                    + r * (u[2:] - 2*u[1:-1] + u[:-2])
                    - C/2 * u[1:-1] * (u[2:] - u[:-2]))
    u_new[0] = 0.0
    u_new[-1] = 0.0
    history.append(u_new.copy())
    times.append((n + 1) * dt)
    u = u_new

plt.figure(figsize=(10, 6))

plot_times = [0.0, 0.25, 0.5, 0.75, 1.0]

for t_val in plot_times:
    idx = round(t_val / dt)
    plt.plot(x, history[idx], label=f"t = {t_val:.2f}")

plt.xlabel("Position (x)")
plt.ylabel("Velocity u(x,t)")
plt.title("1D Viscous Burgers' Equation — FDM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
    




