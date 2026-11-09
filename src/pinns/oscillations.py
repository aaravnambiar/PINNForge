import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm





# Physical parameters
m = 1.0      # mass
c = 4.0      # damping coefficient
k = 400.0    # spring constant


# Derived parameters
delta = c / (2 * m)                # damping ratio
omega_0 = np.sqrt(k / m)           # natural frequency
omega = np.sqrt(omega_0**2 - delta**2) # damped frequency


print(f"Physical parameters:")
print(f"  δ = {delta:.2f} (damping ratio)")
print(f"  ω₀ = {omega_0:.2f} (natural frequency)")
print(f"  ω = {omega:.2f} (damped frequency)")
print(f"  Underdamped: {delta < omega_0}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML


def oscillator(d, w0, x):
   """Analytical solution to the 1D underdamped harmonic oscillator problem."""
   assert d < w0
   w = np.sqrt(w0**2 - d**2)
   phi = np.arctan(-d / w)
   A = 1 / (2 * np.cos(phi))
   cos = np.cos(phi + w * x)
   exp = np.exp(-d * x)
   y = exp * 2 * A * cos
   return y



def create_spring_oscillator_animation_inline():
    d = 2  # damping coefficient
    w0 = 20  # natural frequency

    # Animation variables
    totalTime = 1.0  # Time domain [0, 1]
    dt = 0.0075
    t_array = np.arange(0, totalTime, dt)
    y_array = oscillator(d, w0, t_array)


    # Scaling factors
    scale = 1.0
    centerY = 0.0


    # Compute y_min and y_max from the displacement data
    y_min = np.min(y_array * scale + centerY)
    y_max = np.max(y_array * scale + centerY)


    # Create figure and axes
    fig, (ax_trace, ax_spring) = plt.subplots(1, 2, figsize=(12, 4))
    plt.tight_layout(pad=3.0)


    # Plot the displacement curve on ax_trace
    ax_trace.plot(t_array, y_array * scale + centerY, color='gray')
    trace_point, = ax_trace.plot([], [], 'bo', markersize=8)
    ax_trace.set_xlim(0, totalTime)
    ax_trace.set_ylim(-1.1, 1.1)
    ax_trace.set_xlabel('Time (s)')
    ax_trace.set_ylabel('Displacement')
    ax_trace.set_title('Displacement vs. Time')


    # Set up the mass-spring system on ax_spring
    ax_spring.set_xlim(-1, 1)
    ax_spring.set_ylim(-1.1, 1.1)
    ax_spring.axis('off')
    ax_spring.set_title('Mass-Spring System')


    # Draw the fixed block at equilibrium position (y=0)
    ax_spring.plot([-0.2, 0.2], [1.0, 1.0], 'k-', linewidth=4)


    # Initialize the mass and spring
    mass, = ax_spring.plot([], [], 'bo', markersize=20)
    spring_line, = ax_spring.plot([], [], 'k-', linewidth=1.5)


    def get_spring(y_start, y_end, coils=10, points_per_coil=15):
        length = y_end - y_start
        t = np.linspace(0, 1, coils * points_per_coil)
        x = 0.06 * np.sin(2 * np.pi * coils * t)
        y = y_start + length * t
        return x, y


    def update(frame):
        t = t_array[frame % len(t_array)]
        y = oscillator(d, w0, t) * scale + centerY


        # Update trace point
        trace_point.set_data([t], [y])


        # Update mass position
        mass.set_data([0], [y])


        # Update spring
        x_spring, y_spring = get_spring(1.0, y)  # Starting from y=1.0 (fixed point)
        spring_line.set_data(x_spring, y_spring)


        return trace_point, mass, spring_line


    ani = FuncAnimation(fig, update, frames=len(t_array), interval=20, blit=True)


    plt.close(fig)
    html_content = ani.to_jshtml()
    with open("animation.html", "w") as f:
        f.write(html_content)
        print("Animation saved to animation.html. Open it in your browser to view.")

if __name__ == "__main__":
    # Call the function to display the animation
    create_spring_oscillator_animation_inline()