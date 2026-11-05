import numpy as np
import matplotlib.pyplot as plt

nx = 100
ny = 100

image = np.zeros((ny, nx))

# Create a white square
image[30:70, 30:70] = 255


mask = np.zeros((ny, nx), dtype=bool)

# Missing region
mask[40:60, 55:75] = True

damaged = image.copy()
damaged[mask] = 0


u = damaged.copy()

max_iter = 5000
tol = 1e-4
H, W = u.shape
errors = []
for k in range(max_iter):
    
    error = 0.0

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if mask[i, j]:
                u_old = u[i,j]
                u[i, j] = 0.25 * (
                    u[i-1, j] +
                    u[i+1, j] +
                    u[i, j-1] +
                    u[i, j+1]
                )
                change = abs(u[i, j] - u_old)
                error = max(error, change)
            

   
    errors.append(error)
    if error < tol:
        print(f"Converged after {k+1} iterations")
        break



plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(u, cmap="gray")
plt.title("Gauss-Seidel")

plt.tight_layout()
plt.show()

plt.plot(errors)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Maximum pixel change")
plt.title("Gauss-Seidel Convergence")
plt.grid(True)
plt.show()