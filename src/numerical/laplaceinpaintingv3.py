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
object_region = np.zeros((H, W), dtype=bool)
object_region[30:70, 30:70] = True

for k in range(max_iter):

    error = 0.0

    for i in range(H):
        for j in range(W):

            if mask[i, j]:

                u_old = u[i, j]
                neighbors = []

                # Top
                if i > 0 and object_region[i-1, j] == object_region[i, j]:
                    neighbors.append(u[i-1, j])

                # Bottom
                if i < H-1 and object_region[i+1, j] == object_region[i, j]:
                    neighbors.append(u[i+1, j])

                # Left
                if j > 0 and object_region[i, j-1] == object_region[i, j]:
                    neighbors.append(u[i, j-1])

                # Right
                if j < W-1 and object_region[i, j+1] == object_region[i, j]:
                    neighbors.append(u[i, j+1])

                if neighbors:
                    u[i, j] = sum(neighbors) / len(neighbors)

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