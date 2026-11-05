import numpy as np
import matplotlib.pyplot as plt

nx = 100
ny = 100

image = np.zeros((ny, nx))

# Create a white square
image[30:70, 30:70] = 255
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.colorbar(label="Pixel Intensity")
plt.show()


mask = np.zeros((ny, nx), dtype=bool)

# Missing region
mask[40:60, 55:75] = True

damaged = image.copy()
damaged[mask] = 0
plt.imshow(damaged, cmap='gray', vmin=0, vmax=255)
plt.title("Damaged Image")
plt.colorbar(label="Pixel Intensity")
plt.show()


u = damaged.copy()

max_iter = 5000
tol = 1e-4

errors = []
for k in range(max_iter):
    u_new = u.copy()
    u_new[1:-1, 1:-1] = 0.25 * (
    u[:-2, 1:-1] +
    u[2:, 1:-1] +
    u[1:-1, :-2] +
    u[1:-1, 2:]
    )

    u_new[~mask] = damaged[~mask]
    error = np.max(np.abs(u_new[mask] - u[mask]))
    u = u_new
    if error < tol:
        print(f"Converged after {k} iterations.")
        break
    errors.append(error)
    


plt.imshow(u, cmap='gray', vmin=0, vmax=255)
plt.title("Inpainted Image")    
plt.colorbar(label="Pixel Intensity")
plt.show()