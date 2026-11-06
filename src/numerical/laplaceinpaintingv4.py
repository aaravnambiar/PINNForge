import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = r"E:\pinnforge\PINNForge\image.png"

img = Image.open(img_path).convert("L")
img = np.array(img, dtype=np.float64)

h, w = img.shape
u_original = img/255.0

gx = np.zeros_like(u_original)
gy = np.zeros_like(u_original)

gx[:, 1:-1] = (u_original[:, 2:] - u_original[:, :-2]) / 2
gy[1:-1, :] = (u_original[2:, :] - u_original[:-2, :]) / 2

grad_mag = np.sqrt(gx**2 + gy**2)

plt.imshow(u_original, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.show()

epsilon = 1e-8
grad_mag = grad_mag + epsilon

probs = grad_mag / np.sum(grad_mag)

total_pixels = u_original.size
keep_ratio = 0.10

keep_count = int(keep_ratio * total_pixels)

flat_indices = np.arange(total_pixels)
chosen = np.random.choice(
    flat_indices,
    size=keep_count,
    replace=False,
    p=probs.ravel()
)

known = np.zeros(total_pixels, dtype=bool)

known[chosen] = True

known = known.reshape(u_original.shape)

mask = ~known

damaged = u_original.copy()
damaged[mask] = 0.0

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(known, cmap="gray")
plt.title("Selected Pixels (10%)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(damaged, cmap="gray", vmin=0, vmax=1)
plt.title("Image with 90% Missing Data")
plt.axis("off")

plt.show()



u = damaged.copy()

max_iter = 5000
tol = 1e-5
errors = []

for k in range(max_iter):

    padded = np.pad(u, 1, mode="reflect")

    u_new = 0.25 * (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:]
    )

    u_new[~mask] = damaged[~mask]

    error = np.max(np.abs(u_new[mask] - u[mask]))
    errors.append(error)

    u = u_new

    if error < tol:
        print(f"Converged after {k+1} iterations")
        break

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(u_original, cmap="gray")
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(u, cmap="gray")
plt.title("Inpainted Image")

plt.tight_layout()
plt.show()

