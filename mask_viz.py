import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_height_from_mask(mask_xy, x_tolerance=2.0):
    if not mask_xy:
        return None, None

    mask_xy = np.array(mask_xy)
    max_height = 0
    max_pair = None

    for i, (x1, y1) in enumerate(mask_xy):
        for j in range(i + 1, len(mask_xy)):
            x2, y2 = mask_xy[j]
            if abs(x1 - x2) <= x_tolerance:
                height = abs(y1 - y2)
                if height > max_height:
                    max_height = height
                    max_pair = ((x1, y1), (x2, y2))

    return max_height, max_pair

# Mask ketiga (seperti yang kamu kirim sebelumnya)
mask_3 =[
    [415, 235],
    [415, 340],
    [430, 340],
    [432.5, 342.5],
    [477.5, 342.5],
    [480, 340],
    [487.5, 340],
    [492.5, 335],
    [502.5, 335],
    [502.5, 305],
    [495, 305],
    [492.5, 302.5],
    [492.5, 287.5],
    [490, 285],
    [490, 265],
    [487.5, 262.5],
    [487.5, 255],
    [485, 252.5],
    [485, 247.5],
    [482.5, 245],
    [482.5, 235]
]

interpolated = [(450, y) for y in range(225, 421, 5)]

extended_mask = mask_3 + interpolated

# Hitung tinggi dan pasangan titik
height, point_pair = compute_height_from_mask(mask_3)

# Visualisasikan hasilnya
mask_array = np.array(mask_3)

plt.figure(figsize=(8, 6))
plt.fill(mask_array[:, 0], mask_array[:, 1], color='lightblue', edgecolor='blue', linewidth=2, label='Mask')
if point_pair:
    p1, p2 = point_pair
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=3, label=f'Max Height = {height:.2f}')
    plt.scatter(*zip(*[p1, p2]), color='red')
plt.gca().invert_yaxis()
plt.title("Visualisasi Mask & Ketinggian Maksimum")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
