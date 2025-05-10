import numpy as np
import matplotlib.pyplot as plt

# Data
data = np.array( [np.float32(4.391667), np.float32(2.3954546), np.float32(2.108), np.float32(2.2037036), np.float32(2.125), np.float32(2.0517242), np.float32(2.0517242), np.float32(2.0517242), np.float32(2.3678572), np.float32(2.21), np.float32(2.652), np.float32(2.071875), np.float32(2.21), np.float32(1.8942858), np.float32(2.2151515), np.float32(3.104348), np.float32(1.9833333), np.float32(3.2454545), np.float32(2.013158), np.float32(1.7)])

# Pembulatan ke 1 angka di belakang koma untuk kategori
rounded_data = np.round(data, 1)

# Hitung frekuensi masing-masing nilai
unique_vals, counts = np.unique(rounded_data, return_counts=True)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(unique_vals, counts, width=0.05, color='skyblue', edgecolor='black')
plt.xlabel("Nilai (dibulatkan 1 angka di belakang koma)")
plt.ylabel("Frekuensi")
plt.title("Distribusi Data (Toleransi 1 Angka di Belakang Koma)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(unique_vals, rotation=45)
plt.tight_layout()

plt.show()
