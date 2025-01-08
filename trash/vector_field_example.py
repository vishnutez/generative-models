import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Define the vector field components (example: rotational field)
U = -Y
V = X

# Calculate the magnitude of the vectors
magnitude = np.sqrt(U**2 + V**2)

# Normalize the vectors to constant length
U_normalized = U / magnitude
V_normalized = V / magnitude

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.pcolormesh(X, Y, magnitude, shading='auto', cmap='viridis')
quiver = ax.quiver(X, Y, U_normalized, V_normalized, color='white', scale=50)

# Add a colorbar for the magnitude
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label('Magnitude')

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Vector Field with Magnitude as Heatmap and Angle by Arrows')

# Show the plot
plt.show()
