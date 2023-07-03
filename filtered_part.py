import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the scaled particles data
df = pd.read_csv('scaled_particles.csv')

# Define the radii for the first 10 types and the next 10 types
radii1 = np.linspace(2.85, 10.2, 10)
radii2 = np.linspace(0.3, 0.7, 10)

# Combine the two radii arrays into one
radii = np.concatenate([radii1, radii2])

# Create a mapping from particle types to radii
type_to_radius = {particle_type: radius for particle_type, radius in zip(range(1, 21), radii)}

# Add a new column "radius" based on the "type" column
df['radius'] = df['type'].map(type_to_radius)

# Define the series of heights
heights = range(1, 41)

# Create empty lists to store depth and porosity
depth = []
porosity = []

# Iterate over each height
for height in heights:
    # Filter the DataFrame to include only particles with z < height
    filtered_df = df[df['z'] < height].copy()  # Make a copy to avoid the SettingWithCopyWarning

    # Calculate the volume of each particle and create a new column "volume"
    filtered_df['volume'] = 4 / 3 * np.pi * filtered_df['radius'] ** 3

    # Calculate the total volume of all particles
    total_particle_volume = filtered_df['volume'].sum()

    # Calculate the volume of the box
    box_volume = filtered_df['x'].max() * filtered_df['y'].max() * height

    # Calculate the porosity
    current_porosity = 1 - (total_particle_volume / box_volume)

    # Append the current height and porosity to the lists
    depth.append(height)
    porosity.append(current_porosity)

# Plotting
plt.plot(depth, porosity, 'b-o')
plt.xlabel('Depth')
plt.ylabel('Porosity %')
plt.title('Depth vs. Porosity')
plt.xlim(1, max(depth))  # Set the x-axis limits from 15 onwards
plt.grid(True)

# Plot the constant value of 20.8
plt.axhline(y=.208, color='red', linestyle='--')

plt.show()
df.to_csv('particle_data.csv', index=False)