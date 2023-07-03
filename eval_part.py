import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('particle_data.csv')
print('Data loaded. Number of rows:', len(df))

# Define the particle types for CAM and SE particles
cam_types = list(range(1, 11))
se_types = list(range(11, 21))

# Create DataFrames for the CAM and SE particles
df_cam = df[df['type'].isin(cam_types)]
df_se = df[df['type'].isin(se_types)]
print('Data separated into CAM and SE particles.')
print('Number of CAM particles:', len(df_cam))
print('Number of SE particles:', len(df_se))

# Filter the SE DataFrame to include only particles where the z coordinate is less than or equal to the radius
df_se_boundary = df_se[df_se['z'] <= df_se['radius']]
print('Filtered SE particles on boundary.')
print('Number of SE particles on boundary:', len(df_se_boundary))

df_se_boundary.head()
from scipy.spatial import cKDTree

# Initialize a set to store the indices of the touching SE particles
touching_indices = set()

# Counter for the iteration
iteration = 0

total_se_particles = len(df_se)

while True:
    iteration += 1
    print(f"\nIteration: {iteration}")

    # Create a cKDTree for the touching particles (or the boundary particles in the first iteration)
    boundary_tree = cKDTree(
        df_se.iloc[list(touching_indices)][['x', 'y', 'z']] if touching_indices else df_se_boundary[['x', 'y', 'z']])

    # Create a cKDTree for all SE particles
    se_tree = cKDTree(df_se[['x', 'y', 'z']])

    # Initialize a set to store the indices of the newly found touching SE particles
    new_touching_indices = set()

    # Iterate over each boundary particle
    for i, row in df_se.iloc[list(touching_indices)].iterrows() if touching_indices else df_se_boundary.iterrows():
        # Find the SE particles within a distance equal to the sum of the boundary particle's radius and the maximum SE particle radius
        indices = se_tree.query_ball_point(row[['x', 'y', 'z']], row['radius'] + df_se['radius'].max())
        # Add the indices to the set
        new_touching_indices.update(indices)

    print(f"Number of newly connected particles in this iteration: {len(new_touching_indices - touching_indices)}")
    print(f"Total number of connected particles so far: {len(new_touching_indices)}")
    print(f"Total number of SE particles: {total_se_particles}")

    # If no new touching particles were found, break the loop
    if new_touching_indices == touching_indices:
        break

    # Update the set of touching particle indices
    touching_indices = new_touching_indices

# Get the DataFrame rows for the touching particles
df_se_touching = df_se.iloc[list(touching_indices)]

print('\nFinal results:')
print('Number of SE particles touching the boundary:', len(df_se_touching))

# Create a cKDTree for the connected SE particles
se_touching_tree = cKDTree(df_se_touching[['x', 'y', 'z']])

# Initialize a variable to store the volume of the connected CAM particles
connected_cam_volume = 0
df_cam['volume'] = 4/3 * np.pi * df_cam['radius'] ** 3
# Iterate over each CAM particle
for i, row in df_cam.iterrows():
    # Find the connected SE particles within a distance equal to the sum of the CAM particle's radius and the maximum connected SE particle radius
    indices = se_touching_tree.query_ball_point(row[['x', 'y', 'z']], row['radius'] + df_se_touching['radius'].max())
    # If the CAM particle is touching a connected SE particle, add its volume to the sum
    if indices:
        connected_cam_volume += row['volume']

# Calculate the total volume of all CAM particles
total_cam_volume = df_cam['volume'].sum()

print('Volume of connected CAM particles:', connected_cam_volume)
print('Total volume of CAM particles:', total_cam_volume)
print('Ratio of volumes:', connected_cam_volume / total_cam_volume)
