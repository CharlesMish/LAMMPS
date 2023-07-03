import pandas as pd

# Open the file and read the lines
with open('tail_dump.pour', 'r') as f:
    lines = f.readlines()

# Extract the box dimensions from the lines
box_dims = [float(lines[i].split()[1]) for i in range(5, 8)]

# Print the box dimensions
print(f"Box dimensions: {box_dims}")

# Find the starting line of the particle data
start_line = lines.index("ITEM: ATOMS id type xs ys zs\n") + 1

# Read the particle data into a pandas DataFrame
df = pd.read_csv('tail_dump.pour', skiprows=range(start_line), sep='\s+', names=["id", "type", "xs", "ys", "zs"])

# Print the first few rows of the DataFrame
print(df.head())

# Scale the particle coordinates based on the box dimensions
df['x'] = df['xs'] * box_dims[0]
df['y'] = df['ys'] * box_dims[1]
df['z'] = df['zs'] * box_dims[2]

# Export the particle data to a new file
df.to_csv('scaled_particles.csv', index=False)
