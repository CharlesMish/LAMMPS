import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler

# Load the initial and final particle data
df_initial = pd.read_csv('head_dump.pour').set_index('id')
df_final = pd.read_csv('particle_data.csv').set_index('id')

# Filter the final particle data where z < 40
df_final = df_final[df_final['z'] < 40]

# Create mappings from particle types to radii and densities
type_to_radius = {particle_type: radius for particle_type, radius in zip(range(1, 21), np.concatenate([np.linspace(2.85, 10.2, 10), np.linspace(0.3, 0.7, 10)]))}
type_to_density = {particle_type: density for particle_type, density in zip(range(1, 21), [4.74]*10 + [2.07]*10)}

# Add the radii and densities to the initial particle data
df_initial['radius'] = df_initial['type'].map(type_to_radius)
df_initial['density'] = df_initial['type'].map(type_to_density)

# Split the data into inputs (initial positions, radii, densities) and targets (final positions)
inputs = df_initial[['x', 'y', 'z', 'radius', 'density']].values
targets = df_final[['x', 'y', 'z']].values

# Normalize the input data
scaler = MinMaxScaler()
inputs_scaled = scaler.fit_transform(inputs)

# Convert the inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# Create a TensorDataset and DataLoader for the inputs and targets
dataset = TensorDataset(inputs_tensor, targets_tensor)
loader = DataLoader(dataset, batch_size=32)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3)
        self.mish = Mish()

    def forward(self, x):
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.mish(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the model
input_dim = 5
hidden_dim = 64
model = MLP(input_dim, hidden_dim)

# Define the optimizer and loss function
learning_rate = 0.003
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Train the model
num_epochs = 120
for epoch in range(num_epochs):
    for inputs_batch, targets_batch in loader:
        # Forward pass
        outputs_batch = model(inputs_batch)
        loss = loss_fn(outputs_batch, targets_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    # Early stopping
    early_stop_threshold = 10.0
    if loss.item() < early_stop_threshold:
        print(f'Loss is below the early stop threshold of {early_stop_threshold}. Stopping training.')
        break

# Load the new particle data
df_new = pd.read_csv('write.txt', comment='C', sep='\s+', header=None, skiprows=9, names=['id', 'type', 'radius', 'unknown', 'x', 'y', 'z'])
df_new['density'] = df_new['type'].map(type_to_density)

# Prepare the inputs
inputs_new = df_new[['x', 'y', 'z', 'radius', 'density']].values
inputs_new_scaled = scaler.transform(inputs_new)

# Convert the inputs to PyTorch tensor
inputs_new_tensor = torch.tensor(inputs_new_scaled, dtype=torch.float32)

# Predict the final positions
with torch.no_grad():
    outputs_new_tensor = model(inputs_new_tensor)

# Convert the predicted positions tensor to a numpy array
outputs_new = outputs_new_tensor.numpy()

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler for the coordinates (x, y, z)
scaler_coords = MinMaxScaler(feature_range=(0, 1))
scaler_coords.fit(inputs[:, :3])  # Fit the scaler on the coordinates (x, y, z)

# Normalize the inputs (coordinates, radius, density)
inputs_scaled = scaler.transform(inputs)

# Inverse transform the coordinates (x, y, z) only
outputs_new_coords_scaled = outputs_new[:, :3]
outputs_new_coords = scaler_coords.inverse_transform(outputs_new_coords_scaled)

# Update the coordinates in df_new with the predicted values
df_new[['x', 'y', 'z']] = outputs_new_coords

# Create a DataFrame for the predicted positions
df_predicted = pd.DataFrame(outputs_new, columns=['x', 'y', 'z'])

from sklearn.metrics import mean_squared_error

# The target values for minimum, average, and maximum z
target_min_z = 5   # target value, slightly above 0
target_max_z = 90  # target value, around 80-100
target_avg_z = 45  # target value, around 40-50

best_m = None
best_b = None
best_score = float('inf')

# Grid search over possible values of m and b
for m in np.linspace(0.5, 2.5, 100):
    for b in np.linspace(-2, 2, 100):
        # Apply the transformation
        z_new = (df_predicted['z'] + b) * m

        # Calculate the error terms
        error_min_z = (z_new.min() - target_min_z)**2
        error_max_z = (z_new.max() - target_max_z)**2
        error_avg_z = (z_new.mean() - target_avg_z)**2

        # Calculate the total error
        total_error = error_min_z + error_max_z + error_avg_z

        # If this is the best score so far, save the parameters
        if total_error < best_score:
            best_m = m
            best_b = b
            best_score = total_error

# Apply the best transformation
df_new['z'] = (df_predicted['z'] + best_b) * best_m

# Let's suppose you have 10 bins in the z direction
num_bins = 10

# Use pandas cut function to split z values into bins
bins = pd.cut(df_predicted['z'], num_bins, labels=False)

# Now, we know that each bin should ideally have the same number of values for a homogeneous distribution
# So, we can simply scale the binned values to the range we want
min_z = 5
max_z = 95
df_new['z'] = (bins / num_bins) * (max_z - min_z) + min_z


min_z = df_new['z'].min()
max_z = df_new['z'].max()
ave_z = df_new['z'].mean()
print("Best m:", best_m)
print("Best b:", best_b)
print("bins:", bins)
print("num bins:", num_bins)
print("Minimum value of 'z':", min_z)
print("Maximum value of 'z':", max_z)
print("Average value of 'z':", ave_z)

# Write the metadata and the updated particle data to the new file
with open('write2.txt', 'w') as f:
    # Write the metadata (the first 9 lines of the original file)
    with open('write.txt', 'r') as f_read:
        lines = f_read.readlines()
        f.writelines(lines[:9])
        f.write('\n')

    # Write the updated particle data (including the original x and y coordinates)
    df_new[['id', 'type', 'radius', 'unknown', 'x', 'y', 'z']].to_csv(f, sep=' ', index=False, header=False)
