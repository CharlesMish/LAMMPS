import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)  # RNN output and last hidden state
        out = self.fc(out)  # Pass the entire output sequence through the fully connected layer
        return out


# Map of type to radius
type_to_radius = {i: r for i, r in enumerate(np.concatenate((np.linspace(2.85, 10.2, 10), np.linspace(0.3, 0.7, 10))), start=1)}


def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    if 'particle_data.csv' in filename:
        # For the final file, filter particles with z < 40 and select the necessary columns
        data = data[data['z'] < 40]
        data = data[['id', 'x', 'y', 'z', 'radius']]
        print(f"Shape of {filename}: {data.shape}")
        print(f"Head of {filename}:\n{data.head()}")
        print(f"Number of unique IDs in {filename}: {data['id'].nunique()}")
    else:
        # For the initial and step files, add a radius column and select the necessary columns
        data['radius'] = data['type'].map(type_to_radius)
        data = data[['id', 'x', 'y', 'z', 'radius']]
        print(f"Shape of {filename}: {data.shape}")
        print(f"Head of {filename}:\n{data.head()}")
        print(f"Number of unique IDs in {filename}: {data['id'].nunique()}")

    # Reshape the data to have shape (num_particles, num_features)
    print(f"Shape after pivot: {data.shape}")
    data = data.to_numpy()  # Convert DataFrame to numpy array

    return data








num_particles = 47432
# Load the final data to get the particle IDs
df_final = pd.read_csv('particle_data.csv')
df_final = df_final[df_final['z'] < 40]
particle_ids = df_final['id'].tolist()

# Load the training data
train_filenames = [
    'filtered_head_dump.pour', 'filtered_step_11.txt', 'filtered_step_21.txt', 'filtered_step_31.txt',
    'filtered_step_41.txt', 'filtered_step_51.txt', 'filtered_step_101.txt', 'filtered_step_151.txt',
    'filtered_step_201.txt', 'filtered_step_251.txt', 'filtered_step_501.txt', 'particle_data.csv'
]

train_data = [load_and_preprocess_data(filename) for filename in train_filenames]


train_data_shapes = [data.shape for data in train_data]
print("Shapes of train_data:")
for filename, shape in zip(train_filenames, train_data_shapes):
    print(f"{filename}: {shape}")


