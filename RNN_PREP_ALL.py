import pandas as pd
import glob

def get_bounds(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    bounds_line_index = next(i for i, line in enumerate(lines) if line.startswith('ITEM: BOX BOUNDS pp pp fm'))
    x_bound = float(lines[bounds_line_index + 1].split()[1])
    y_bound = float(lines[bounds_line_index + 2].split()[1])
    z_bound = float(lines[bounds_line_index + 3].split()[1])

    return x_bound, y_bound, z_bound

# Load the final particle data
df_final = pd.read_csv('particle_data.csv')

# Filter the particles with z < 40 and get the particle IDs
particle_ids = df_final[df_final['z'] < 40]['id'].values

# Load the initial particle data and filter it
df_initial = pd.read_csv('head_dump.pour')
df_initial = df_initial[df_initial['id'].isin(particle_ids)]
df_initial.to_csv('filtered_head_dump.pour', index=False)

# Get all the step files
step_files = glob.glob('step_*.txt')

# For each step file
for step_file in step_files:
    x_bound, y_bound, z_bound = get_bounds(step_file)

    # Load the step data
    df_step = pd.read_csv(step_file, comment='C', sep='\s+', header=None, skiprows=9, names=['id', 'type', 'xs', 'ys', 'zs'])

    # Scale the xs, ys, zs to x, y, z
    df_step['x'] = df_step['xs'] * x_bound
    df_step['y'] = df_step['ys'] * y_bound
    df_step['z'] = df_step['zs'] * z_bound
    df_step = df_step.drop(columns=['xs', 'ys', 'zs'])

    # Filter the step data
    df_step = df_step[df_step['id'].isin(particle_ids)]

    # Save the filtered step data
    filtered_step_file = 'filtered_' + step_file
    df_step.to_csv(filtered_step_file, index=False)
