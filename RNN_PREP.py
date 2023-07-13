import os

def tail(lines, n):
    return lines[-n:]

def head(lines, n):
    return lines[:n]

def save_steps_to_files(source_file, num_lines, steps_to_capture, output_dir):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all lines in the source file
    with open(source_file, 'r') as f:
        all_lines = f.readlines()

    for step in steps_to_capture:
        # Calculate the number of lines to grab for this step
        lines_to_grab = step * num_lines

        # Grab the lines from the source file
        lines = tail(all_lines, lines_to_grab)

        # Grab the lines for this step
        step_lines = head(lines, num_lines)

        # Create a file for this step
        step_file = os.path.join(output_dir, f'step_{step}.txt')

        # Write the lines for this step to the file
        with open(step_file, 'w') as f:
            f.writelines(step_lines)

# Use the function
source_file = "dump.pour"
num_lines = 69557
steps_to_capture = [11, 21, 31, 41, 51, 101, 151, 201, 251, 501]
output_dir = "step_files"

save_steps_to_files(source_file, num_lines, steps_to_capture, output_dir)
