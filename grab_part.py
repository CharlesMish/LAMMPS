# Define the source file and the number of lines to be read
source_file = "dump.pour"
num_lines = 69557

# Create a destination file name
dest_file = f"tail_{source_file}"

# Check the total number of lines in the source file
with open(source_file, 'r') as f:
    total_lines = sum(1 for _ in f)

# Calculate the starting line for reading
start_line = max(total_lines - num_lines, 0)

# Read the last num_lines from the source file and write to the destination file
with open(source_file, 'r') as f, open(dest_file, 'w') as out_file:
    for i, line in enumerate(f):
        if i >= start_line:
            out_file.write(line)