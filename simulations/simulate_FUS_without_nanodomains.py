import numpy as np
import pandas as pd
from rich.progress import track
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
box_size = 5.0  # microns
time_resolution = 0.1  # seconds (100 ms)
diffusion_coefficient = 0.8  # um^2/s
total_time = 10  # seconds
condensate_R_um = 2.0  # microns
condensate_center = np.array([box_size / 2, box_size / 2])  # Center of the box
num_initial_molecules = 5
mean_photobleaching_steps = 10
mean_new_molecules_per_frame = 5

# Calculate the step size, MSD = 2nDt
step_size = np.sqrt(4 * diffusion_coefficient * time_resolution)  # unit, um

# Output file path
output_folder = "/Users/GGM/Library/CloudStorage/Dropbox/UMich_PhD/Nils_Walter_Lab/Writing/MyPublications/SMT_FUS_Nanodomain-Nat2024/simulations"
output_file = "simulate_FUS_without_nanodomain.csv"
output_path = os.path.join(output_folder, output_file)

# Check if the condensate is smaller than the simulation box
if condensate_R_um > box_size / 2:
    raise ValueError(
        "Condensate radius is larger than half the box size. Please adjust the parameters."
    )

# list all time points
num_steps = int(total_time / time_resolution)
all_time = np.array([step * time_resolution for step in range(num_steps)])


# Function to generate random positions within condensate
def generate_random_positions_xy(num_molecules, condensate_center, condensate_R_um):
    positions = []
    for _ in range(num_molecules):
        while True:
            pos = np.random.uniform(
                condensate_center - condensate_R_um,
                condensate_center + condensate_R_um,
                2,
            )
            if np.linalg.norm(pos - condensate_center) <= condensate_R_um:
                positions.append(pos)
                break
    return positions[:, 0], positions[:, 1]


# Initialize the trackIDs, positions, photobleaching times of the molecules
track_ids = np.arange(num_initial_molecules)
arr_x, arr_y = generate_random_positions_xy(
    num_initial_molecules, condensate_center, condensate_R_um
)
photobleaching_times = np.random.exponential(
    mean_photobleaching_steps, num_initial_molecules
)

# Create a DataFrame to store the trajectory information
columns = ["trackID", "x_um", "y_um", "t_s", "box_size_um", "condensate_R_um"]
df = pd.DataFrame(columns=columns)
df = df.append(
    {
        "trackID": track_ids,
        "x_um": arr_x,
        "y_um": arr_y,
        "t_s": np.repeat(0, num_initial_molecules),
        "box_size_um": np.repeat(box_size, num_initial_molecules),
        "condensate_R_um": np.repeat(condensate_R_um, num_initial_molecules),
    },
    ignore_index=True,
)

# Initialize the next available trackID
next_track_id = num_initial_molecules

# Simulate the diffusion process
for t in track(all_time, description="Simulating..."):
    # get all molecules from the previous update
    df_previous = df[df["t_s"] == t]
    # update all pre-existing molecules one by one
    for j in range(positions.shape[0]):
        # Generate random steps
        step = np.random.normal(0, step_size, 2)
        new_position = positions[j] + step

        # Check if the molecule is within the condensate
        in_condensate = (
            np.linalg.norm(new_position - condensate_center) <= condensate_R_um
        )

        # Append the trajectory information to the lists
        x_positions.append(new_position[0])
        y_positions.append(new_position[1])
        times.append(i * time_resolution)
        in_condensates.append(in_condensate)
        condensate_radii.append(condensate_R_um)

        if not in_condensate or i >= photobleaching_times[j]:
            active_mask[j] = False
        else:
            positions[j] = new_position

    # Remove molecules that have left the condensate or photobleached
    positions = positions[active_mask]
    track_ids = track_ids[active_mask]
    photobleaching_times = photobleaching_times[active_mask]

    # Generate new molecules joining the system
    num_new_molecules = np.random.poisson(mean_new_molecules_per_frame)

    if num_new_molecules > 0:
        # add new molecule trackIDs
        new_track_ids = np.arange(next_track_id, next_track_id + num_new_molecules)
        next_track_id += num_new_molecules
        track_ids = np.concatenate((track_ids, new_track_ids))

        # add new molecule positions
        new_positions = generate_random_positions(
            num_new_molecules, condensate_center, condensate_R_um
        )
        positions = np.vstack((positions, new_positions))

        # add photobleaching times for new molecules
        new_photobleaching_times = np.random.exponential(
            mean_photobleaching_steps, num_new_molecules
        )
        photobleaching_times = np.concatenate(
            (photobleaching_times, new_photobleaching_times)
        )

# Create a DataFrame from the lists
df = pd.DataFrame(
    {
        "trackID": track_ids,
        "x": x_positions,
        "y": y_positions,
        "t": times,
        "InCondensate": in_condensates,
        "CondensateR": condensate_radii,
    }
)

# Sort the DataFrame by trackID and then by t
df = df.sort_values(["trackID", "t"])

# Save the DataFrame as a CSV file
df.to_csv(output_path, index=False)

# Plot the trajectories
plt.figure(figsize=(8, 8))

# Plot the condensate as a more transparent blue circle without border
condensate = plt.Circle(
    condensate_center, condensate_R_um, facecolor="blue", alpha=0.1, edgecolor="none"
)
plt.gca().add_patch(condensate)

# Color the trajectories by time (t)
colors = cm.viridis(df["t"] / total_time)

# Plot the trajectories
for _, trajectory in df.groupby("trackID"):
    plt.plot(
        trajectory["x"],
        trajectory["y"],
        "-o",
        color=colors[trajectory.index[0]],
        markersize=4,
    )

plt.xlim(0, box_size)
plt.ylim(0, box_size)
plt.xlabel("x, $\mu$m")
plt.ylabel("y, $\mu$m")
plt.axis("scaled")
plt.grid(True)

# Save the plot as a PNG image
plot_output_path = os.path.join(output_folder, "simulate_FUS_without_nanodomain.png")
plt.savefig(plot_output_path, dpi=300)

plt.show()

# Calculate the number of existing molecules for each time step
num_existing_molecules = df.groupby("t").size().reset_index(name="num_molecules")

# Plot the number of existing molecules over time
plt.figure(figsize=(8, 6))
plt.plot(num_existing_molecules["t"], num_existing_molecules["num_molecules"])
plt.xlabel("Time, s")
plt.ylabel("Number of Existing Molecules")
plt.title("Number of Existing Molecules over Time")
plt.grid(True)

# Save the plot as a PNG image
plot_output_path_Nt = os.path.join(
    output_folder, "simulate_FUS_without_nanodomain_N-t.png"
)
plt.savefig(plot_output_path_Nt, dpi=300)

plt.show()

# Plot the step size distribution
plt.figure(figsize=(8, 6))
plt.hist(step_sizes, bins=50, density=True, alpha=0.7)
plt.axvline(np.mean(step_sizes), color="red", linestyle="--", label="Mean")
plt.xlabel("Step Size, $\mu$m")
plt.ylabel("Probability Density")
plt.title("Step Size Distribution")
plt.legend()
plt.grid(True)

# Save the plot as a PNG image
plot_output_path_steps = os.path.join(
    output_folder, "simulate_FUS_without_nanodomain_step_sizes.png"
)
plt.savefig(plot_output_path_steps, dpi=300)

plt.show()
