import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 10000
beta = 0.5  # Exponent for the waiting time distribution
alpha = 0.5  # Exponent for the MSD

# Generate waiting times from a power-law distribution
waiting_times = np.random.pareto(beta, num_steps)

# Generate step sizes from a normal distribution
step_sizes = np.random.normal(0, 1, num_steps)

# Initialize position and time arrays
positions = np.zeros(num_steps)
times = np.zeros(num_steps)

# Simulate the random walk
for i in range(1, num_steps):
    times[i] = times[i - 1] + waiting_times[i - 1]
    positions[i] = positions[i - 1] + step_sizes[i - 1]

# Calculate MSD
msd = np.mean(positions**2)

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(times, positions, label="Trajectory")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Sub-diffusive Random Walk")
plt.legend()
plt.show()

# Plot MSD
plt.figure(figsize=(10, 6))
plt.loglog(times, positions**2, label="MSD")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.title("Mean Squared Displacement")
plt.legend()
plt.show()
