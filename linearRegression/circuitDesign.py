import matplotlib.pyplot as plt

# Sample data (replace with your actual experimental values)
time_data = [75, 150, 234, 302, 412]  # Time in seconds
rate_data = [71, 72, 73, 74, 75]  # Charging rate (e.g., dV/dt)

# Plotting the graph
plt.figure(figsize=(8, 5))
plt.plot(time_data, rate_data, marker='o', linestyle='-', color='b', label="Charging Rate")

# Labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Charging Rate (%)")
plt.title("CHARGING RATE vs TIME (Current State)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
