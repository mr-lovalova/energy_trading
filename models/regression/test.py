import seaborn as sns
import matplotlib.pyplot as plt

# Data
solar_errors = [1.1, 0.36]
wind_errors = [6.3, 3.1]

# Combine data for plotting
data = solar_errors + wind_errors
labels = ["Solar ", "Solar", "Wind", "Wind "]

# Define custom colors for solar and wind
colors = sns.color_palette("husl", n_colors=2)

# Create a bar plot using Seaborn
ax = sns.barplot(x=labels, y=data, palette=colors)

# Customize plot
plt.ylabel("Prediction Errors (MWh)")
plt.title("Avg hourly prediction errors for municipality energy production")

# Add legend
legend_labels = ["Regression", "Deep learning"]
legend = plt.legend(legend_labels, loc="upper right")
for i in range(len(legend.legendHandles)):
    legend.legendHandles[i].set_color(colors[i])

# Save the plot
plt.savefig("prediction_errors_plot.png")

plt.show()
