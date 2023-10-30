import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart_with_dynamic_range(values):
    # Determine the dynamic range from the values
    rmin = min(values)
    rmax = max(values)
    
    # Number of variables
    num_vars = len(values)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Make the plot repeat back to the start to close the circle
    values += values[:1]
    angles += angles[:1]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, color='blue', linewidth=2, label='Embedding')
    ax.fill(angles, values, color='blue', alpha=0.25)
    
    # Improve the appearance
    ax.set_theta_offset(np.pi / 2)  # Set the starting point to top
    ax.set_theta_direction(-1)  # Plot clockwise
    ax.set_rlabel_position(180 / num_vars)  # Position for radial labels
    ax.set_rlim([rmin, rmax])  # Set radial limits based on data range
    
    # Add a title
    ax.set_title("Radar Chart with Dynamic Range", size=16, color="blue", y=1.1)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

embedding_list = "add embeddings here to analyze"

# Plot the radar chart for the embedding values with dynamic range based on data values
plot_radar_chart_with_dynamic_range(embedding_list)