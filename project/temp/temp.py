import matplotlib.pyplot as plt

# # Define the points for the straight line
# x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# # Masking the point at index 5
# masked_index = 5
# masked_x = x1[masked_index]
# masked_y = y1[masked_index]

# # Plotting the line
# plt.plot(x1, y1, marker='o', color='blue')

# # Plotting the masked point
# plt.scatter(masked_x, masked_y, color='red', zorder=5)  # zorder ensures the masked point appears on top

# # Adding labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.xlim(0,25)
# plt.ylim(0,25)
# plt.grid(True)

# # Saving the figure
# plt.savefig('fig2.png')

# # Display the plot
# plt.show()
# Define the points for the straight line
x2 = [1, 2, 3, 4, 5]
y2 = [1, 3, 5, 7, 9]

# Plotting the line up to index 3
plt.plot(x2[:4], y2[:4], color='green')

# Plotting the dashed line for prediction
plt.plot(x2[3:], y2[3:], '--', color='orange')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(0,10)
plt.ylim(0,10)
plt.grid(True)

# Saving the figure
plt.savefig('fig3.png')

# Display the plot
plt.show()
