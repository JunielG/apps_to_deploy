# Your age groups
age_groups = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

# Assuming you have a DataFrame with an 'age' column
# Sample DataFrame (replace this with your actual DataFrame)

# Function to assign age group
def assign_age_group(age):
    for group in age_groups:
        lower, upper = map(int, group.split('-'))
        if lower <= age <= upper:
            return group
    return 'Other'  # For ages outside defined groups

# Apply the function to create a new column
df['age_group'] = df['Age'].apply(assign_age_group)

# Display the result
df


import matplotlib.pyplot as plt
import numpy as np

# Create the figure
plt.figure(figsize=(16, 6))

# Create histogram with density=True to normalize
counts, bins, patches = plt.hist(df['age_group'], density=True)

# Scale the y-axis to show percentages (multiply by 100)
# Get the maximum count as a percentage
y_max = np.ceil(max(counts) * 100 * 1.1)  # Add 10% padding

# Set the y-axis to display percentages
plt.gca().set_ylim(0, y_max)
plt.gca().set_yticks(np.arange(0, 101, 10))  # 0% to 100% by 10%
plt.gca().set_yticklabels([f'{int(x)}%' for x in np.arange(0, 101, 10)])

# Add labels and title
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.title('Distribution by Age Group')

# Show the plot
plt.tight_layout()
plt.show()

