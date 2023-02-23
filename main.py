import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the image as a NumPy array
img = plt.imread('coins.png')

# Remove the alpha channel
img = img[:, :, :3]

# Display the image using Matplotlib:
plt.imshow(img)
plt.show()

# Create a binary mask of the coins using thresholding:
mask = np.zeros_like(img)
mask[img > 0] = 1

# Convert the mask to a 2-dimensional array by taking the mean along the color axis
mask_2d = np.mean(mask, axis=2)

# Plot the mask using Seaborn
sns.heatmap(mask_2d, cmap='gray', cbar=False)
plt.show()

# Replace the coin values with 255 using Pandas:
df = pd.DataFrame(np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2])))
df = df.where(df == 0, 255)
img = np.reshape(df.values, (img.shape[0], img.shape[1], img.shape[2]))

# Display the segmented image using Matplotlib:
plt.imshow(img)
plt.show()

# Loop completed
print("Funcionó!")