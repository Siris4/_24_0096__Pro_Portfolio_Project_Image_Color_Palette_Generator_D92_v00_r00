import numpy
# import Pillow
import matplotlib
import sklearn

from PIL import Image
import numpy as np

image = Image.open("1.jpg")  # use your own file path here.

# convert the image to RGB mode
image = image.convert("RGB")

# resize the image to speed up processing, if necessary
image = image.resize((200, 200))  # optional resizing to make it faster


image_array = np.array(image)
pixels = image_array.reshape((-1, 3))

from sklearn.cluster import KMeans


num_colors = 10
kmeans = KMeans(n_clusters=num_colors)

# fit the KMeans model to our pixel data
kmeans.fit(pixels)

# get the RGB values of the cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

hex_colors = [rgb_to_hex(color) for color in dominant_colors]

import matplotlib.pyplot as plt

# create a figure to display the colors
plt.figure(figsize=(10, 2))

# for each color, create a colored box and display the hex code
for i, color in enumerate(dominant_colors):
    plt.subplot(1, num_colors, i + 1)
    plt.axis("off")  # turn off axis

    # display color block
    plt.imshow([[color / 255]])  # divide by 255 to normalize RGB values for display

    # display hex code as the title
    plt.title(hex_colors[i], fontsize=10)

# show the figure
plt.tight_layout()
plt.show()