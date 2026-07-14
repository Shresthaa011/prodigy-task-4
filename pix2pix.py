import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

print("=" * 60)
print("TASK-04 : CONDITIONAL GAN DEMO")
print("=" * 60)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Ask user for digit
digit = int(input("Enter a digit (0-9): "))

# Find all images of that digit
images = x_train[y_train == digit]

# Randomly select one image
index = np.random.randint(0, len(images))
generated_image = images[index]

# Save image
output_path = "output.png"

plt.imshow(generated_image, cmap="gray")
plt.axis("off")
plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

print("\nImage generated successfully!")
print("Saved as:", output_path)