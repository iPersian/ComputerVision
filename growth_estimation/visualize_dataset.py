import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def select_images():
    # Open a file dialog for selecting 16 images
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select 16 Images",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if len(file_paths) != 16:
        print("Please select exactly 16 images.")
        return None
    return file_paths

def display_images(image_paths):
    # Create a 4x4 grid to display the images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(image_paths[i])  # Load the image
        ax.imshow(img)
        # ax.set_title(f"Image {i+1}")
        ax.axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

# Main script execution
image_paths = select_images()
if image_paths:
    display_images(image_paths)
