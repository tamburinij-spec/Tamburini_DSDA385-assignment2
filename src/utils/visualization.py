"""Visualization helpers for images and predictions."""

import matplotlib.pyplot as plt

def show_image(img, title=None):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
