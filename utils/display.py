import math

from matplotlib import pyplot


def display_images(images):
    fig = pyplot.figure(figsize=(16, 16))
    rows = cols = int(math.ceil(len(images) ** 0.5))
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
        ax.axis("off")
    pyplot.show()


__all__ = [
    "display_images",
]
