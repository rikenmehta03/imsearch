from imsearch.backend.object_detector.yolo import Detector
from imsearch import utils
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def show_output(output, img):
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for obj in output:
        box = obj['box']
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        box_w = x2 - x1
        box_h = y2 - y1
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                 linewidth=2, edgecolor='red', facecolor="none")
        ax.add_patch(bbox)
        plt.text(
            x1,
            y1,
            s=obj['name'],
            color="white",
            verticalalignment="top",
            bbox={"color": 'red', "pad": 0},
        )

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()


if __name__ == "__main__":
    PATH = '../images/000000055167.jpg' 
    detector = Detector()

    img = utils.check_load_image(PATH)
    output = detector.predict(img)
    show_output(output, img)
