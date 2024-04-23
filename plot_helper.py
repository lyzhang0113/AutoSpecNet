import torchvision
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from util.dataset import load_class_names, separate_class

def get_all_classes():
    class_names = load_class_names()
    info = separate_class(class_names)
    year_codes = ic(info['year'].astype('category').cat.codes)
    make_codes = ic(info['make'].astype('category').cat.codes)
    type_codes = ic(info['model_type'].astype('category').cat.codes)

    year = {}
    make = {}
    type = {}

    for i in range(ic(len(class_names))):
        year[year_codes[i]] = info['year'][i]
        make[make_codes[i]] = info['make'][i]
        type[type_codes[i]] = info['model_type'][i]

    return year, make, type


def imshow(example, classes, filename=None):
    img = torchvision.utils.make_grid(example[0])
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.cpu().numpy()
    plt.title(f'Predicted: {classes[example[1]]}, True: {classes[example[2]]}')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename) if filename else plt.show()


if __name__ == "__main__":
    get_all_classes()
