import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from util.constants import *


def get_num_classes():
    class_names = load_class_names()
    info = separate_class(class_names)
    num_classes = len(class_names)
    num_years = len(info['year'].unique())
    num_makes = len(info['make'].unique())
    num_types = len(info['model_type'].unique())
    return num_classes, num_years, num_makes, num_types


def load_class_names(path='data/stanford_cars/devkit/class_names.csv'):
    cn = pd.read_csv(path, header=None).values.reshape(-1)
    cn = cn.tolist()
    return cn


def load_annotations(path, info):
    ann = pd.read_csv(path, header=None).values
    ret = {}
    year_codes = info['year'].astype('category').cat.codes
    make_codes = info['make'].astype('category').cat.codes
    type_codes = info['model_type'].astype('category').cat.codes

    for idx in range(len(ann)):
        x1, y1, x2, y2, target, imgfn = ann[idx]

        r = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'target': target - 1,
            'year_target': year_codes[target - 1].item(),
            'make_target': make_codes[target - 1].item(),
            'type_target': type_codes[target - 1].item(),
            'filename': imgfn
        }

        ret[idx] = r
    return ret


def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if make == 'Aston' and model.startswith('Martin'):
            make = 'Aston Martin'
            model = model.replace('Martin ', '')
        
        if 'Convertible' in model:
            model = model.replace('Convertible', '').replace('  ', '')

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'

        if model == 'Integra Type R':
            model_type = 'Type-R'

        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'

        if 'SRT' in model_type:
            model_type = 'SRT'

        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pd.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr


class StanfordCars(Dataset):
    def __init__(self, imgdir, anno_path, transform = None):
        self.class_names = load_class_names()
        self.info = separate_class(self.class_names)
        self.annos = load_annotations(anno_path, self.info)
        self.images = [os.path.join(imgdir, file) for file in os.listdir(imgdir)]
        self.transform = transform
        # self.resize = transforms.Resize(RESCALE_SIZE)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        r = self.annos[index]

        target = r['target']
        year_target = r['year_target']
        make_target = r['make_target']
        type_target = r['type_target']

        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, target, year_target, make_target, type_target


def prepare_loader() -> tuple[DataLoader, DataLoader]:

    trainset = StanfordCars(TRAIN_IMGDIR, TRAIN_ANNOPATH, TRAIN_TRANSFORM)
    testset = StanfordCars(TEST_IMGDIR, TEST_ANNOPATH, TEST_TRANSFORM)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, test_loader
