import torch
from torch.optim import SGD, Adam
import torchvision.transforms.v2 as transforms
from torchvision.models import resnext50_32x4d, resnet34, mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

MODEL_PATH = 'model.pth'

RESCALE_SIZE = 400
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
NUM_WORKERS = 4
BATCH_SIZE = 32
NUM_EPOCHS = 40

YEAR_LOSS = 0.1
MAKE_LOSS = 0.1
TYPE_LOSS = 0.1

# resnext50 / resnet34 / mobilenet_v2
BASE_MODEL = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Optimizer
OPTIM = SGD
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

TRAIN_IMGDIR = 'data/stanford_cars/cars_train'
TRAIN_ANNOPATH = 'data/stanford_cars/devkit/cars_train_annos.csv'

TEST_IMGDIR = 'data/stanford_cars/cars_test'
TEST_ANNOPATH = 'data/stanford_cars/devkit/cars_test_annos_withlabels.csv'

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomIoUCrop(), # https://arxiv.org/abs/1512.02325
    transforms.TrivialAugmentWide(), # https://arxiv.org/abs/2103.10158
    transforms.ToImage(),  # Converts PIL Images or NumPy arrays to torch.float32 tensors
    transforms.ToDtype(torch.float32, scale=True), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomIoUCrop(), # https://arxiv.org/abs/1512.02325
    transforms.TrivialAugmentWide(), # https://arxiv.org/abs/2103.10158
    transforms.ToImage(),  # Converts PIL Images or NumPy arrays to torch.float32 tensors
    transforms.ToDtype(torch.float32, scale=True), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])