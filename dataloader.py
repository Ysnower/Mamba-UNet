from PIL import Image
from torch.utils.data import Dataset
from utils import *
import os
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip, ColorJitter


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


class MydataLoader(Dataset):
    def __init__(self, dataRoot, loadSize, training=True, maskDir = "masks"):
        super(MydataLoader, self).__init__()
        self.imageFiles = [os.path.join(dataRootK, files) for dataRootK, dn, filenames in os.walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.imgTrans = Compose([ToTensor(),
                                 RandomCrop(loadSize, pad_if_needed=True, fill=1, padding_mode="constant"),
                                 RandomHorizontalFlip(p=0.5),
                                 RandomRotation(degrees=10, fill=1),
                                 RandomVerticalFlip(p=0.5)])
        self.maskTrans = Compose([ToTensor(),
                                 RandomCrop(loadSize, pad_if_needed=True, fill=0, padding_mode="constant"),
                                 RandomHorizontalFlip(p=0.5),
                                 RandomRotation(degrees=10, fill=0),
                                 RandomVerticalFlip(p=0.5)])
        self.colorjitter = Compose([ColorJitter(brightness=0.2, contrast=0.2)])

        self.valTrans = Compose([ToTensor(), RandomCrop(loadSize)])
        self.training = training
        self.maskDir = maskDir

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.imageFiles[index].replace("images", self.maskDir).replace("jpg", "png"))
        if self.training:

            # dataInput = [img, mask]
            # dataInput = RandomHorizontalFlip(dataInput)
            # dataInput = RandomRotate(dataInput)
            # img = dataInput[0]
            # mask = dataInput[1]
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            inputImage = self.imgTrans(img)
            inputImage = self.colorjitter(inputImage)
            torch.random.manual_seed(seed)
            mask = self.maskTrans(mask.convert("L"))
        else:
            # mask = Image.open(self.imageFiles[index].replace("images", self.maskDir))
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            inputImage = self.imgTrans(img)
            torch.random.manual_seed(seed)
            mask = self.maskTrans(mask.convert("L"))
        return inputImage, mask

    def __len__(self):
        return len(self.imageFiles)

