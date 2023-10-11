# Import libraries
import torch, torchvision, os, numpy as np
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from glob import glob; from PIL import Image, ImageFile
from torchvision import transforms as T

# To load large image files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set the manual seed for reproductivity
torch.manual_seed(2023)

def get_dl(ds_name, tr_tfs, val_tfs, bs):
    
    """ 
    
    This function gets dataset name, transformations, and batch size and returns train, test dataloaders along with number of classes.
    
    Parameters:
    
        ds_name        - dataset name, str;
        tfs            - transformations, torchvision transforms object;
        bs             - batch size, int. 
        
    Outputs:
    
        trainloader    - train dataloader, torch dataloader object;
        testloader     - test dataloader, torch dataloader object;
        num_classes    - number of classes in the dataset, int.
        
    """
    
    # Assertions for the dataset name
    assert ds_name == "cifar10" or ds_name == "mnist", "Please choose one of these datasets: mnist, cifar10"
    
    # CIFAR10 dataset
    if ds_name == "cifar10":
        
        cls_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tr_tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 4)
        
        # Get testset
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = val_tfs)
        
        val_len = int(len(testset) * 0.5)
        val_set, test_set = random_split(testset, [val_len, len(testset) - val_len])
        
        # Initialize test dataloader
        
        val_dl =  DataLoader(val_set, batch_size = bs, shuffle = False, num_workers = 4)
        test_dl = DataLoader(test_set, batch_size = bs, shuffle = False, num_workers = 4)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    # MNIST dataset
    elif ds_name == "mnist":
        
        cls_names = [i for i in range(10)]
        
        # Get trainset
        trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = tr_tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True)
        
        # Get testset
        testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = val_tfs)
        
        val_len = int(len(testset) * 0.5)
        val_set, test_set = random_split(testset, [val_len, len(testset) - val_len])
        
        # Initialize test dataloader
        
        val_dl =  DataLoader(val_set, batch_size = bs, shuffle = False)
        test_dl = DataLoader(test_set, batch_size = bs, shuffle = False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
        
    print(f"{ds_name} is loaded successfully!")
    print(f"{ds_name} has {num_classes} classes!")
    
    return trainloader, val_dl, test_dl, cls_names, num_classes

class CustomDataset(Dataset):
    
    def __init__(self, root = "/mnt/data/dataset/bekhzod/im_class/korean_landmarks/kts/", tr_val = "train", transformations = None):
        super().__init__()
        
        # Get image paths
        self.im_paths = sorted(glob(f"{root}{tr_val}/*/*/images/*.jpg"))
        # Get the transformations to be applied
        self.transformations = transformations
        
        self.classes = {}
        cls_count = 0
        for idx, im_path in enumerate(self.im_paths):
            gt = self.get_label(im_path)
            if gt not in self.classes: self.classes[gt] = cls_count; cls_count += 1
        
    def __len__(self): return len(self.im_paths)

    def get_label(self, path): return path.split("/")[-3]

    def get_info(self): return {v: k for k, v in self.classes.items()}, len(self.classes)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path)
        gt = self.classes[self.get_label(im_path)]
        if self.transformations: im = self.transformations(im)
        
        return im, gt    
    
class CustomDataloader(nn.Module):
    
    """
    
    This class gets several parameters and returns train, validation, and test dataloaders.
    
    Parameters:
    
        root              - path to data with images, str;
        transformations   - transformations to be applied, torchvision transforms object;
        bs                - mini batch size of the dataloaders, int;
        im_files          - valid image extensions, list -> str;
        data_split        - data split information, list -> float.

    Outputs:

        dls               - train, validation, and test dataloaders, torch dataloader objects.
    
    """
    
    def __init__(self, transformations, bs, im_files = [".jpg", ".png", ".jpeg"], data_split = [0.9, 0.1]):
        super().__init__()
        
        # Assertion
        assert sum(data_split) == 1, "Data split elements' sum must be exactly 1"
        # assert ds_name in ["ghim", "cars", "default_sketch"], "Please choose either ghim or cars or sketch dataset"
        
        # Get the class arguments
        self.im_files, self.bs = im_files, bs
        
        # Get dataset from the root folder and apply image transformations
        self.tr_ds = CustomDataset(transformations = transformations, tr_val = "train")
        self.val_ds = CustomDataset(transformations = transformations, tr_val = "valid")
        self.test_ds = CustomDataset(transformations = transformations, tr_val = "test")
        # Get total number of images in the dataset
        self.total_ims = len(self.tr_ds)
        
        # Data split
        
        # Create datasets dictionary for later use and print datasets information
        self.all_ds = {"train": self.tr_ds, "validation": self.val_ds, "test": self.test_ds}
        for idx, (key, value) in enumerate(self.all_ds.items()): print(f"There are {len(value)} images in the {key} dataset.")
        
    # Function to get data length
    def __len__(self): return len(self.total_ims) + len(self.test_ds) + len(self.val_ds)

    def check_validity(self, path):
        
        """
        
        This function gets an image path and checks whether it is a valid image file or not.
        
        Parameter:
        
            path       - an image path, str.
            
        Output:
        
            is_valid   - whether the image in the input path is a valid image file or not, bool  
        
        """
        if os.path.splitext(path)[-1] in self.im_files: return True
        return False
    
    # Get dataloaders based on the dataset objects
    def get_dls(self): return [DataLoader(dataset = ds, batch_size = self.bs, shuffle = True, drop_last = True, num_workers = 8) for ds in self.all_ds.values()]
    
    # Get information on the dataset
    def get_info(self): return {v: k for k, v in self.ds.classes.items()}, len(self.ds.classes)
        
# tfs = T.Compose([T.Resize((224,224)), T.ToTensor()])
# ddl = CustomDataloader(transformations = tfs, bs = 64)
# tr_dl, val_dl, test_dl = ddl.get_dls()
# a, b = ddl.get_info()
# print(a, b)
