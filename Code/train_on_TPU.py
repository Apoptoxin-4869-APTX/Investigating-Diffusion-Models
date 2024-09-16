#special learning
#t = invNormal(1, self.T, mu=self.T/2, sd=self.T/5, size=len(x0), bottleneck=flags["bottleneck"])
#t = torch.tensor(t, device=self.device, dtype=torch.long)
import subprocess
command = "curl -L -o unet.py https://raw.githubusercontent.com/MaximeVandegar/Papers-in-100-Lines-of-Code/main/Denoising_Diffusion_Probabilistic_Models/unet.py"
subprocess.run(command, shell=True, check=True)
command = "pip install -q gdown"
subprocess.run(command, shell=True, check=True)
command = "pip install --upgrade -q tensorflow-datasets"
subprocess.run(command, shell=True, check=True)
command = "pip install -q kaggle"
subprocess.run(command, shell=True, check=True)
command = "pip install --upgrade -q tensorflow"
subprocess.run(command, shell=True, check=True)



from unet import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import time
import os
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow as tf
#import torchvision
from torch.utils.tensorboard import SummaryWriter



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')] # Adjust file extensions as needed

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def data(dataset_name):
    match dataset_name:
        case "MNIST":
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.MNIST(root='./data', 
                                           train=True, 
                                           download=True, 
                                           transform=transform)
            
            test_dataset = datasets.MNIST(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)
            
            return train_dataset, test_dataset

        
        case "FashionMNIST":
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.FashionMNIST(root='./data', 
                                           train=True, 
                                           download=True, 
                                           transform=transform)
            
            test_dataset = datasets.FashionMNIST(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)
            
            return train_dataset, test_dataset

        
        case "CIFAR10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            train_dataset = datasets.CIFAR10(root='./data', 
                                           train=True, 
                                           download=True, 
                                           transform=transform)
            
            test_dataset = datasets.CIFAR10(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)
            
            return train_dataset, test_dataset
        
        case "SVHN":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            train_dataset = datasets.SVHN(root='./data',
                                          split='train',
                                          download=True,
                                          transform=transform)
            
            test_dataset = datasets.SVHN(root='./data',
                                         split='test',
                                         download=True,
                                         transform=transform)
            
            return train_dataset, test_dataset

        
        case "STL10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            train_dataset = datasets.STL10(root='./data',
                                          split='train+unlabeled',
                                          download=True,
                                          transform=transform)
            
            test_dataset = datasets.STL10(root='./data',
                                         split='test',
                                         download=True,
                                         transform=transform)
            
            return train_dataset, test_dataset        
        
        
        case "CelebA-HQ":
            #command = "kaggle datasets download -d badasstechie/celebahq-resized-256x256 | unzip celebahq-resized-256x256.zip | rm celebahq-resized-256x256.zip"
            #subprocess.run(command, shell=True, check=True)
            
            transform = transforms.Compose([transforms.Resize((64, 64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                           ])
    
            train_dataset = CustomDataset(root_dir = 'celeba_hq_256', transform=transform)
            
            test_dataset = {}
            
            return train_dataset, test_dataset        

        
        case _:
            print("Not valid dataset name")


def invNormal(low, high, mu=0, sd=1, *, size=1, block_size=1024, bottleneck = 1):
    remain = size
    result = []
    
    mul = -0.5 * sd**-2

    while remain:
        # draw next block of uniform variates within interval
        x = np.random.uniform(low, high, size=min((remain+5)*2, block_size))
        x = np.round(x).astype(int)
        # reject proportional to normal density
        x = x[(1-bottleneck)*np.exp(mul*(x-mu)**2) < np.random.rand(*x.shape)]

        
        # make sure we don't add too much
        if remain < len(x):
            x = x[:remain]

        result.append(x)
        remain -= len(x)

    return np.concatenate(result)


class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    
    def training(self, train_loader, optimizer, epoch, flags):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """
        para_loader = pl.ParallelLoader(train_loader, [self.device]).per_device_loader(self.device)

        # Create a tqdm progress bar
        if xm.is_master_ordinal():  # Only the master process creates the progress bar
            pbar = tqdm(
                enumerate(para_loader), 
                total=len(train_loader), 
                desc=f"Epoch {epoch}",
                bar_format="{l_bar}{bar:10}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                unit="batch"
            )
        else:
            pbar = enumerate(train_loader)  # Non-master processes just iterate


        # tracking losses
        losses = []

        for batch_idx, (data, target) in pbar:
            x0 = data

            #uniform learning
            #t = torch.randint(1, self.T + 1, (len(x0),), device=self.device, dtype=torch.long)

            #special learning
            t = invNormal(1, self.T, mu=flags["mu"], sd=flags["sd"], size=len(x0), bottleneck=flags["bottleneck"])
            t = torch.tensor(t, device=self.device, dtype=torch.long)

            eps = torch.randn_like(x0)
    
            # Take one gradient descent step
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                
    
            eps_predicted = self.function_approximator(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t - 1)

            
            loss = nn.functional.mse_loss(eps, eps_predicted)
            optimizer.zero_grad()
            loss.backward()
    
            xm.optimizer_step(optimizer)  #optimizer.step()

            losses.append(loss.item())

            flags["writer"].add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

            if xm.is_master_ordinal():
                loss_tensor = torch.tensor([loss.item()], device=self.device)
                #xm.all_reduce(loss_tensor, xm.REDUCE_SUM)  # Gather loss from all processes
                avg_loss = loss_tensor.item() / xm.xrt_world_size()  # Calculate average
                pbar.set_postfix(loss=f"{avg_loss:.5f}")

        avg = np.mean(losses)
        std = np.std(losses)

        flags["writer"].add_scalar('Average Loss/epoch', avg, epoch)
        flags["writer"].add_scalar('Standard Deviation Loss/epoch', std, epoch)

    
    def sampling(self, n_samples=1, image_channels=3, img_size=(32, 32), use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """
    
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]), device=self.device)

        for t in tqdm(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t
    
            beta_t = self.beta[t - 1].view(-1, 1, 1, 1)  
            alpha_t = self.alpha[t - 1].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t - 1].view(-1, 1, 1, 1) 
    
            # Predicted noise (from your model)
            predicted_noise = self.function_approximator(x, t - 1)
    
            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
    
        return x

def train_model(index, flags):
    #torch.manual_seed(flags["seed"])


    # Managing the directory
    directory_path = flags["path"]  # Replace with the actual path

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully!")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Initializing the model
    device = xm.xla_device()
    model = UNet(ch=128, in_ch=3).to(device)
    diffusion_model = DiffusionModel(flags["depth"], model, device)
    
    # Data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        flags["train_dataset"], #train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )
    
    train_loader = DataLoader(
        flags["train_dataset"],
        batch_size=flags["batch_size"],
        sampler=train_sampler,
        num_workers=flags["num_workers"],
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=flags["lr"])

    # loss function
    loss_fn = "custom"

    # Train LOOP + tqdm for additional info
    for epoch in range(1, flags["num_epochs"] + 1):
        diffusion_model.function_approximator.train()
        diffusion_model.training(train_loader, optimizer, epoch, flags)
        
        if epoch % 50 == 0:
            a = time.time()
            tmp = flags["path"]
            xm.save(diffusion_model.function_approximator.state_dict(), f"{tmp}epoch_{epoch:05d}.pth")
            b = time.time()
            print(f"Saving completed in {(b-a):.3f} seconds")

# Best usage b_s = 1500 on MNIST
# Best usage b_s = 1250 on CIFAR10

if __name__ == "__main__":
    
    FLAGS = {}
    FLAGS["seed"] = 4869
    FLAGS["batch_size"] = 1250
    FLAGS["num_epochs"] = 10
    FLAGS["num_workers"] = 16  # Adjust if needed for your environment
    
    FLAGS["depth"] = 1000
    FLAGS["mu"] = FLAGS["depth"] / 2
    FLAGS["sd"] = FLAGS["depth"] / 5
    
    dataset = "SVHN"
    FLAGS["train_dataset"], FLAGS["test_dataset"] = data(dataset)

    
    FLAGS["lr"] = 5e-5
    
    #RUN 2
    FLAGS["path"] = f'{dataset}/left_btlnck_pos_0_50_lr_5e-5/'
    FLAGS["writer"] = SummaryWriter(f'runs/{FLAGS["path"]}')
    FLAGS["bottleneck"] = 0.50
    
    xmp.spawn(train_model, args=(FLAGS,), nprocs=1, start_method="fork")