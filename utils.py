import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import attr
import torch
import torch.nn as nn
import torchvision
from PIL import ImageFilter
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder, STL10
import copy
import numpy as np
from functools import partial
from model_params import ModelParams
import torch.nn.functional as F
from torch import utils


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]), 
]

@attr.s(auto_attribs=True)
class SimSIAMTransforms:
    crop_size: int = 224
    resize: int = 256
    normalize_means: list = [0.4914, 0.4822, 0.4465]
    normalize_stds: list = [0.2023, 0.1994, 0.2010]
    s: float = 0.5
    apply_blur: bool = True

    def split_transform(self, img) -> torch.Tensor:
        transform = self.single_transform()
        return torch.stack((transform(img), transform(img)))

    def single_transform(self):
        transform_list = [
            transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
        ]
        if self.apply_blur:
            transform_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=self.normalize_means, std=self.normalize_stds))
        return transforms.Compose(transform_list)

    def get_test_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_means, std=self.normalize_stds),
            ]
        )

@attr.s(auto_attribs=True, slots=True)
class DatasetBase:
    _train_ds: Optional[torch.utils.data.Dataset] = None
    _validation_ds: Optional[torch.utils.data.Dataset] = None
    _test_ds: Optional[torch.utils.data.Dataset] = None
    transform_train: Optional[Callable] = None
    transform_test: Optional[Callable] = None

    def get_train(self) -> torch.utils.data.Dataset:
        if self._train_ds is None:
            self._train_ds = self.configure_train()
        return self._train_ds
    
    
    def configure_train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError
    
    def get_validation(self) -> torch.utils.data.Dataset:
        if self._validation_ds is None:
            self._validation_ds = self.configure_validation()
        return self._validation_ds
    
    def configure_validation(self) -> torch.utils.data.Dataset:
        raise NotImplementedError
    
    @property
    def data_path(self):
        pathstr = os.environ.get('DATA_PATH', os.getcwd())
        os.makedirs(pathstr, exist_ok=True)
        return pathstr
    
    @property
    def instance_shape(self):
        img = next(iter(self.get_train()))[0]
        return img.shape
    
    @property
    def num_classes(self):
        train_ds = self.get_train()
        if hasattr(train_ds, 'classes'):
            return len(train_ds.classes)
        return None
    
stl10_default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


@attr.s(auto_attribs=True, slots=True)
class STL10UnlabeledDataset(DatasetBase):
    transform_train: Callable[[Any], torch.Tensor] = stl10_default_transform
    transform_test: Callable[[Any], torch.Tensor] = stl10_default_transform

    def configure_train(self):
        return STL10(self.data_path, split="train+unlabeled", download=True, transform=self.transform_train)

    def configure_validation(self):
        return STL10(self.data_path, split="test", download=True, transform=self.transform_test)


@attr.s(auto_attribs=True, slots=True)
class STL10LabeledDataset(DatasetBase):
    transform_train: Callable[[Any], torch.Tensor] = stl10_default_transform
    transform_test: Callable[[Any], torch.Tensor] = stl10_default_transform

    def configure_train(self):
        return STL10(self.data_path, split="train", download=True, transform=self.transform_train)

    def configure_validation(self):
        return STL10(self.data_path, split="test", download=True, transform=self.transform_test)


def get_dataset(hparams: ModelParams) -> DatasetBase:
    if hparams.dataset_name == "stl10":
        crop_size = 32
        resize = 48
        normalize_means = [0.4914, 0.4823, 0.4466]
        normalize_stds = [0.247, 0.243, 0.261]
        transforms = SimSIAMTransforms(crop_size, resize, normalize_means, normalize_stds, 0.5)
        return STL10UnlabeledDataset(transform_train=transforms.split_transform, transform_test=transforms.get_test_transform())
    else:
        raise NotImplementedError
    
"""def get_class_transforms(crop_size, resize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    transform_test = transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(crop_size), transforms.ToTensor(), normalize]
    )
    return transform_train, transform_test


def get_class_dataset(name: str) -> DatasetBase:
    if name == "stl10":
        transform_train, transform_test = get_class_transforms(96, 128)
        return STL10LabeledDataset(transform_train=transform_train, transform_test=transform_test)
    else:
        raise NotImplementedError"""

####################
# Model utils
####################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.input_dim, bias=False),
                                   torch.nn.BatchNorm1d(self.input_dim),
                                   torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(self.input_dim, self.input_dim, bias=False),
                                   torch.nn.BatchNorm1d(self.input_dim),
                                   torch.nn.ReLU(inplace=True),
                                   self.encoder.fc,
                                   torch.nn.BatchNorm1d(self.output_dim),)
        self.encoder.fc[6].bias.requires_grad = False

    def forward(self, x):    
        return self.model(x)
    

def _get_encoder(name: str = 'resnet-50', dataset: str = "stl10", **kwargs) -> torch.nn.Module:
    model_creator = torchvision.models.__dict__.get(name)
    model = model_creator(**kwargs)
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()

    return model
    
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.linear_mean = nn.Linear(prev_dim, prev_dim)
        self.linear_logvar = nn.Linear(prev_dim, prev_dim)
        self.encoder_fc_copy = copy.deepcopy(self.encoder.fc)
        # build a 2-layer projector
        self.projection = nn.Sequential(#nn.Linear(prev_dim, prev_dim, bias=False),
                                        #nn.BatchNorm1d(prev_dim),
                                        #nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder_fc_copy,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.projection[3].bias.requires_grad = False # hack: not use bias as it is followed by BN
        self.encoder.fc = nn.Identity()
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        
        self.decoder = Decoder(latent_dim=prev_dim, in_channels= [512, 256, 128, 64], hidden_dim= [256, 128, 64, 32], out_channels=3)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        h1 = self.encoder(x1) # NxC
        h2 = self.encoder(x2) # NxC
        #mean_q = self.linear_mean(h1)
        #logvar_q = self.linear_logvar(h1)
        #mean_k = self.linear_mean(h2)
        #logvar_k = self.linear_logvar(h2)
        #std1 = torch.exp(0.5*logvar_q)
        #eps1 = torch.randn_like(std1, requires_grad=True)
        #std2 = torch.exp(0.5*logvar_k)
        eps2 = torch.randn_like(h1, requires_grad=True)
        #h1 = eps1.mul(std1).add_(mean_q)
        #h2 = eps1.mul(std2).add_(mean_k)

        z1 = self.projection(h1) # Nxproj_dim
        z2 = self.projection(h2) # Nxproj_dim

        x1_hat = self.decoder(h1 + eps2)
        x2_hat = self.decoder(h2 + eps2)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach(), h1, h2, x1_hat, x2_hat #, mean_q, std1, mean_k, std2
    

class BasicBlockDecoder(nn.Module):

    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, self.hidden_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.upsample = partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.learnable_sc = in_channel != out_channel or upsample
        if self.learnable_sc:
            self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.bn1(h)
        h = self.relu(h)
        if self.upsample is not None:
            h = self.upsample(h) 
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv2(h)
        if self.upsample is not None:
            x = self.upsample(x)
        if self.learnable_sc:
            x = self.conv1x1(x)
        return h + x

class Decoder_2(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, upsample, bottom_width: int=1, observed_channel:int=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.bottom_width = bottom_width
        self.observed_channel = observed_channel
        self.block = []
        for i in range(len(self.out_channels)):
            self.block += [BasicBlockDecoder(self.in_channels[i], self.out_channels[i], self.upsample[i])]
        self.block = nn.ModuleList(self.block)
        self.linear = nn.Linear(self.latent_dim, self.bottom_width**2*self.in_channels[0])
        self.relu = nn.ReLU()
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.out_channels[-1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels[-1], self.observed_channel, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for block in self.block:
            h = block(h)
        h = self.output_layer(h)
        return h
    
class Decoder(nn.Module):

    def __init__(self, latent_dim:int, in_channels: list= [512, 256, 128, 64], hidden_dim: list = [256, 128, 64, 32], out_channels: int=3, bottom_width: int=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.bottom_width = bottom_width
        self.block = []
        for i in range(len(self.hidden_dim)):
            self.block += [nn.ConvTranspose2d(self.in_channels[i], self.hidden_dim[i], kernel_size=3, stride=2, padding=1)]
        self.linear = nn.Linear(self.latent_dim, self.bottom_width**2*self.in_channels[0])
        self.block = nn.ModuleList(self.block)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.hidden_dim[-1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[-1], self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for block in self.block:
            size = h.size()
            h = block(h, output_size=torch.Size([size[0],size[1]//2,size[2]*2,size[3]*2]))
        h = self.output_layer(h)
        return h

####################
# Evaluation utils #
####################


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_softmax_with_factors(logits: torch.Tensor, log_factor: float = 1, neg_factor: float = 1) -> torch.Tensor:
    exp_sum_neg_logits = torch.exp(logits).sum(dim=-1, keepdim=True) - torch.exp(logits)
    softmax_result = logits - log_factor * torch.log(torch.exp(logits) + neg_factor * exp_sum_neg_logits)
    return softmax_result

def M_O(z):
    Z = z / np.linalg.norm(z, axis=1, keepdims=True)
    o = Z.mean(axis=0)
    r = Z - o
    norm_m = np.linalg.norm(o)
    norm_r = np.linalg.norm(r, axis=1)
    m_r = norm_r / np.linalg.norm(z, axis=1)
    m_o = norm_m / np.linalg.norm(z, axis=1)
    return m_r, m_o, r, o