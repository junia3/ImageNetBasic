import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
def collate_func(data):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   inputs = [input for input, _ in data]
   labels = [label for _, label in data]   
   repeat = inputs[0].shape[0]   

   if inputs[0].dim() == 3:
       inputs = torch.stack(inputs, 0)
       labels = torch.LongTensor(labels)   
   else:
       inputs = torch.cat(inputs, 0)
       labels = torch.LongTensor(labels)
       labels = torch.repeat_interleave(labels, repeat)
        
   return inputs.to(device), labels.to(device)
    
def build_STL10_dataloader(path, params, split="train"):
    if split not in ['train', 'test']:
        raise ValueError(split, "is not available(train or test is available)")
	
    data_transformer = transforms.Compose([transforms.ToTensor()])
    if split == 'train':
        dataset = datasets.STL10(path, split='train', download=True, transform=data_transformer)
		
    else:
        dataset = datasets.STL10(path, split='test', download=True, transform=data_transformer)

    meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset]	
    stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset]
	
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
 
    if split == 'train':
        if params['crop']:
            transformer = transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(params['input_dim']),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
			])

        else:
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(params['input_dim']), 
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
			])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(params['input_dim']),
            transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
        ])
	
    dataset.transform = transformer

	# if split is test, re-split it into validation dataset and test dataset
	
    if split != 'train':
        sss = StratifiedShuffleSplit(n_splits=params['n_split'], test_size=params['test_size'], random_state=0)
        indices = list(range(len(dataset)))
        test_label = [y for _, y in dataset]
		
        for t_idx, v_idx in sss.split(indices, test_label):
            print("test :",len(t_idx), "val :",len(v_idx))

	
        val_dataset = Subset(dataset, v_idx)
        test_dataset = Subset(dataset, t_idx)
		
        val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_func)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_func)
        return val_dataloader, test_dataloader

    else:
        train_dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_func)
        return train_dataloader

