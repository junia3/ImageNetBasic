import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import torch.optim as optim
from alexnet import AlexNet
from vggnet import VGGNet
from resnet import ResidualBlock, BottleNeckResidualBlock, ResNet 
from train import *
from datasets import *
from torchsummary import summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    available_model = ["AlexNet", "VGG11", "VGG13", "VGG16", "VGG19", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    model_name = input("Enter your model name ("+str(available_model)+" is available) : ")
    
    # default parameters
    num_epochs = 90
    batch_size = 128
    momentum = 0.9
    lr_decay = 5e-4
    lr = 1e-2
    input_dim = 227
    num_classes = 10
    loss_func = nn.CrossEntropyLoss()

    if model_name not in available_model:
        print("Wrong model, You have to choose one in "+str(available_model))
        return

    elif model_name == "AlexNet":
        num_classes=10
        model = AlexNet(num_classes)
        num_epochs = 90
        batch_size = 128
        momentum = 0.9
        lr_decay = 5e-4
        lr = 0.01
        input_dim = 227
        loss_func = nn.CrossEntropyLoss()
        
    elif 'VGG' in model_name:
        num_classes=10
        model = VGGNet(model_name, num_classes)
        num_epochs = 90
        batch_size = 4
        momentum = 0.9
        lr_decay = 5e-4
        lr = 0.0001
        input_dim = 224
        loss_func = nn.CrossEntropyLoss()

    elif 'ResNet' in model_name:
        num_classes=10
        model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)
        if '18' in model_name:
            model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)
        elif '34' in model_name:
            model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)
        elif '50' in model_name:
            model = ResNet(BottleNeckResidualBlock, [3, 4, 6, 3], num_classes)
        elif '101' in model_name:
            model = ResNet(BottleNeckResidualBlock, [3, 4, 23, 3], num_classes)
        elif '152' in model_name:
            model = ResNet(BottleNeckResidualBlock, [3, 8, 36, 3], num_classes)
        num_epochs = 90
        batch_size = 64
        momentum = 0.9
        lr_decay = 1e-4
        lr = 0.1
        input_dim = 224
        loss_func = nn.CrossEntropyLoss()


    INPUT_ROOT_DIR = "data_in"
    TRAIN_IMG_DIR = "data_in/STL10_TRAIN"
    TEST_IMG_DIR = "data_in/STL10_TEST"
    OUTPUT_DIR = "data_out"
    LOG_DIR = OUTPUT_DIR+"/"+model_name+"/logs"
    CHECKPOINT_DIR = OUTPUT_DIR+"/"+model_name+"/models"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(model_name+" created(device is {}) ...".format(device))
	
    params = {}
    params['input_dim']	= input_dim
    params['crop'] = False # if set this parameter true, five crop transform occurs
    params['n_split'] = 1
    params['test_size'] = 0.2
    params['batch_size'] = batch_size

    train_dataloader = build_STL10_dataloader(TRAIN_IMG_DIR, params, "train")
    val_dataloader, test_dataloader = build_STL10_dataloader(TEST_IMG_DIR, params, "test")

    print("Dataloader created ...")
    # SGD optimizer
    '''
    optimizer = optim.SGD(
        params = model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=lr_decay
    )
    '''
    # Adam optimizer
    
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = lr,
        weight_decay = lr_decay
    )
    
    print("Optimizer created ...")
    # stepLR scheduler
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # reduced on plateau scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
    print("Scheduler created ...")

    print("Start trainining !!")
    summary(model.cuda(), input_size=(3, input_dim, input_dim), device=device.type)
    train_model(model, train_dataloader, val_dataloader, num_epochs, loss_func, optimizer, lr_scheduler, LOG_DIR, CHECKPOINT_DIR, device, model_name, params['crop'])

if __name__ == "__main__":
    main()

