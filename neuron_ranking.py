import dill  # in order to save Lambda Layer
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from resnet import *

def main():

    criterion = nn.CrossEntropyLoss().cuda()

    # the network architecture cor6esponding to the checkpoint
    model = resnet20()

    # remember to set map_location
    check_point = torch.load('D:/Documents/Kuliah/OneDrive - Telkom University/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/pretrained_models/resnet20-12fca82f.th', map_location={'cuda:1': 'cuda:0'})

    # cause the model are saved from Parallel, we need to wrap it
    model = torch.nn.DataParallel(model)
    model.load_state_dict(check_point['state_dict'])

    # pay attention to .module! without this, if you load the model, it will be attached with [Parallel.module]
    # that will lead to some trouble!
    torch.save(model.module, 'D:/Documents/Kuliah/OneDrive - Telkom University/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/pretrained_models/resnet20_check_point.pth', pickle_module=dill)

    # load the converted pretrained model
    net = torch.load('D:/Documents/Kuliah/OneDrive - Telkom University/Kuliah/S1/GARUDA ACE/Coding/1-20-23_resnet_pytorch/pretrained_models/resnet20_check_point.pth', map_location={'cuda:1': 'cuda:0'})
    net.eval()
    net.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)    

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total

    print('Accuracy of the network on the test images:', accuracy * 100, '%')
    print('----------------------------------------------------------------')

    # Set the model in evaluation mode 
    net.eval()

    # Perturb the values of each neuron and measure the impact on the performance
    scores = []
    #with torch.no_grad(): tambah tab after for
    for name, param in net.named_parameters():
        if 'weight' in name:
            # Save the original weights
            original_weights = param.data.clone()

            # Perturb the weights
            param.data += 0.1 
            #param.add_(0.1)

            # Forward pass
            output = net(images)
            loss = criterion(output, labels)

            # Compute the score (impact on the performance)
            score = loss.item()

            # Append the score to the list
            scores.append((name, score))

            # Restore the original weights
            param.data = original_weights

    # Sort the scores and print the ranking
    scores = sorted(scores, key=lambda x: x[1],reverse=True)
    for i, (name, score) in enumerate(scores):
        print(f'{i+1}. {name} - {score:.4f}')

    '''    
    # Define a threshold for pruning
    threshold = 0.01

    # Prune the neurons
    for name, param in net.named_parameters():
        if 'weight' in name:
            # Get the absolute values of the weights
            weights = torch.abs(param)

            # Get the mask of the weights below the threshold
            mask = weights < threshold

            # Zero out the weights below the threshold
            param.data[mask] = 0

    # Print the number of pruned neurons
    print('-----------------------------------')
    print(f'Pruned {mask.sum().item()} neurons')
    '''
if __name__ == '__main__':
    main()