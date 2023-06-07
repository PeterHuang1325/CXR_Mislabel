import torch
import argparse
import torch.nn as nn
import torch.optim as optim
#import time
import numpy as np
from tqdm import tqdm
from utils import gam_logit_select, gamma_logit_loss


# Training function.
def train(model, trainloader, optimizer, criterion, wei_loss, gamma, mislabel_rate, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    '''store gamma updates'''
    gamma_full = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        
        '''weighted loss'''
        if wei_loss:
            pos_wei= 1-(labels.sum()/len(labels)) #[1,1,0,1,0,...] -> [0,0,1,0,1,]
            weights = torch.Tensor([pos_wei]).float().to(device) #[0.25, 0.75]
        else:
            weights = None
        '''add mislabel'''
        if mislabel_rate > 0:
            mis_idx = round(mislabel_rate*len(labels))
            labels[:mis_idx] = 1 - labels[:mis_idx]
        
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        
        '''Calculate the loss'''
        if criterion == 'gamma_logit_loss':
            loss, gam_update = gamma_logit_loss(outputs, labels, weights, gamma)
            gamma_full.append(gam_update)
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=weights)(outputs, labels.to(torch.float32))
        
        train_running_loss += loss.item()
        '''Calculate the accuracy'''
        #_, preds = torch.max(outputs.data, 1)
        preds = torch.where(outputs.data>0.5, 1, 0)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        # Update the weights.
        optimizer.step()
    
    '''average of gamma update'''
    gamma = np.mean(gamma_full)
    '''Loss and accuracy for the complete epoch'''
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, gamma

# Validation function.
def validate(model, validloader, criterion, wei_loss, gamma, mislabel_rate, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            '''weighted loss'''
            if wei_loss:
                pos_wei= 1-(labels.sum()/len(labels)) #[1,1,0,1,0,...] -> [0,0,1,0,1,]
                weights = torch.Tensor([pos_wei]).float().to(device)
            else:
                weights = None
            '''add mislabel'''
            if mislabel_rate > 0:
                mis_idx = round(mislabel_rate*len(labels))
                labels[:mis_idx] = 1 - labels[:mis_idx]
                
            #forward pass
            outputs = model(image)
            '''Calculate the loss'''
            if criterion == 'gamma_logit_loss':
                loss, _  = gamma_logit_loss(outputs, labels, weights, gamma)
            else:
                loss = nn.BCEWithLogitsLoss(pos_weight=weights)(outputs, labels.to(torch.float32))
            valid_running_loss += loss.item()
            '''Calculate the accuracy'''
            #_, preds = torch.max(outputs.data, 1)
            preds = torch.where(outputs.data>0.5, 1, 0) #0.8, mislabeling 0.8
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(validloader.dataset))
    
    return epoch_loss, epoch_acc