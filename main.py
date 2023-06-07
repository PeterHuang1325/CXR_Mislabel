import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import seed_everything
import time
import os
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, focal_loss
from train import train, validate

def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', type=str, default='xxxx',
        help='experiment setting name'
    )
    parser.add_argument(
         '--epochs', type=int, default=50,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '--loss_fn', type=str, default='cross_entropy',
        help='loss function'
    )
    parser.add_argument(
        '--weight_loss', action='store_true', default=False,
        help='Whether to use weighted loss'
    )
    parser.add_argument(
        '--gamma', type=int, default=1e-4,
        help='initial gamma value'
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=False,
        help='Whether to use pretrained weights or not'
    )
    parser.add_argument(
        '--learning_rate', type=float,
        dest='learning_rate', default=1e-4,
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '--image_size', type=int, default=224,
        help='image size for training the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, #16
        help='batch size for training'
    )
    parser.add_argument(
        '--num_classes', type=int, default=1,
        help='number of label classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='parellel num workers'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='seed for experiment'
    )
    parser.add_argument(
        '--patience', type=int, default=20,
        help='patience for early stop'
    )
    parser.add_argument(
        '--mislabel_rate', type=float, default=0,
        help='mislabel rate for training set'
    )
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 


#main function
def main():
    
    #read parsed
    args = read_options()
    
    '''set seed'''
    seed_everything(args['seed'])
    
    # Load the training and validation datasets.
    train_set, valid_set = get_datasets(args['image_size'])
    dataset_classes = args['num_classes']
    wei_loss = args['weight_loss']
    
    print(f"[INFO]: Number of training images: {len(train_set)}")
    print(f"[INFO]: Number of validation images: {len(valid_set)}")

    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(train_set, valid_set, args['batch_size'], args['num_workers'])
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    
    '''Build Model'''
    model = build_model(pretrained=args['pretrained'], fine_tune=True, num_classes=args['num_classes']).to(device)
    
    '''Total parameters and trainable parameters'''
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    '''Optimizer'''
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], last_epoch=-1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    
    '''Loss function'''
    criterion = args['loss_fn']
    '''mislabel rate'''
    mislabel_rate = args['mislabel_rate']
    
    '''create log path for expr'''
    exp_name = args['exp_name']
    log_path =  f'./outputs/logs/{exp_name}/'
    os.makedirs(log_path, exist_ok=True)
    
    '''some initials'''
    best_loss = 1e3
    stale = 0 
    gamma = args['gamma']
    '''Training'''
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, gamma = train(model, train_loader, optimizer, criterion, wei_loss, gamma, mislabel_rate, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, wei_loss, gamma, mislabel_rate, device)
        '''save gamma values'''
        if criterion == 'gamma_logit_loss':
            with open(os.path.join(log_path, 'gamma.txt'),'a') as f:
                '''print results'''
                print(f"[epoch {epoch}]: gamma_value = {gamma:.4f}")
                '''write to file'''
                print(f"epoch {epoch}:, gamma_value = {gamma:.4f}", file=f)
                
        '''write results'''
        with open(os.path.join(log_path, 'history.txt'),'a') as f:
            '''print results'''
            print(f"[epoch {epoch}]: train_loss = {train_epoch_loss:.4f}, train_acc = {train_epoch_acc:.4f}, val_loss = {valid_epoch_loss:.4f}, val_acc = {valid_epoch_acc:.4f}")
            '''write to file'''
            print(f"epoch {epoch}, train_loss = {train_epoch_loss:.4f}, train_acc = {train_epoch_acc:.4f}, val_loss = {valid_epoch_loss:.4f}, val_acc = {valid_epoch_acc:.4f}", file=f)
        
        scheduler.step()
        
        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            
            # Save the trained model weights.
            print(f'Save best model at {epoch+1} epoch:')
            save_model(exp_name, epochs, model, optimizer, criterion)
            stale = 0
        else:
            stale += 1
            if stale > args['patience']:
                print(f"No improvment {args['patience']} consecutive epochs, early stopping")
                break
            
        print('-'*50)
        time.sleep(5)
        
    print('TRAINING COMPLETE')
    print('-'*50)

if __name__ == '__main__':
    main()
