import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

#focal loss
def focal_loss(Dataset):
    #focal loss
    focal_loss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                #alpha=weights.to(device),
                gamma=2,
                reduction='mean',
                force_reload=False)
    return focal_loss


'''gamma selection'''
def gam_logit_select(points, pred, label, weights, rho=0.5):
    #points shape: (batch_size, 1, 256, 256)
    eps = 1e-5
    '''get median position index'''
    slice_points = torch.mean(points.reshape((points.shape[0],-1)), axis=-1) #(16, 224x224)
    med_idx = torch.argsort(slice_points)[len(slice_points)//2] #median index
    #print(pred.shape, med_idx)
    '''compute pos and neg'''
    logit = torch.sigmoid(pred[med_idx]) #compute logit for median
    '''compute positive and negative'''
    pos = (rho-1)/(torch.log(logit)+eps)
    neg = (1-rho)/(torch.log(1+torch.exp(pred[med_idx]))+eps)
    
    #gamma = torch.mean(label[med_idx]*pos+(1-label[med_idx])*neg)
    gamma = weights*label[med_idx]*pos+(1-weights)*(1-label[med_idx])*neg
    gam_clip = torch.clamp(gamma, 1e-4, 1).data.cpu().numpy()
    return gam_clip

def gamma_logit_loss(pred, label, weights, gamma=1e-4):
    '''
    gamma selection: positive number, min: 1e-6
    '''
    pred = torch.sigmoid(pred) #activation layer
    pos = torch.sigmoid((1+gamma)*pred) #gamma logistic layer
    neg = 1 - pos
    power = gamma/(1+gamma)
    #gamma_loss = (1/gamma)*torch.mean(label*(1-(pos**power)) + (1-label)*(1-(neg**power)))
    if weights is None:
        score = label*((1-(pos**power))/gamma) + (1-label)*((1-(neg**power))/gamma)
    else:
        score = weights*label*((1-(pos**power))/gamma) + (1-weights)*(1-label)*((1-(neg**power))/gamma)
    '''gamma selection for this batch, used for computing mean gamma for next FL round'''
    #gam_update = gam_logit_select(score, pred, label, weights)
    gam_update = 1e-4
    gamma_loss = torch.mean(score)
    return gamma_loss, gam_update
            
            
def save_model(exp_name, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    save_dir = f'./outputs/model_ckpt/{exp_name}/'
    os.makedirs(save_dir, exist_ok=True)  
    torch.save({'model_state_dict': model.state_dict()}, f"{save_dir}/model_best.pth")

    
    
    
