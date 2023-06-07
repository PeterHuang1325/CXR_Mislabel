import torch
import numpy as np
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from pytorch_lightning import seed_everything
from PIL import Image
import time
import os
from tqdm.auto import tqdm
from model import build_model
import pandas as pd
#some metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--exp_name', type=str, default='xxxx',
        help='experiment setting name'
    )
    parser.add_argument(
         '--pretrained', action='store_false', default=True,
        help='Whether to use pretrained weights or not'
    )
    parser.add_argument(
        '--image_size', type=int, default=224,
        help='image size for training the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
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

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed 

#transform
def get_eval_transform(image_size):
    
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    return eval_transform

#dataset
class CXR_Dataset(Dataset):

    def __init__(self, path, tfm, files = None):
        super(CXR_Dataset).__init__()
        self.path = path
        self.normal = [os.path.join(path+'/NORMAL',x) for x in os.listdir(path+'/NORMAL') if x.endswith(".jpeg")]
        self.pneumo = [os.path.join(path+'/PNEUMONIA',x) for x in os.listdir(path+'/PNEUMONIA') if x.endswith(".jpeg")]
        self.files = self.normal+self.pneumo
        if files != None:
            self.files = files
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
      
    def __filename__(self,idx):
        return f'file_{idx}'
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        split = fname.split('/')[-2]
        if split == 'PNEUMONIA':
            label = 1
        else:
            label = 0
        return im, label


    
'''prediction by thrs'''
def pred_thrs(results, labels):
    
    step = 0.1
    thrs_list = np.arange(0.1,1,step)
    test_columns = ['threshold', 'precision', 'recall', 'f1-score', 'accuracy']
    score_matrix = np.zeros((len(thrs_list),len(test_columns)), dtype=object)
    
    best_f1 = 0
    best_pred = np.zeros(len(labels))
    #for idx in range(1, len(thrs_list)):
    for idx, threshold in enumerate(thrs_list):
        print('-'*50)
        print('threshold:', threshold.round(1))
        
        #for t, scores in enumerate(results):
        test = np.copy(results) #copy for not covered
        test = np.where(test<threshold, 0, 1)        
        '''compute metrics'''
        prec = precision_score(labels, test).round(4)
        recall = recall_score(labels, test).round(4)
        f1 = f1_score(labels, test).round(4)
        acc = accuracy_score(labels, test).round(4)
        '''update best predictions'''
        #if f1 > best_f1:
        if threshold == 0.5:
            best_pred = test
        
        print(f'precision:{prec}/recall:{recall}/f1-score:{f1}/acc:{acc}')
        
        score_matrix[idx] = [threshold.round(1), prec, recall, f1, acc]
    score_result = pd.DataFrame(score_matrix, columns=test_columns)   
    return score_result, best_pred    
    
    
#main function
def main():
    
    #read parsed
    args = read_options()
    exp_name = args['exp_name']
    '''set seed'''
    seed_everything(args['seed'])
    '''data path'''
    _dataset_dir = '../../Shared/Pin-Han/chest_xray/'
    # Load the training and validation datasets.
    eval_set = CXR_Dataset(os.path.join(_dataset_dir,'test'), tfm=get_eval_transform(args['image_size']))
    print(f"[INFO]: Number of testing images: {len(eval_set)}")

    # Load the training and validation data loaders.
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'],shuffle=False, num_workers=args['num_workers'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device: {device}")
    
    #build model
    model = build_model(pretrained=args['pretrained'], fine_tune=True, num_classes=args['num_classes']).to(device)
    model.load_state_dict(torch.load(f'./outputs/model_ckpt/{exp_name}/model_best.pth')['model_state_dict'])
    
    model.eval()
    labels, proba_full = [], []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            # image and labels
            imgs, lbls = batch
            eval_prob = model(imgs.to(device)).view(-1)
            labels += lbls.data.cpu().numpy().tolist()
            proba_full += eval_prob.data.cpu().numpy().tolist()
    
    #get files
    '''
    eval_files = [eval_set.__filename__(idx) for idx in range(len(eval_set))]
    out_df = pd.DataFrame(np.array([eval_files, prediction, labels], dtype=object).T, columns=['filename', 'prediction', 'label'])
    print('EVALUATION COMPLETE')
    print('-'*50)
    '''
    eval_files = [eval_set.__filename__(idx) for idx in range(len(eval_set))]
    thrs_df, prediction = pred_thrs(proba_full, labels)
    out_df = pd.DataFrame(np.array([eval_files, prediction, labels], dtype=object).T, columns=['filename', 'prediction', 'label'])
    print('EVALUATION COMPLETE')
    print('-'*50)
    
    save_dir = f'./outputs/logs/{exp_name}/'
    os.makedirs(save_dir, exist_ok=True)
    #output to csv
    thrs_df.to_csv(os.path.join(save_dir, 'thrs_report.csv'), index=False)
    out_df.to_csv(os.path.join(save_dir, 'pred_report.csv'), index=False)
    
    
    
    #classification report
    print('*********************************************************')
    print(classification_report(labels, prediction))
    
    
    #confusion matrix
    conf_matrix = confusion_matrix(prediction, labels)
    fig1, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6))
    plt.xlabel('Actuals ', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(save_dir, 'conf_mtrx.png'))
    
    #ROC and AUC score
    #predicted probability
    fpr, tpr, _ = roc_curve(labels,  proba_full)
    auc = np.round(roc_auc_score(labels, proba_full),4)
    fig2 = plt.figure(figsize = (6,6))
    plt.plot(fpr,tpr,label="ROC, with AUC="+str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig(os.path.join(save_dir, 'roc_auc.png'))
    
if __name__ == '__main__':
    main()