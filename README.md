# Chest X-Ray Pneumonia classification
In this repo, we provide an experiment on CXR dataset(Chest X-Ray) with mislabeled data featuring:

- **Pneumonia** binary classification problem
- Compare **γ-logistic loss** with conventional **cross entropy**
- Data-driven γ-selection methods for simple γ-mean and γ-logistic

## Usage
1. **Prepare data:**
     - Download dataset from [CXR link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):
     
2. **Run experiments (training):**
     - Train from scratch:
       - Without mislabel data:
          - Run `python main.py`
       - With mislabel data:
          - Run `python main.py --mislabel_rate=float(0-1)`

     - Train with pretrained weight:
       - Without mislabel data:
          - Run `python main.py --pretrained`
       - With mislabel data:
          - Run `python main.py --pretrained --mislabel_rate=float(0-1)`
3. **Run experiments (testing):**
     - Run `python inference.py`.
4. **Output:**
     - ROC curve, confusion matrix, prediction report csv are saved in `outputs/logs/{exp_name}/`.
     <img src=/images/cxr_table.png width=50% height=50%>
