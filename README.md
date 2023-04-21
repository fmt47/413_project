# Optimizing CNN Models for COVID-19 Detection: A Comparative Analysis of Optimizers and Loss Functions
 Course project for CSC413: Neural Networks and Deep Learning

Our report can be found in `report.pdf`

## Train-validation-test split:
On around 28000 training data, performed train-validation split of 80%-20%

All evaluation metrics reported based on an unseen test dataset: 200 negative cases, 200 positive cases

## Model 1: Modified ResNet 152

Best Optimizer and Loss function: RMSprop + Cross Entropy

Accuracy: 98.0%, Sensitivity: 96.0%, Specificity: 100.0%, F-1 Score: 96.6%

<img width="635" alt="image" src="https://user-images.githubusercontent.com/32078486/233146058-6df2ce5a-bf2a-432d-89e7-133792d535be.png">

Run the model by executing the file `resnet152/resnet152.ipynb` 

## Model 2: Small COVID-Net


Best Optimizer and Loss function: AdamW + Cross Entropy

Accuracy: 89.5%, Sensitivity: 82.5%, Specificity: 96.5%

![image](https://user-images.githubusercontent.com/32078486/233145939-3c3c7498-67ad-45c9-9136-9ab5e1a16098.png)

Run the model by executing the file `small_covid_net.ipynb`, preferably on Kaggle

## Model 3: Shallow CNN

Best Optimizer and Loss function: SGD (Stochastic Gradient Descent) + Cross Entropy

Accuracy: 89.0%, Sensitivity: 82.0%, Specificity: 96.0%

<img width="558" alt="image" src="https://user-images.githubusercontent.com/32078486/233146128-b10bb290-8d21-445a-b36f-2fc47ffa5488.png">

Run the model by executing the files `shallowCNN/train_ce_nll.py` and `shallowCNN/train_mse.py`

### Note: file paths in the .py or .ipynb files may need to be changed depending on the executing environment
