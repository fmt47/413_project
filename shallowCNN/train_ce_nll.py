# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

torch.manual_seed(3407)

class CustomDataset(Dataset):
    def __init__(self, root_dir, description_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        with open(description_file, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            # lines = lines[0:100]
            for line in lines:
                patient_id, filename, label, data_source = line.strip().split()
                img_path = os.path.join(root_dir, filename)
                if label == "negative":
                    label_idx = 0
                elif label == "positive":
                    label_idx = 1
                self.data.append((img_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


# Define root directories and description files
train_root_dir = './archive/train/'
train_description_file = './archive/train.txt'
test_root_dir = './archive/test/'
test_description_file = './archive/test.txt'

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])

trainset = CustomDataset(train_root_dir, train_description_file, transform=transform)
testset = CustomDataset(test_root_dir, test_description_file, transform=transform)

batch_size = 25

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

train_indices, val_indices = train_test_split(list(range(len(trainset))), test_size=0.2, random_state=42)

train_subset = Subset(trainset, train_indices)
val_subset = Subset(trainset, val_indices)


class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3,
                               padding=1)  # Assuming input image has 1 channel (grayscale), 32 filters, 3x3 kernel, and padding=1
        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 window
        self.fc1 = nn.Linear(32 * 25 * 25, 256)  # Dense layer with 256 units
        self.fc2 = nn.Linear(256, 2)  # Output layer with 2 units

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution layer followed by ReLU activation and max-pooling
        x = x.view(-1, 32 * 25 * 25)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First dense layer with ReLU activation
        x = self.fc2(x)  # Output layer
        return F.softmax(x, dim=1)  # Softmax activation


train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)


def train(model, optimizer, batch_size, best_model_path, criterion, epochs):
    best_val_acc = 0
    best_val_epoch = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print('start_train')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # i = 0
        # Wrap train_loader with tqdm for progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # if i % 100 == 0:
            #     print('.', end='')
            # i += 1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            torch.save(model.state_dict(), best_model_path)  # Save the best model's state

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_val_epoch + 1}")

    # Test accuracy, loss, sensitivity, and specificity with best epoch
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if labels[i] == 1 and predicted[i] == 1:
                    TP += 1
                elif labels[i] == 0 and predicted[i] == 0:
                    TN += 1
                elif labels[i] == 1 and predicted[i] == 0:
                    FN += 1
                elif labels[i] == 0 and predicted[i] == 1:
                    FP += 1

    test_acc = correct_test / total_test
    test_sensitivity = TP / (TP + FN)
    test_specificity = TN / (TN + FP)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Sensitivity: {test_sensitivity:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")

    plt.figure(figsize=(12, 6))

    # Train Loss vs Epoch
    plt.subplot(221)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epoch')
    plt.legend()

    # Validation Loss vs Epoch
    plt.subplot(222)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs Epoch')
    plt.legend()

    # Train Accuracy vs Epoch
    plt.subplot(223)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Epoch')
    plt.legend()

    # Validation Accuracy vs Epoch
    plt.subplot(224)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print(f'config above: Loss: {criterion}, Opt: {optimizer}, best_model_path: {best_model_path}')


if __name__ == '__main__':
    learning_rate = 0.005
    model = ShallowCNN()

    SGDoptimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    ProPoptimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    Adamoptimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_path_0 = 'best_model_0.pth'

    CEcriterion = nn.CrossEntropyLoss()
    MSEcriterion = nn.MSELoss()

    epochs = 100

    print('start')
    train(model, SGDoptimizer, batch_size, best_model_path_0, CEcriterion, epochs)
