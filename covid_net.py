import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from data import COVIDDataset

# Things to try: search "TUNABLE"

class PEPX(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(PEPX, self).__init__()
        hidden_channels = int(out_channels / expansion_factor)

        self.projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.expansion = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.second_projection = nn.Conv2d(out_channels, hidden_channels, kernel_size=1)
        self.extension = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.projection(x)
        # identity = x
        x = self.expansion(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.depthwise(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.second_projection(x)
        x = self.extension(x)
        # x += identity
        x = nn.functional.relu(x, inplace=True)
        return x
    
class PEPX_Downsample(PEPX):
    """Same as pepx but makes the input smaller by half"""
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(PEPX_Downsample, self).__init__(in_channels, out_channels, expansion_factor)
        self.downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        
    def forward(self, x):
        x = self.projection(x)
        # identity = x
        x = self.expansion(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.depthwise(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.second_projection(x)
        x = self.extension(x)
        # x += identity
        x = nn.functional.relu(x, inplace=True)
        x = self.downsample(x)
        return x


class COVIDNet(nn.Module):
    def __init__(self, num_classes=2):
        super(COVIDNet, self).__init__()
        # first conv layer
        self.conv7x7 = nn.Conv2d(3, 56, kernel_size=7, stride=2, padding=3)
        # top conv layers
        self.conv1 = nn.Conv2d(56, 56, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(56 * 4, 112, kernel_size=4, stride=4)
        # self.conv3 = nn.Conv2d(112 * 5, 224, kernel_size=2, stride=2)
        # self.conv4 = nn.Conv2d(216 * 5 + 224 * 2, 424, kernel_size=2, stride=2)

        # pepx layers
        self.pepx11 = PEPX_Downsample(56, 56)
        self.pepx12 = PEPX(56 * 2, 56)
        self.pepx13 = PEPX(56 * 3, 56)

        self.pepx21 = PEPX_Downsample(56 * 4, 112)
        self.pepx22 = PEPX(112 * 2, 112)
        self.pepx23 = PEPX(112 * 3, 112)
        self.pepx24 = PEPX(112 * 4, 112)

        # self.pepx31 = PEPX_Downsample(112 * 5, 216)
        # self.pepx32 = PEPX(216 + 224, 216)
        # self.pepx33 = PEPX(216 * 2 + 224, 216)
        # self.pepx34 = PEPX(216 * 3 + 224, 216)
        # self.pepx35 = PEPX(216 * 4 + 224, 216)
        # self.pepx36 = PEPX(216 * 5 + 224, 224)

        # self.pepx41 = PEPX_Downsample(216 * 5 + 224 * 2, 424)
        # self.pepx42 = PEPX(424 * 2, 424)
        # self.pepx43 = PEPX(424 * 3, 400)

        # final layers
        self.flatten = nn.Flatten()
        # 370800? 15x15x(400+424+424+424)
        # NOT 460800? 15x15x(400+400+400+424+424)
        self.fc = nn.Linear(126000, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Conv7x7 layer
        x = self.conv7x7(x)
        # x = nn.BatchNorm2d(x.size()[1])(x)
        x = nn.ReLU(inplace=True)(x)
        
        # print(pepx11")
        # PEPX11 module
        pepx11 = self.pepx11(x)
        conv1 = self.conv1(x)

        # print(f"size of pepx11 = {pepx11.size()}, size of conv1 = {conv1.size()}")
        # print(pepx12")        
        # PEPX12 module
        pepx12 = self.pepx12(torch.cat((pepx11, conv1), dim=1))
        # print(f"size of pepx12 = {pepx12.size()}")

        # print(pepx13")
        # PEPX13 module
        pepx13 = self.pepx13(torch.cat((pepx12, pepx11, conv1), dim=1))
        # print(f"size of pepx13 = {pepx13.size()}")

        # print(pepx21")
        # PEPX21 module
        cat = torch.cat((pepx13, pepx12, pepx11, conv1), dim=1)
        pepx21 = self.pepx21(cat)
        conv2 = self.conv2(cat)
        # print(f"size of pepx21 = {pepx21.size()}, size of conv2 = {conv2.size()}")

        # print(pepx22")
        # PEPX22 module
        pepx22 = self.pepx22(torch.cat((pepx21, conv2), dim=1))
        # print(f"size of pepx22 = {pepx22.size()}")

        # print(pepx23")
        # PEPX23 module
        pepx23 = self.pepx23(torch.cat((pepx22, pepx21, conv2), dim=1))
        # print(f"size of pepx23 = {pepx23.size()}")

        # print(pepx24")
        # PEPX24 module
        pepx24 = self.pepx24(torch.cat((pepx23, pepx22, pepx21, conv2), dim=1))
        # print(f"size of pepx24 = {pepx24.size()}")

        # # print(pepx31")
        # # PEPX31 module
        # cat = torch.cat((pepx24, pepx23, pepx22, pepx21, conv2), dim=1)
        # pepx31 = self.pepx31(cat)
        # conv3 = self.conv3(cat)
        # # print(f"size of pepx31 = {pepx31.size()}, size of conv3 = {conv3.size()}")

        # # print(pepx32")
        # # PEPX32 module
        # pepx32 = self.pepx32(torch.cat((pepx31, conv3), dim=1))
        # # print(f"size of pepx32 = {pepx32.size()}")

        # # print(pepx33")
        # # PEPX33 module
        # pepx33 = self.pepx33(torch.cat((pepx32, pepx31, conv3), dim=1))
        # # print(f"size of pepx33 = {pepx33.size()}")

        # # print(pepx34")
        # # PEPX34 module
        # pepx34 = self.pepx34(torch.cat((pepx33, pepx32, pepx31, conv3), dim=1))
        # # print(f"size of pepx34 = {pepx34.size()}")

        # # print(pepx35")
        # # PEPX35 module
        # pepx35 = self.pepx35(torch.cat((pepx34, pepx33, pepx32, pepx31, conv3), dim=1))
        # # print(f"size of pepx35 = {pepx35.size()}")

        # # print(pepx36")
        # # PEPX36 module
        # pepx36 = self.pepx36(torch.cat((pepx35, pepx34, pepx33, pepx32, pepx31, conv3), dim=1))
        # # print(f"size of pepx36 = {pepx36.size()}")

        # # print(pepx41")
        # # PEPX41 module
        # cat = torch.cat((pepx36, pepx35, pepx34, pepx33, pepx32, pepx31, conv3), dim=1)
        # pepx41 = self.pepx41(cat)
        # conv4 = self.conv4(cat)
        # # print(f"size of pepx41 = {pepx41.size()}, size of conv4 = {conv4.size()}")

        # # print(pepx42")
        # # PEPX42 module
        # pepx42 = self.pepx42(torch.cat((pepx41, conv4), dim=1))
        # # print(f"size of pepx42 = {pepx42.size()}")

        # # print(pepx43")
        # # PEPX43 module
        # pepx43 = self.pepx43(torch.cat((pepx42, pepx41, conv4), dim=1))
        # # print(f"size of pepx43 = {pepx43.size()}")
    
        # Flatten and FC layer
        x = self.flatten(torch.cat([pepx24, pepx23, pepx22, pepx21, conv2], dim=1))
        # print(f"size of flatten = {x.size()}")
        x = self.fc(x)
        # print(f"size of fc = {x.size()}")
        
        # Softmax activation
        x = self.softmax(x)
        # print(f"size of softmax = {x.size()}")
        
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(f'Loading # {len(train_loader)} datas')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f'At # {batch_idx}')
        data, target = data.to(device), target.to(device)

        # instead of  optimizer.zero_grad(), do it in forloop for more efficiency
        for param in model.parameters():
            param.grad = None

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    acc = 100.*correct/total
    return train_loss, acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
    acc = 100.*correct/total
    return val_loss, acc


if __name__ == '__main__':

    # Set up hyperparameters and data loaders
    # TUNABLE:
    lr = 2e-4
    # TUNABLE:
    epochs = 22
    # TUNABLE: original 64
    batch_size = 8
    # TUNABLE:
    factor = 0.7
    # TUNABLE:
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the transformations to be applied to the data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # create COVIDDataset object
    # TUNABLE: --> can try different datasets
    dataset = COVIDDataset('train.txt', transform=transform)

    # get size of dataset
    dataset_size = len(dataset)

    # split dataset into train and validation sets
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Set up model, optimizer, and loss function
    model = COVIDNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TUNABLE:
    criterion = F.cross_entropy
    #criterion = F.mse_loss
    
    # Train the model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    best_acc = 0

    # record training and validation losses and accuracies
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # record training and validation losses and accuracies
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'covid_net.pt')
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

    # plot training and validation losses and accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.legend()
    plt.show()

    # test data
    test_dataset = COVIDDataset('test.txt', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # load best model
    model.load_state_dict(torch.load('covid_net.pt'))

    # evaluate model on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%')

