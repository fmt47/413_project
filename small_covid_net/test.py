from data import COVIDDataset
from small_covid_net import COVIDNet, validate
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please provide exactly one model pt filename")
        sys.exit(1)
    
    # transforms
    criterion = F.cross_entropy
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = COVIDNet()

    # test data
    test_dataset = COVIDDataset('test.txt', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # load best model
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))

    # evaluate model on test set
    test_loss, test_acc, sensitivity, specificity = validate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%')
    print(f'Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
