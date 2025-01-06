import torch
from models import YourModel
from datasets import YourDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_model(checkpoint_path):
    model = YourModel()  # Initialize your model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

def main():
    # Define transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    # Load the test dataset
    test_dataset = YourDataset(transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    model = load_model('path/to/your/checkpoint.pth')

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()