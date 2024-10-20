import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def plot_conv_layer(image, conv_layer):
    feature_maps = conv_layer(image.unsqueeze(0))
    
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    for i in range(32):
        ax = axs[i // 8, i % 8]
        ax.imshow(feature_maps[0, i].detach().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('conv_layer_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pooling_layer(image, pool_layer):
    feature_maps = pool_layer(image.unsqueeze(0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(feature_maps[0].permute(1, 2, 0).detach().numpy())
    ax2.set_title('After Max Pooling')
    ax2.axis('off')
    plt.show()
    plt.savefig('pooling_layer_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_activations(model, image, layer_name):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.eval()
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(layer_name))

    with torch.no_grad():
        output = model(image.unsqueeze(0))

    activations = activation[layer_name]
    
    num_channels = activations.size(1)
    rows = int(np.ceil(np.sqrt(num_channels)))
    cols = int(np.ceil(num_channels / rows))
    
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.imshow(activations[0, i].numpy(), cmap='viridis')
        ax.axis('off')
    
    # Turn off any unused subplots
    for i in range(num_channels, rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{layer_name}_activations.png', dpi=300, bbox_inches='tight')
    plt.close()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def main():
    # Load and preprocess the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Visualize convolutional and pooling layers
    image = next(iter(trainloader))[0][0]
    conv_layer = nn.Conv2d(3, 32, 3, padding=1)
    plot_conv_layer(image, conv_layer)
    pool_layer = nn.MaxPool2d(2, 2)
    plot_pooling_layer(image, pool_layer)

    # Build and train a simple CNN
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_losses, train_accuracies, val_losses, val_accuracies = train(model, trainloader, testloader, criterion, optimizer, epochs=10)
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    # Visualize activations
    visualize_activations(model, image, 'conv1')

    print("All visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main()