import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Define directory for saving weights and status
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_resnet50.pth")
STATUS_PATH = os.path.join(BASE_DIR, "training_status.json")

class MNIST_ResNet50(nn.Module):
    def __init__(self):
        super(MNIST_ResNet50, self).__init__()
        # Load ResNet50 with default pre-trained weights
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        
        # Modify the first conv layer to accept 1 input channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the fully connected layer to output 10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        
    def forward(self, x):
        return self.resnet(x)


def update_status(status, progress, message, loss=None, accuracy=None):
    import tempfile
    data = {
        "status": status,      # "idle", "training", "completed", "failed"
        "progress": progress,  # percentage 0 to 100
        "message": message,
        "loss": loss,
        "accuracy": accuracy
    }
    dir_name = os.path.dirname(STATUS_PATH)
    # Write atomically using a temporary file in the same directory
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
        json.dump(data, tf)
        temp_name = tf.name
    os.replace(temp_name, STATUS_PATH)

def train_mnist_model(epochs=2, batch_size=128):
    try:
        update_status("training", 0, "Initializing training environment...")
        
        # Check device (use MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        print(f"Training MNIST model on device: {device}")
        
        # Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        update_status("training", 10, "Downloading / Loading MNIST dataset...")
        
        data_dir = os.path.join(BASE_DIR, "data")
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = MNIST_ResNet50().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        total_batches = len(train_loader)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update status every 50 batches
                if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                    batch_progress = int(((epoch * total_batches + batch_idx) / (epochs * total_batches)) * 80) + 10
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = 100.0 * correct / total
                    msg = f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches}"
                    update_status("training", batch_progress, msg, round(current_loss, 4), round(current_acc, 2))
            
            epoch_loss = running_loss / total_batches
            epoch_acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
        update_status("training", 95, "Saving trained model parameters...")
        torch.save(model.state_dict(), MODEL_PATH)
        update_status("completed", 100, "Model trained successfully!", round(epoch_loss, 4), round(epoch_acc, 2))
        print("Model saved to", MODEL_PATH)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        update_status("failed", 100, f"Error: {str(e)}")

if __name__ == "__main__":
    # Initialize the status to idle if running script directly
    update_status("idle", 0, "Ready to train.")
    train_mnist_model(epochs=2, batch_size=128)
