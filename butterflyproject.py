
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



transform = transforms.Compose([
    transforms.Resize((64 , 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15), 
    transforms.RandomCrop((64 , 64), padding = 4), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
])

train_path = './data/train'
test_path = './data/test'
valid_path = './data/valid'


# load the dataset 
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

dataset = datasets.ImageFolder(root="./data/train")

# Check classes
print(f"Classes: {dataset.classes}")
print(f"Number of classes: {len(dataset.classes)}")

class FilteredDataset(Dataset):
    def __init__(self, original_dataset, selected_classes):
        self.dataset = original_dataset
        self.selected_classes = selected_classes
        self.indices = [
            i for i, (_, label) in enumerate(self.dataset.samples)
            if self.dataset.classes[label] in self.selected_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

selected_classes = dataset.classes[:40]  # number of classes im working with first
print(f"Selected classes: {len(selected_classes)}")

# Filter dataset to include only these classes
selected_indices = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in selected_classes]
filtered_dataset = FilteredDataset(train_dataset, selected_classes)

# Split the filtered dataset into train, validation, and test sets
train_size = int(0.7 * len(filtered_dataset)) 
val_size = int(0.15 * len(filtered_dataset))  
test_size = len(filtered_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(filtered_dataset, [train_size, val_size, test_size])


# Create dataloaders
trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=100, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

class ResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride = 1): 
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride=stride , padding = 1 , bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3 , stride= 1 , padding=1 , bias= False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_channels != out_channels: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels , out_channels, kernel_size=1 , stride= stride , bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x): 
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Shortcut path
        shortcut = self.shortcut(x)
        # Add the shortcut to the main path
        out += shortcut
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # Global Average Pooling
        out = torch.flatten(out, 1)  # Flatten
        out = self.fc(out)
        return out
''''          
class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32 , kernel_size= 3 , padding= 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(32 , 64 , kernel_size= 3 , padding= 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size= 2)
        )

        flattened_size = self.get_flattened_size(input_shape = (3, 64 , 64))
        self.fc1 = nn.Linear(flattened_size, 128) # from 28x28 input image to hidden layer of size 256
        self.fc2 = nn.Linear(128, num_classes) # from hidden layer to 10 class scores

    def get_flattened_size( model , input_shape = (3,64 , 64)):
        # Helper function to compute the flattened size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape) 
            sample_output = model.conv_layers(sample_input)
            return sample_output.view(1, -1).size(1) 
        
    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1) # Flatten each image in the batch
        x = self.fc1(x)
        relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        x = relu(x)
        x = self.fc2(x)
        # The loss layer will be applied outside Network class
        return x
'''''

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure devicex
model = ResNet(ResidualBlock,[2,2,2,2], num_classes = 40).to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.SGD(model.parameters(), lr= 10e-4, momentum = 0.9, weight_decay=5e-4 ) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 70 # TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = num_epoch): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    epoch_losses = []
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        loss = np.mean(running_loss)
        epoch_losses.append(loss)
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
         # Plot Loss vs. Epoch
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epoch + 1), epoch_losses, marker='o')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    all_pred = []
    all_labels = []
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            preds = torch.argmax(pred, dim=1)
            all_pred.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))

       # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(40))
    disp.plot(cmap="viridis", xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()
    return acc


if __name__ == '__main__': 
    
    train(model, trainloader, num_epoch)
    print("Evaluate on validation set")
    evaluate(model, valloader)
    print("Evaluate on test set")
    evaluate(model, testloader)

