
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



# Load the dataset and train, val, test splits
print("Loading datasets...")

IMAGE_SIZE = 64
BATCH_SIZE = 64

BUTTERFLY_train_nonnorm = datasets.ImageFolder(root="./Documents/cs583_final_project/train/", transform=transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]))

#Calculating mean and std of training set
meanloader = DataLoader(BUTTERFLY_train_nonnorm, batch_size=len(BUTTERFLY_train_nonnorm))
num_images = 0
mean = 0
std = 0
for images, _ in meanloader:
    # Flatten the images and calculate the mean and std for each channel (R, G, B)
    batch_samples = images.size(0)  # Number of images in the batch
    images = images.view(batch_samples, images.size(1), -1)  # Flatten the images for each channel
    
    mean += images.mean(dim=2).sum(dim=0)  # Sum the means for each channel
    std += images.std(dim=2).sum(dim=0)  # Sum the stds for each channel
    num_images += batch_samples  # Total number of samples processed

mean /= num_images
std /= num_images

BUTTERFLY_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),          # Transform from [0,255] uint8 to [0,1] float
    transforms.Normalize(mean, std)
])

BUTTERFLY_train = datasets.ImageFolder(root="./Documents/cs583_final_project/train/", transform=BUTTERFLY_transform)
BUTTERFLY_val = datasets.ImageFolder(root="./Documents/cs583_final_project/valid/", transform=BUTTERFLY_transform)
BUTTERFLY_test = datasets.ImageFolder(root="./Documents/cs583_final_project/test/", transform=BUTTERFLY_transform)

print("Done!")

# Create dataloaders
trainloader = DataLoader(BUTTERFLY_train, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(BUTTERFLY_val, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(BUTTERFLY_test, batch_size=BATCH_SIZE, shuffle=True)

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
    plt.title("Epochs vs Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    topFiveCorrect = 0
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

            # top 5 choice correct
            softmax = torch.softmax(pred,dim=1)
            # values, indices of top k values
            topkvals, topkindices = torch.topk(softmax, 5)
            topFiveCorrect += torch.isin(label, topkindices).sum().item()

    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))

    topFiveAcc = topFiveCorrect/len(loader.dataset)
    print("Top 5 Evaluation accuracy: {}".format(topFiveAcc))

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

