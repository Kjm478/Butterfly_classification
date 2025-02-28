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

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = 40
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(2*2*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        # 64 Block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 128 Block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 256 Block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 512 Block
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # 512 Block # 2
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # FC Layers
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = VGG11().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
optimizer = optim.Adam(model.parameters(), lr=10**-4, weight_decay=5*(10**-4)) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 70
lossArray = []

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
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
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        lossArray.append(np.mean(running_loss))
    print("Done!")

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

            # Confusion Matrix
            all_pred.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # top choice correct
            correct += (torch.argmax(pred,dim=1)==label).sum().item()

            # top 5 choice correct
            softmax = torch.softmax(pred,dim=1)
            # values, indices of top k values
            topkvals, topkindices = torch.topk(softmax, 5)
            topFiveCorrect += torch.isin(label, topkindices).sum().item()

    acc = correct/len(loader.dataset)
    topFiveAcc = topFiveCorrect/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    print("Top 5 Evaluation accuracy: {}".format(topFiveAcc))

       # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(40))
    disp.plot(cmap="viridis", xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()
    return acc
    
train(model, trainloader, num_epoch)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("\nEvaluate on test set")
evaluate(model, testloader)

epochArray = np.arange(1, num_epoch + 1, 1)

plt.figure(1)
plt.title("Epochs vs Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.plot(epochArray, lossArray, marker = 'o')
plt.show()