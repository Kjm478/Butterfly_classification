import os 
from PIL import Image 
import numpy as np 
import torch 



def loaddata(path): 
    data = []
    labels = []
    # making sure the consistent labering order 
    classes = sorted(os.listdir(path))
    for id , class_name in enumerate(classes): 
        if class_name == '.DS_Store':
            continue
        class_dir = os.path.join(path, class_name)
        for image_name in os.listdir(class_dir): 
            image_path = os.path.join(class_dir, image_name)
            with Image.open(image_path) as img: 
                    # resizing the image to 224 X 224
                    img = img.resize((128 , 128))
                    # normmalize pixcel values between 0 and 1 
                    img_array =np.array(img) / 255.0 
                    if img_array.shape == (128, 128, 3): 
                        data.append(img_array)
                        # assign class index as label
                        labels.append(id)

    return (torch.tensor(data, dtype=torch.float16 ) , torch.tensor(labels, dtype= torch.float16))
 
def apply_pca( data, num_components = 50): 
     data = data.to(device)
     mean = torch.mean(data, axis= 0)
     std = torch.std(data, dim = 0)
     data_centered = (data - mean) / std 

     #compute the covariance. 
     covarience = torch.matmul(data_centered.T, data_centered) / (data.size(0)-1)

     eig_values, eig_vector = torch.linalg.eigh(covarience)

     #sort the values 
     sorted_index = torch.argsort(eig_values)[:: -1]
     top_eig = eig_vector[: , sorted_index[:num_components]]

     # project the data onto the most relevant eig values 
     X = np.dot(data_centered , top_eig)

     return X, top_eig


# KNN 
def euclidean_distance(x1,x2): 
     return torch.sqrt(torch.sum((x1 - x2) **2 , axis = 1))

def knn(x_train, y_train, x_test, k): 
     predictions = []
     for test_point in x_val: 
          distance = euclidean_distance(x_train , test_point)

          # sort by distances and select the k closet points 
          k_index = torch.topk(distance, k , largest=False).indices
          k_nearest_labels = y_train[k_index]

          unique, counts = torch.unique(k_nearest_labels, return_counts=True)
          predictions.append(unique[np.argmax(counts)])
     return torch.stack(predictions)


if __name__ == '__main__': 
     # set device to Mps if available 
     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
     path_train = 'train'
     path_test = 'test'
     x_train , y_train = loaddata(path_train)
     x_val , y_val = loaddata(path_test)
     print(f'using device: {device}')
     

     # flatten the data for pca 
     x_train_flat = x_train.view(x_train.size(0) , -1)
     x_val_flat  = x_val.view(x_val.size(0), -1)

     #Apply PCA 
     x_train_pca = pca_components = apply_pca(x_train_flat)
     x_val_centered = x_val_flat - torch.mean(x_train_flat, dim = 0)
     x_val_pca = torch.matmul(x_val_centered, pca_components)

     #upload the data on KNN 
     y_pred = knn(x_train_pca, y_train , x_val_pca, k = 3)

     print(f"y_pred shape: {y_pred.shape}")
     print(f"y_val shape: {y_val.shape}")
     print(f"y_pred: {y_pred}")
     print(f"y_val: {y_val}")

     accuracy = np.mean(y_pred == y_val)
     print(f'Accuracy on PCA reduced Validation: {accuracy * 100: .2f}%')





