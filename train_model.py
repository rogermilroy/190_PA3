#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from baseline_cnn import *
from baseline_cnn import BasicCNN
import torch
from torchvision import transforms
from torch import optim
from torch.nn import functional
import testing


# Setup: initialize the hyperparameters/variables
num_epochs = 50           # Number of full passes through the dataset
early_stop_epochs = 5
batch_size = 32          # Number of samples in each minibatch
learning_rate = 0.001  
seed = np.random.seed(42)# Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

#TODO: Define the loss criterion and instantiate the gradient descent optimizer
criterion = functional.binary_cross_entropy #TODO - loss criteria are defined in the torch.nn package

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters()) #TODO - optimizers are defined in the torch.optim package


# In[ ]:


# Track the loss across training
total_loss = []
avg_minibatch_loss = []

# Begin training procedure
for epoch in range(num_epochs):

    N = 50
    N_minibatch_loss = 0.0
    current_best_val = 10000000.0
    increasing_epochs = 0

    # Get the next minibatch of images, labels for training
    torch.cuda.empty_cache()
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += float(loss)
        
        #TODO: Implement cross-validation
        
        if minibatch_count % N == 0:    
            
            # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0

    print("Finished", epoch + 1, "epochs of training")
    
    # validation
    accuracy, precision, recall, bcr = None, None, None, None
    total_val_loss = 0.0
    for i, (val_images, val_labels) in enumerate(val_loader):

        val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
        val_out = model(val_images)

        val_loss = criterion(val_out, val_labels)
        total_val_loss += float(val_loss)
        if i == 0:
            accuracy = torch.zeros_like(val_labels[0], dtype=torch.float)
            precision = torch.zeros_like(val_labels[0], dtype=torch.float)
            recall = torch.zeros_like(val_labels[0], dtype=torch.float)
            bcr = torch.zeros_like(val_labels[0], dtype=torch.float)
        accuracy += testing.accuracy(val_out, val_labels)
        precision += testing.precision(val_out, val_labels)
        recall += testing.recall(val_out, val_labels)
        bcr += testing.bcr(val_out, val_labels)

    if total_val_loss < current_best_val:
        current_best_val = total_val_loss
        increasing_epochs = 0
    else:
        increasing_epochs += 1
    if increasing_epochs > early_stop_epochs:
        break

    avg_val_loss = total_val_loss / float(i)
    accuracy /= float(i)
    precision /= float(i)
    recall /= float(i)
    bcr /= float(i)

    print(total_val_loss, avg_val_loss, accuracy, precision, recall, bcr)
    
print("Training complete after", epoch, "epochs, with total loss: ", total_loss, " and average "
                                                                                 "minibatch loss "
                                                                                 "of: ", avg_minibatch_loss)
# test
total_test_loss = 0.0
for j, (test_images, test_labels) in enumerate(test_loader):
    test_images, test_labels = test_images.to(computing_device), test_labels.to(computing_device)
    test_out = model(test_images)

    test_loss = criterion(test_out, test_labels)
    total_test_loss += test_loss
# taccuracy = testing.accuracy(test_out, test_labels)
# tprecision = testing.precision(test_out, test_labels)
# trecall = testing.recall(test_out, test_labels)
# tbcr = testing.bcr(test_out, test_labels)
# avg_test_loss = total_test_loss/float(j)
print(total_test_loss)
