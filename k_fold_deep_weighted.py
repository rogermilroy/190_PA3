
from deep_cnn import DeepCNN
import torch
from torch import optim
from torch.nn import functional
import testing
from preprocessed_dataloader import *

# Setup: initialize the hyperparameters/variables
num_epochs = 10  # Number of full passes through the dataset
early_stop_epochs = 10
batch_size = 128  # Number of samples in each minibatch
learning_rate = 0.001
seed = np.random.seed(42)  # Seed the random number generator for reproducibility
p_test = 0.1  # Percent of the overall dataset to reserve for testing
num_folds = 4
results_dir = './results/kfold-deep-weighted'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 3, "pin_memory": True}
    print("CUDA is supported")
else:  # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")



# Instantiate a DeepCNN to run on the GPU or CPU based on CUDA support
model = DeepCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

total_samples = 112120.0

frequencies = torch.tensor([11535.0, 2772.0, 13307.0, 19870.0, 5746.0, 6323.0, 1353.0, 5298.0 ,
                            4667.0, 2303.0, 2516.0, 1686.0, 3385.0, 227.0])

weights = frequencies / total_samples

# Use bce with logits for additional numerical stability.
criterion = functional.binary_cross_entropy_with_logits(weight=weights)


# Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters())


for i in range(num_folds):
    trace_file = results_dir + '/trace-' + str(i)
    val_file = results_dir + '/val-' + str(i)
    test_file = results_dir + '/test-' + str(i)
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = processed_split_loaders(num_folds, i, batch_size,
                                                                    seed,
                                                                    p_test=p_test,
                                                                    shuffle=True,
                                                                    extras=extras)
    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    best_params = None

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

            if minibatch_count % N == 0 and minibatch_count != 0:
                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                with open(trace_file, 'a+') as f:
                    f.write(str(epoch + 1) + ',' + str(minibatch_count) + ',' +
                            str(N_minibatch_loss) + '\n')

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

            # validate every 4 N minibatches. as validation more expensive now.
            if minibatch_count % (4 * N) == 0 and minibatch_count != 0:

                # validation
                total_val_loss, avg_val_loss, accuracy, precision, recall, balance, \
                conf = testing.test(
                    model,
                    computing_device,
                    val_loader,
                    criterion)
                if total_val_loss < current_best_val:
                    current_best_val = total_val_loss
                    best_params = model.state_dict()
                    increasing_epochs = 0
                else:
                    increasing_epochs += 1
                with open(val_file, 'a+') as f1:
                    f1.write(str(total_val_loss) + ',' + str(avg_val_loss) + ',' + str(accuracy)
                             + ',' + str(precision) + ',' + str(recall) + ',' + str(balance)
                             + '\n')
                torch.save(conf, val_file + '-conf' + str(epoch) + '-' + str(minibatch_count))
                if increasing_epochs > early_stop_epochs:
                    break

    if best_params is not None:
        model.load_state_dict(best_params)
    # test
    total_test_loss, avg_test_loss, tacc, tpr, tre, tbal , tconf = testing.test(model,
                                                                             computing_device,
                                                                        test_loader, criterion)

    with open(test_file, 'a+') as f2:
        f2.write(str(epoch) + ',' + str(total_loss) + ',' + str(avg_minibatch_loss) + ',' +
                 str(total_test_loss) + ',' + str(avg_test_loss) + ',' + str(tacc) + ',' + str(tpr)
                 + ',' + str(tre) + ',' + str(tbal) + '\n')
    torch.save(tconf, test_file + '-test-conf')

