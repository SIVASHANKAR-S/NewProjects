# This train.py used to train a new network on a dataset and save the model as checkpoint
# code below are directly copied from My jupyter notebook implementation of Neuralnetwork to predict the image on flower dataset
# Those jupyter notebook implementation of codes are created for Udacity Nanodegree program project
# Thanks to Udacity, currently I can train a neural nertwork by own and identify/classifying images with my model

from linkfile import *
from getinput import *

# Necessary datasets, transforms and input commands imorted from the supporting files share & getinput

# Setting default values if user didnt provide it(These defaults values feeded based on the instruction given in imageClassifier part on project assignment)
# vgg13 - inputsize and output size
if (arch == 'vgg13'):
    input_size , output_size = 25088 , 102
elif (arch == "resnet18"):
    input_size , output_size = 2048 , 500
elif (arch == "densenet121"):
    input_size, output_size = 1024 , 102
else:
    print("kindly select model architecture 'vgg13' or 'resnet18' or 'densenet121'")
    exit()

# if user didnt provide learning rate,then default to 0.01(as per the instruction)
if learning_rate == None:
    learning_rate = 0.01
else:
    learning_rate = float(learning_rate)

# Hidden layer for the vgg13
if hidden_units is None:
    if (arch == 'vgg13'):
        hidden_units = 512 # Default value as per the instruction in pre worksapce
    elif (arch == "resnet18"):
        hidden_units = 500 # Default value fixing
    elif (arch == "densenet121"):
        hidden_units = 400 # Default value fixing
else:
    hidden_units = int(hidden_units)
# epoch for the network
if epochs is None:
    epochs = 20
else:
    epochs = int(epochs)
# If User didnt specify the gpu, the default to cpu
if device is None:
    device = "cpu"
# Save directory
if save_dir is None:
    save_dir = "checkpoint.pth"

# Defining dataloaders with help of Imagesets and transforms
# Training image transformation
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
# Validation image transformation
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
# Training image transformation
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)

# Switch between gpu cuda / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
if (arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
elif (arch == 'resnet18'):
        model = models.resnet18(pretrained=True)
elif (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
print(model)
# No need to apply autograd for vgg parameters
for param in model.parameters():
    param.requires_grad = False
# Defining classifier
classifier = nn.Sequential(OrderedDict([
                                       ('fc1', nn.Linear(input_size, hidden_units)),
                                       ('relu',nn.ReLU()),('dropout1',nn.Dropout(0.2)),
                                       ('fc2', nn.Linear(hidden_units, output_size)),
                                       ('output', nn.LogSoftmax(dim=1))
                                       ]))
model.classifier = classifier
#defining loss function_learned from previous training lessons
criterion = nn.NLLLoss()
# defining optimizer_learned from the previous training lessons
# only train the classifier paramter
optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)
# Model changes gpu/cpu
model.to(device)

# Taining the model
def Training_network(model,trainloaders,validloaders,epochs,print_every,criterion,optimizer,device = 'cpu'):
    # Changing gpu/cpu based on the availability of gpu
    model.to(device)
    # Raw input hyperparameters
    epochs = epochs
    # Initially step = 0
    steps = 0
    print_every = print_every
    for e in range(epochs):
        #initially running loss should be zero
        running_loss = 0
        # Accuracy counter
        accuracy = 0
        for (inputs,labels) in trainloaders:
            steps += 1
            # Moving input and label to default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # feedforward and backpropagation
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            #updating running loss
            running_loss += loss.item()
            if (steps % print_every) == 0:
                # getting data from the validation loop
                v_acc = validation(validloaders,device)
                # printing the results
                print("Epoch: {}/{}...".format(e+1,epochs),
                    "TrainLoss : {:.5f}".format(running_loss/print_every),
                    "Valid Loss : {:.5f}".format(running_loss/len(validloaders)),
                    "Validation Accuracy : {}".format(round(v_acc,5)))

# defining seperate validation function to avoid conflicts,if any             
def validation(valid_loader,device = 'cpu'):
    totalvalue = 0
    correctvalue = 0
    # Turning off the gradients to save the memory and computations
    with torch.no_grad():
        # for everyimgae in the validation image dataset
        for i in valid_loader:
            # images and labels
            images,labels = i[0].to(device),i[1].to(device)
            outputs = model(images)
            _,foundval = torch.max(outputs.data,1)
            totalvalue += labels.size(0)
            correctvalue += (foundval == labels).sum().item()
    return (correctvalue / totalvalue)
# Generating the output from the training network modules and printing it
Training_network(model,trainloaders,validloaders,5,20,criterion,optimizer,'cuda')

def test_accuracy(testloader,device ='cpu'):
    correctvalue = 0
    totalvalue = 0
    # Turning off the gradients to save the memory and computations
    with torch.no_grad():
        for i in testloader:
            images,labels = i[0].to(device),i[1].to(device)
            outputs = model(images)
            _,foundval = torch.max(outputs.data,1)
            totalvalue += labels.size(0)
            correctvalue += (foundval == labels).sum().item()
    # priniting the accuracy of the network
    print('Accuracy: {} %' .format(round(100 * correctvalue / totalvalue),5))
test_accuracy(testloaders,'cuda')

# Saving the module
model.class_to_idx = trainloaders.class_to_idx
model.cpu()
torch.save({'model': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
print("ModelSaving in :" + save_dir)






