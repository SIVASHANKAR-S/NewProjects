from linkfile import *
from getinput import *



if top_k is None:
    top_k = 5
    
if device is None:
    device = "cpu"

if category_names is None:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:
    print("'.json' file is Missing...Open Images with json file")
    exit()

def loading(checkpoint_path):
    check_path = torch.load(checkpoint_path)
    model.class_to_idx = check_path['class_to_idx']
    arch = "vgg13"
    if (arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
        input_size = 25088
        hidden_units = 512
        output_size = 102
    elif (arch == 'resnet18'):
        model = models.resnet18(pretrained=True)
        input_size = 2048
        hidden_units = 500
        output_size = 102
    elif (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_units = 400
        output_size = 102
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                                       ('fc1', nn.Linear(input_size, hidden_units)),
                                       ('relu',nn.ReLU()),('dropout1',nn.Dropout(0.2)),
                                       ('fc2', nn.Linear(hidden_units, output_size)),
                                       ('output', nn.LogSoftmax(dim=1))
                                       ]))
    model.classifier = classifier
    model.load_state_dict(check_path['state_dict'])
    return model, model_info
model, model_info = loading('checkpoint.pth')
print(model)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    x_image = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    convert_image = transform_image(x_image)
    np_image = np.array(convert_image)
    np_image = np_image.transpose((2, 0, 1))
    return np_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    image = image.float()
    with torch.no_grad():
        if device == "cuda":
            output = model.forward(image.cuda())
        elif device == "cpu":
            output = model.forward(image.cpu())
    model, _ = loading(model)
    prob = torch.exp(model(image))
    prob_val, class_img = prob.topk(topk)
    return prob_val[0].tolist(), class_img[0].add(1).tolist()

def predict_image(image_path,model):
    prob_val, class_img = predict(image_path,'checkpoint.pth')
    plant_classes = [cat_to_name[str(cls)] + "({})".format(str(cls)) for cls in classes]
    x_image = Image.open(image_path)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(x_image);
    y_positions = np.arange(len(plant_classes))
    ax[1].barh(y_positions,probs,color='blue')
    ax[1].set_yticks(y_positions)
    ax[1].set_yticklabels(plant_classes)
    ax[1].invert_yaxis() 
image_path = 'flowers/train/1/image_06734.jpg'
predict_image(image,'checkpoint.pth')
imagedir = listdir('flowers/train/')
imagetype = random.choice(imagedir)
print('REAL_IMAGE /n CLASS = {} /n IMAGETYPE: {}'.format(imagetype,cat_to_name[imagetype]))
images = listdir('flowers/train/{}/'.format(imagetype))
image = random.choice(images)
image = 'flowers/train/{}/'.format(imagetype)+image
predict_image(image,'checkpoint.pth')
