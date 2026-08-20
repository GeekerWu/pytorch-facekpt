'''
2 数据预处理
为了防止神经网络过拟合训练数据集，我们需要随机变换数据集。我们将对训练和验证数据集应用以下操作：

由于人脸只占整个图像的一小部分，所以裁剪图像并仅使用人脸进行训练。

将裁剪后的人脸调整为（224x224）的图像。

随机改变调整后的人脸的亮度和饱和度。

在上述三个转换之后，随机旋转人脸。

将图像和关键点转换为torch张量，并在[-1, 1]之间进行归一化。
'''

class Transforms():
    def __init__(self):
        pass
    
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)
 
        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])
 
        image = imutils.rotate(np.array(image), angle)
 
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks
 
    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks
 
    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks
 
    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])
 
        image = TF.crop(image, top, left, height, width)
 
        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks
 
    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks
'''
3 数据集类
现在我们已经准备好了转换，让我们编写我们的数据集类。labels_ibug_300W_train.xml包含图像路径、关键点和边界框的坐标（用于裁剪人脸）。我们将这些值存储在列表中，以便在训练期间轻松访问。在本文章中，神经网络将在灰度图像上进行训练。
'''


class FaceLandmarksDataset(Dataset):
 
    def __init__(self, transform=None):
 
        tree = ET.parse('ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()
 
        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'ibug_300W_large_face_landmark_dataset'
        
        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
 
            self.crops.append(filename[0].attrib)
 
            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)
 
        self.landmarks = np.array(self.landmarks).astype('float32')     
 
        assert len(self.image_filenames) == len(self.landmarks)
 
    def __len__(self):
        return len(self.image_filenames)
 
    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        
        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])
 
        landmarks = landmarks - 0.5
 
        return image, landmarks
 
dataset = FaceLandmarksDataset(Transforms())

'''
4 神经网络
我们将使用ResNet18作为基本框架。我们需要修改第一层和最后一层以适应我们的目的。在第一层中，我们将输入通道数设为1，以便神经网络接受灰度图像。同样，在最后一层中，输出通道数应为68 * 2 = 136，以便模型预测每张人脸的68个关键点的（x，y）坐标。
'''
class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

'''
5 训练神经网络
我们将使用预测关键点和真实关键点之间的均方误差作为损失函数。请记住，要避免梯度爆炸，学习率应保持低。每当验证损失达到新的最小值时，网络权重将被保存。至少训练20个epochs以获得最佳性能。
'''
network = Network()
network.cuda()    
 
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)
 
loss_min = np.inf
num_epochs = 10
 
start_time = time.time()
for epoch in range(1,num_epochs+1):
    
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    
    network.train()
    for step in range(1,len(train_loader)+1):
    
        images, landmarks = next(iter(train_loader))
        
        images = images.cuda()
        landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
        
        predictions = network(images)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
        loss_train_step = criterion(predictions, landmarks)
        
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        running_loss = loss_train/step
        
        print_overwrite(step, len(train_loader), running_loss, 'train')
        
    network.eval() 
    with torch.no_grad():
        
        for step in range(1,len(valid_loader)+1):
            
            images, landmarks = next(iter(valid_loader))
        
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda()
        
            predictions = network(images)
 
            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)
 
            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step
 
            print_overwrite(step, len(valid_loader), running_loss, 'valid')
    
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    
    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')
    
    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), 'face_landmarks.pth') 
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')
     
print('Training Complete')