import os
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# 配置参数
data_dir = "HAM10000_images_part_1"
image_size = 128  
batch_size = 64
epochs = 2
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 自定义数据集
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        lr_image = transforms.Resize((image_size // 4, image_size // 4))(image)
        lr_image = transforms.Resize((image_size, image_size))(lr_image)
        
        return lr_image, image


dataset = MedicalImageDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义超分辨率模型 (EDSR)
class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.resblocks = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        res = x.clone()
        x = self.resblocks(x) + res
        x = self.conv2(x)
        return x

# 初始化模型
model = EDSR().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train():
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # 显示进度
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        loss_history.append(epoch_loss / len(dataloader))
    
    # 绘制损失曲线
    plt.figure()
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(range(1, epochs+1), loss_history, label='训练损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.show()


def evaluate():
    model.eval()
    psnr_values, ssim_values = [], []
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs).cpu().numpy()
            hr_imgs = hr_imgs.cpu().numpy()
            
            for i in range(len(outputs)):
                psnr_values.append(psnr(hr_imgs[i].transpose(1,2,0), outputs[i].transpose(1,2,0)))
                ssim_values.append(ssim(hr_imgs[i].transpose(1,2,0), outputs[i].transpose(1,2,0), multichannel=True))
    
    print(f"平均 PSNR: {np.mean(psnr_values):.2f}, 平均 SSIM: {np.mean(ssim_values):.4f}")
    
    plt.figure()
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(range(len(psnr_values)), psnr_values, label='PSNR')
    plt.xlabel('测试样本')
    plt.ylabel('PSNR')
    plt.title('PSNR 变化曲线')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
    evaluate()
