import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform

        # 이미지 파일 목록 생성
        self.content_images = [f"{i}.jpg" for i in range(10000)]
        self.style_images = [f"{i}.jpg" for i in range(1119)]

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        # 콘텐츠 이미지 로드
        content_path = os.path.join(self.content_dir, self.content_images[idx])
        content_img = Image.open(content_path).convert('RGB')

        # 스타일 이미지를 랜덤하게 선택
        style_idx = np.random.randint(0, len(self.style_images))
        style_path = os.path.join(self.style_dir, self.style_images[style_idx])
        style_img = Image.open(style_path).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)

        return content_img, style_img


# VGG Encoder 정의 (ReLU4_1까지)
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features

        # ReLU4_1까지의 레이어 추출 (index 21까지)
        self.encoder = nn.Sequential()
        for i in range(22):  # 0~21 (ReLU4_1까지)
            self.encoder.add_module(str(i), vgg[i])

        # 파라미터 고정
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


# Decoder 정의
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Decoder는 Encoder와 대칭적으로 구성
        self.decoder = nn.Sequential(
            # ReLU4_1 -> ReLU3_4
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # ReLU3_4 -> ReLU3_3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # ReLU3_3 -> ReLU3_2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # ReLU3_2 -> ReLU3_1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # ReLU3_1 -> ReLU2_2
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # ReLU2_2 -> ReLU2_1
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # ReLU2_1 -> ReLU1_2
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # ReLU1_2 -> ReLU1_1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 최종 출력 레이어
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


# AdaIN 함수
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# 손실 함수들
def content_loss(generated_features, target_features):
    return F.mse_loss(generated_features, target_features)


def style_loss(generated_features, style_features):
    generated_mean, generated_std = calc_mean_std(generated_features)
    style_mean, style_std = calc_mean_std(style_features)
    return F.mse_loss(generated_mean, style_mean) + F.mse_loss(generated_std, style_std)


# 학습 설정
def train_model():
    # 하이퍼파라미터
    batch_size = 8  # RTX 3080에 적합한 배치 크기
    learning_rate = 1e-4
    num_epochs = 50
    content_weight = 1.0
    style_weight = 10.0

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 및 데이터로더
    dataset = StyleTransferDataset(
        content_dir='image/content_image',
        style_dir='image/style_image',
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 모델 초기화
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)

    # 옵티마이저
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # 학습 루프
    encoder.eval()
    decoder.train()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (content_imgs, style_imgs) in enumerate(progress_bar):
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)

            # Forward pass
            with torch.no_grad():
                content_features = encoder(content_imgs)
                style_features = encoder(style_imgs)

            # AdaIN
            target_features = adaptive_instance_normalization(content_features, style_features)

            # Decoder
            generated_imgs = decoder(target_features)

            # 생성된 이미지의 특징 추출
            generated_features = encoder(generated_imgs)

            # 손실 계산
            c_loss = content_loss(generated_features, target_features)
            s_loss = style_loss(generated_features, style_features)

            total_loss = content_weight * c_loss + style_weight * s_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            if batch_idx % 125 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Content': f'{c_loss.item():.4f}',
                    'Style': f'{s_loss.item():.4f}'
                })

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # 체크포인트 저장 (매 10 에포크마다)
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'model_save/checkpoint_epoch_{epoch + 1}.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')

    # 최종 모델 저장
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'losses': losses
    }, 'model_save/adain_model.pth')

    print('Training completed! Model saved as model_save/adain_model.pth')

    # 손실 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()


if __name__ == '__main__':
    print("Starting AdaIN Neural Style Transfer Training...")
    print(f"Using device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    train_model()