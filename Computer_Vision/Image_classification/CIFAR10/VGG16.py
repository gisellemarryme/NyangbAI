import os
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 1. GPU 사용 확인
is_cuda=torch.cuda.is_available()
device=torch.device('cuda' if is_cuda else 'cpu')
print('current device is:',device)

# 2. 데이터셋 로딩 및 전처리, 이미지 크기 확인
batch_size=32
learning_rate=0.001
epoch_num=10

# 데이터 증강 포함한 전처리 (학습 데이터에만 데이터 증강)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # 이미지 크기 변경 및 비율 무작위
    transforms.RandomHorizontalFlip(),  # 수평 뒤집기
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균과 표준편차 정규화
])

# 테스트 데이터는 데이터 증강 없이 전처리
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data=torchvision.datasets.CIFAR10(root='./',
                                      train=True,
                                      download=True,
                                      transform=train_transform)

test_data=torchvision.datasets.CIFAR10(root='./',
                                      train=False,
                                      download=True,
                                      transform=test_transform)
image,label=train_data[0]
print(f'data size: {image.size()}') #data size: torch.Size([3, 32, 32]) -> RGB, 32*32 이미지인 것을 확인 가능함

train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=batch_size,
                                         shuffle=True)

# 3. 사전 학습 모델 로드 및 파라미터 설정
model=models.vgg16(pretrained=True) #사전 학습 모델(vgg16) 불러옴

# 모델 파라미터 고정(특징 추출기 파라미터 고정)
def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

set_parameter_requires_grad(model) #vgg16 모델 특징 추출기 고정

model.classifier[6] = nn.Linear(4096, 10) #vgg16 분류기 중 6번째 레이어 -> fc 레이어 수정(10개 클래스로)

# fc 레이어만 학습할 수 있도록
for param in model.classifier[6].parameters():
    param.requires_grad = True #고정 아님

model=model.to(device)
optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)
criterion=torch.nn.CrossEntropyLoss()
print(model) #모델 구조 출력

# 4. 모델 학습 및 평가
best_acc=0.0

for epoch in range(epoch_num):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    # 훈련 데이터셋에 대해 학습
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) 
        loss.backward()
        optimizer.step()

        running_loss+=loss.item() #에폭 loss 합계 갱신

    # 에폭별 평균 loss 출력
    avg_loss=running_loss/len(train_loader)
    print('Epoch {}/{}, Train Loss: {:.4f}'.format(epoch+1,epoch_num,avg_loss))

    # 테스트 데이터셋에 대해 정확도 계산
    model.eval()
    with torch.no_grad():
        for data,target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            prediction=output.data.max(1)[1] #output에서 가장 높은 값을 가진 클래스의 인덱스 선택  
            correct+=prediction.eq(target.data).sum().item()
            total+=target.size(0)

    epoch_acc=100.*correct/total
    print('Epoch {}/{}, Test Accuracy: {:4f}'.format(epoch+1,epoch_num,epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc

        torch.save(model.state_dict(), os.path.join('./03_pth/', f'best_model{epoch+1}_{epoch_acc:.4f}.pth'))

print('Best Test Acc: {:4f}'.format(best_acc)) # Best Test Acc: 58.340000

# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )
# Epoch 1/10, Train Loss: 1.6175
# Epoch 1/10, Test Accuracy: 56.360000
# Epoch 2/10, Train Loss: 1.6001
# Epoch 2/10, Test Accuracy: 58.340000
# Epoch 3/10, Train Loss: 1.6029
# Epoch 3/10, Test Accuracy: 57.830000
# Epoch 4/10, Train Loss: 1.6044
# Epoch 4/10, Test Accuracy: 57.270000
# Epoch 5/10, Train Loss: 1.6039
# Epoch 5/10, Test Accuracy: 57.240000
# Epoch 6/10, Train Loss: 1.6151
# Epoch 6/10, Test Accuracy: 58.210000
# Epoch 7/10, Train Loss: 1.6044
# Epoch 7/10, Test Accuracy: 57.070000
# Epoch 8/10, Train Loss: 1.6163
# Epoch 8/10, Test Accuracy: 57.070000
# Epoch 9/10, Train Loss: 1.6034
# Epoch 9/10, Test Accuracy: 57.490000
# Epoch 10/10, Train Loss: 1.6137
# Epoch 10/10, Test Accuracy: 57.390000
# Best Test Acc: 58.340000
