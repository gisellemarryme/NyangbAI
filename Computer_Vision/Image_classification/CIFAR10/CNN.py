import torch
import torchvision
import os

# 1. GPU 사용 확인
is_cuda=torch.cuda.is_available()
device=torch.device('cuda' if is_cuda else 'cpu')
print('current device is:',device)

# 2. 데이터셋 로딩 및 전처리, 이미지 크기 확인
batch_size=32
learning_rate=0.0001
epoch_num=10

cifar10_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data=torchvision.datasets.CIFAR10(root='./',
                                      train=True,
                                      download=True,
                                      transform=cifar10_transform)

test_data=torchvision.datasets.CIFAR10(root='./',
                                      train=False,
                                      download=True,
                                      transform=cifar10_transform)
image,label=train_data[0]
print(f'data size: {image.size()}') #data size: torch.Size([3, 32, 32]) -> RGB, 32*32 이미지인 것을 확인 가능함

train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=batch_size,
                                         shuffle=True)
# 3. CNN 모델 설계
class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    self.conv1=torch.nn.Conv2d(3,16,kernel_size=3,padding=1) #3x32x32 -> 16x32x32, kernel=3x3, padding=1, stride=1

    self.pool1=torch.nn.MaxPool2d(kernel_size=2,stride=2) #Max pooling 거치면 16x32x32 -> 16x16x16

    self.fc1=torch.nn.Linear(16*16*16,64) #16*16*16 -> 64

    self.fc2=torch.nn.Linear(64,10) #64 -> 10

    self.dropout = torch.nn.Dropout(0.5)

  def forward(self,x):
    x=self.pool1(torch.nn.functional.relu(self.conv1(x))) #conv1 layer 통과 -> ReLU 적용 -> Max Pooling layer 통과 >> 32x14x14
   
    x=x.view(x.size(0),-1) #Flatten 적용 >> 32x8x8, -1의 의미는: batch size에 맞게 3126 들어가는데 batch size는 계속 변동 -> batch size에 맞게 유연하게 넣기 위해

    x=torch.nn.functional.relu(self.fc1(x)) #fc1 layer 통과 -> ReLU 적용 >> 32*8*8 -> 128
    x = self.dropout(x)  # Dropout 적용
    x=self.fc2(x) #fc2 layer 통과 >> 64 -> 10

    return x

# 4. 모델 초기화, 손실 함수 및 옵티마이저 설정
model=CNN().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion=torch.nn.CrossEntropyLoss()
print(model) #모델 구조 확인

# 5. 모델 학습 및 평가
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
            prediction=output.data.max(1)[1] #output에서 가장 높은 값을 가진 클래스의 인덱스 선택 -> 
            correct+=prediction.eq(target.data).sum().item()
            total+=target.size(0)

    epoch_acc=100.*correct/total
    print('Epoch {}/{}, Test Accuracy: {:4f}'.format(epoch+1,epoch_num,epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc

        torch.save(model.state_dict(), os.path.join('./02_pth/', f'best_model{epoch+1}_{epoch_acc:.4f}.pth'))

print('Best Test Acc: {:4f}'.format(best_acc)) # Best Test Acc: 56.260000

# CNN(
#   (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (fc1): Linear(in_features=4096, out_features=64, bias=True)
#   (fc2): Linear(in_features=64, out_features=10, bias=True)
#   (dropout): Dropout(p=0.5, inplace=False)
# )
# Epoch 1/10, Train Loss: 1.9140
# Epoch 1/10, Test Accuracy: 43.050000
# Epoch 2/10, Train Loss: 1.6904
# Epoch 2/10, Test Accuracy: 48.350000
# Epoch 3/10, Train Loss: 1.5890
# Epoch 3/10, Test Accuracy: 49.670000
# Epoch 4/10, Train Loss: 1.5252
# Epoch 4/10, Test Accuracy: 51.550000
# Epoch 5/10, Train Loss: 1.4748
# Epoch 5/10, Test Accuracy: 52.560000
# Epoch 6/10, Train Loss: 1.4392
# Epoch 6/10, Test Accuracy: 53.560000
# Epoch 7/10, Train Loss: 1.4119
# Epoch 7/10, Test Accuracy: 54.890000
# Epoch 8/10, Train Loss: 1.3873
# Epoch 8/10, Test Accuracy: 55.130000
# Epoch 9/10, Train Loss: 1.3670
# Epoch 9/10, Test Accuracy: 55.820000
# Epoch 10/10, Train Loss: 1.3459
# Epoch 10/10, Test Accuracy: 56.260000
# Best Test Acc: 56.260000
