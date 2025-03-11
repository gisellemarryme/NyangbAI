import os
import torch
import torchvision
import torch.nn as nn

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
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #평균, 표준편차 0.5로 정규화
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

# 3. MLP 모델 설계(fc 3개, drop 2개)
class MLP(nn.Module):
  def __init__(self):
    super(MLP,self).__init__()

    self.fc1=nn.Linear(3*32*32,128) #
    self.drop1=nn.Dropout2d(0.25) #25% 확률로 가중치 0
    self.fc2=nn.Linear(128,64)
    self.drop2=nn.Dropout2d(0.25)
    self.fc3=nn.Linear(64,10)

  def forward(self,x):
    x=x.view(-1,3*32*32)
    x=torch.nn.functional.relu(self.fc1(x))
    x=self.drop1(x)
    x=torch.nn.functional.relu(self.fc2(x))
    x=self.drop2(x)
    x=self.fc3(x)
    return x

# 4. 모델 초기화, 손실 함수 및 옵티마이저 설정(L2 정규화 사용)
model=MLP().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
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
            prediction=output.data.max(1)[1] #output에서 가장 높은 값을 가진 클래스의 인덱스 선택
            correct+=prediction.eq(target.data).sum().item()
            total+=target.size(0)

    epoch_acc=100.*correct/total
    print('Epoch {}/{}, Test Accuracy: {:4f}'.format(epoch+1,epoch_num,epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc

        torch.save(model.state_dict(), os.path.join('./01_pth/', f'best_model{epoch+1}_{epoch_acc:.4f}.pth'))

print('Best Test Acc: {:4f}'.format(best_acc)) #Best Test Acc: 51.400000

# MLP(
#   (fc1): Linear(in_features=3072, out_features=128, bias=True)
#   (drop1): Dropout2d(p=0.25, inplace=False)
#   (fc2): Linear(in_features=128, out_features=64, bias=True)
#   (drop2): Dropout2d(p=0.25, inplace=False)
#   (fc3): Linear(in_features=64, out_features=10, bias=True)
# )
# Epoch 1/10, Train Loss: 1.8780
# Epoch 1/10, Test Accuracy: 42.320000
# Epoch 2/10, Train Loss: 1.6948
# Epoch 2/10, Test Accuracy: 45.470000
# Epoch 3/10, Train Loss: 1.6266
# Epoch 3/10, Test Accuracy: 46.990000
# Epoch 4/10, Train Loss: 1.5823
# Epoch 4/10, Test Accuracy: 47.940000
# Epoch 5/10, Train Loss: 1.5467
# Epoch 5/10, Test Accuracy: 49.160000
# Epoch 6/10, Train Loss: 1.5144
# Epoch 6/10, Test Accuracy: 49.930000
# Epoch 7/10, Train Loss: 1.4925
# Epoch 7/10, Test Accuracy: 50.460000
# Epoch 8/10, Train Loss: 1.4713
# Epoch 8/10, Test Accuracy: 50.760000
# Epoch 9/10, Train Loss: 1.4501
# Epoch 9/10, Test Accuracy: 51.400000
# Epoch 10/10, Train Loss: 1.4305
# Epoch 10/10, Test Accuracy: 51.350000
# Best Test Acc: 51.400000