# 패키지 import
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from utils.dataloader_image_classification import ImageTransform, make_datapath_list, HymenopteraDataset

from tqdm import tqdm

# 모델을 학습시키는 함수를 작성
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 초기 설정
    # GPU가 사용 가능한지 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치: ", device)

    # 네트워크를 GPU로
    net.to(device)

    # 네트워크가 어느 정도 고정되면, 고속화시킨다
    torch.backends.cudnn.benchmark = True

    # epoch 루프
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 훈련 및 검증 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 모델을 훈련 모드로
            else:
                net.eval()   # 모델을 검증 모드로

            epoch_loss = 0.0  # epoch 손실의 합
            epoch_corrects = 0  # epoch 정답수

            # 미학습시의 검증 성능을 확인하기 위해 epoch=0의 훈련은 생략
            if (epoch == 0) and (phase == 'train'):
                continue

            # 데이터 로더에서 미니 배치를 꺼내 루프
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPU가 사용 가능하면 GPU에 데이터 보내기
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer를 초기화
                optimizer.zero_grad()

                # 순전파(forward) 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 손실 계산
                    _, preds = torch.max(outputs, 1)  # 라벨 예측

                    # 훈련시에는 오차 역전파법
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 결과 계산
                    epoch_loss += loss.item() * inputs.size(0)  # loss의 합계를 갱신
                    # 정답 수의 합계를 갱신
                    epoch_corrects += torch.sum(preds == labels.data)

            # epoch별 loss와 정답률을 표시
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))



if __name__=="__main__":
    # 난수 시드 설정
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # 개미와 벌의 이미지 파일 경로 리스트를 작성
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    # 2. Dataset을 만든다
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = HymenopteraDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

    val_dataset = HymenopteraDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    # 3. DataLoader를 만든다
    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 사전 객체에 정리
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 4. 네트워크 모델 작성
    # 학습된 VGG-16 모델을 로드
    # VGG-16 모델의 인스턴스를 생성
    use_pretrained = True  # 학습된 파라미터를 사용
    net = models.vgg16(pretrained=use_pretrained)

    # VGG16의 마지막 출력층의 출력 유닛을 개미와 벌의 2개로 바꾸기
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # 훈련 모드로 설정
    net.train()

    print('네트워크 설정 완료: 학습된 가중치를 로드하고 훈련 모드로 설정했습니다')

    #5. 손실함수 정의
    criterion = nn.CrossEntropyLoss()

    #6. 최적화 방법 설정
    params_to_update1 = []
    params_to_update2 = []
    params_to_update3 = []
    
    #학습시킬 층의 파라미터 지정
    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    #파라미터를 각 리스트에 저장
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update1.append(param)
            print("params_to_update_1에 저장 : ", name)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update2.append(param)
            print("params_to_update_2에 저장: ", name)

        elif name in update_param_names_3:
                param.requires_grad = True
                params_to_update3.append(param)
                print("params_to_update_3에 저장 : ", name)

        else:
            param.requires_grad = False
            print("경사 계산 없음. 학습하지 않음: ", name)


        #최적화 방법 설정 - 뒤로 갈 수록 세진다
    optimizer = optim.SGD([
        {'params' : params_to_update1, 'lr':1e-4},
        {'params' : params_to_update2, 'lr':5e-4},     
        {'params' : params_to_update3, 'lr':1e-3}
    ], momentum=0.9)
        
    #7. 학습 및 검증 실행
    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    




















