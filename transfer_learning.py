#패키지 import
import argparse
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm           #progress를 보여준다
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import wandb

#라이브러리 충돌 억제 - 있어야 한다 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ImageTransform():
    """
    화상 전처리 클래스. 훈련시, 검증시의 동작이 다르다.
    화상 크기를 리사이즈하고, 색상을 표준화한다.
    훈련시에는 RandomResizedCrop과 RandomHorizontalFlip으로 데이터를 확장한다.

    Attributes
    ----------
    resize : int
        리사이즈 대상 화상의 크기.
    mean : (R, G, B)
        각 색상 채널의 평균 값.
    std : (R, G, B)
        각 색상 채널의 표준 편차.
    """
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(      #data augmentation
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(), #data augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean, std) #표준화
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),  #리사이즈   
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img)

def make_datapath_list(phase="train"):
    """
    데이터의 경로를 저장한 리스트를 작성한다.

    Parameters
    ----------
    phase : 'train' or 'val'
        훈련 데이터 또는 검증 데이터를 지정

    Returns
    -------
    path_list : list
        데이터 경로를 저장한 리스트
    """
    rootpath="./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')  #**은 임의의 디렉토리, *은 임의의 문자열
    print(target_path)

    path_list = []  #여기에 저장
    
    #glob를 이용해서 하위 디렉토리의 파일 경로를 가져온다.
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list

#개미와 벌의 화상에 대한 Dataset 작성
class HymenopteraDataset(data.Dataset):
    """
    개미와 벌 화상의 Dataset 클래스. PyTorch의 Dataset 클래스를 상속한다.

    Attributes
    ----------
    file_list : 리스트
        화상 경로를 저장한 리스트
    transform : object
        전처리 클래스의 인스턴스
    phase : 'train' or 'test'
        학습인지 훈련인지를 설정한다.
    """
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  #파일 경로 리스트
        self.transform = transform  #전처리 클래스의 인스턴스
        self.phase = phase  #train of val 지정

    
    def __len__(self):
        '''화상 개수를 반환'''
        return len(self.file_list)
    
    def __getitem__(self, index):
        ''''
        전처리한 화상의 텐서 형식의 데이터와 라벨 취득
        ''' 

        #index번째의 화상 로드
        img_path = self.file_list[index]
        img = Image.open(img_path)  #[높이][폭][색RGB]

        #화상의 전처리 실시
        img_transformed = self.transform(img, self.phase)   #torch.Size([3, 224, 224])

        #화상 라벨을 파일 이름에서 추출
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]
        
        #라벨을 숫자로 변경
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
        
        return img_transformed, label

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    #에폭 루프
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        #에폭별 학습 및 검증 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() #모델을 훈련 모드로
            else:
                net.eval()  #모델을 검증 모드로
            
            epoch_loss = 0.0    #에폭 손실 합
            epoch_corrects = 0  #에폭 정답 수

            # 학습하지 않을 시 검증 성능을 확인하기 위해 epoch=0의 훈련 생략
            if (epoch == 0) and (phase == 'train'):
                continue

            #데이터 로더로 미니 배치를 꺼내는 루프
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                #옵티마이저 초기화
                optimizer.zero_grad()

                #순전파 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)   #손실 계산
                    _, preds = torch.max(outputs, 1)    #라벨 예측 - 최댓값과 최댓값의 인덱스를 같이 출력한다

                    # 훈련 시에는 오차 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # 반복 결과 갱신
                    # 손실 합계 갱신
                    epoch_loss += loss.item() * inputs.size(0)
                    # 정답 수의 합계 갱신
                    epoch_corrects += torch.sum(preds == labels.data)
                    # 에폭당 손실과 정답률 표시
                    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                    epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
                    
                    if phase == 'train':
                        wandb.log({"Train_loss": epoch_loss})
                    else:
                        wandb.log({"Val_loss": epoch_loss, "Val_Acc": epoch_acc})
                    

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch_Size")
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser_args = parser.parse_args()
    
    # 난수 시드 설정
    torch.manual_seed(55)
    np.random.seed(55)
    random.seed(55)

    wandb.init(
        project="Pytorch_Advanced1 - VGG",
        entity="bc6817",
        name="VGG_Train",
        resume ="allow",
        mode="online",
    )

    wandb.run.name = 'VGG'
    wandb.run.save()

    # datapath 얻어 오기
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    #전처리 하이퍼 파라미터 
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    cfg = {
        "learning_rate": parser_args.learning_rate,
        "batch_size": parser_args.batch_size,
        "epochs": parser_args.epochs,
    }
    wandb.config.update(cfg)
    print(f'lr = {cfg["learning_rate"]}, batch = {cfg["batch_size"]}, epoch = {cfg["epochs"]}')


    # 2. 데이터셋 작성
    train_dataset = HymenopteraDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase = 'train')

    val_dataset = HymenopteraDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase = 'val')
    
  

    #3. 데이터 로더 작성
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    
    #사전형 변수에 정리
    dataloaders_dict = {"train" : train_dataloader, "val" : val_dataloader}

    
    '''#동작 확인 
    batch_iterator = iter(dataloaders_dict["train"])
    inputs, labels = next(batch_iterator)   #첫번째 요소 추출 - iterator로 하면 고차원도 쉽게 할 수 있다
    print(inputs.size())
    print(labels)'''

    # 4.네트워크 모델 작성
    # 학습된 VGG-16 모델 로드
    # VGG-16 모델의 인스턴스 생성
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)

    # 5. 순전파 정의
    # VGG16의 마지막 출력층의 출력 유닛을 개미와 벌인 2개로 변경
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    #훈련 모드로 설정 - net.train() / net.eval()은 아니다
    net.train()    
    print('네트워크 설정 완료: 학습된 가중치를 읽어들여 훈련 모드로 설정했습니다.')

    # 6. 손실함수 정의
    criterion = nn.CrossEntropyLoss()

    # 7. 최적화 기법 설정
    # 전이 학습에서 학습시킬 파라미터를 params_to_update 변수에 저장
    params_to_update = []

    # 학습시킬 파라미터명 - 전이학습이므로 마지막 계층만 학습시킨다
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]

    #학습시킬 파라미터 외에는 경사를 계산하지 않고 변하지 않도록 설정
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.required_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.require_grad = False  #Stop Gradient
    
    # params_to_update의 내용을 확인
    print("-----------")
    print(params_to_update)

    # 최적화 기법 설정
    optimizer = optim.SGD(params=params_to_update, lr = cfg["learning_rate"], momentum=0.9)

    # 8. 학습/검증 실시 
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=cfg["epochs"])

