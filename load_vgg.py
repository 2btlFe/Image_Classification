#패키지 import 
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt 
#%matplotlib inline     #주피터 노트북에서 별도의 창없이 내부에서 그래프를 보여준다

import torch
import torchvision
from torchvision import models, transforms

#라이브러리 충돌 억제 - 있어야 한다 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class BaseTransform():
    """
    화상의 크기를 리사이즈하고 색상을 표준화한다.

    Attributes
    ---------
    resize : int
        리사이즈 대상 이미지의 크기.
    mean : (R, G, B)
        각 색상 채널의 평균 값.
    std :  (R, G, B)
        각 색상 채널의 표준 편차.
    """

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([  #데이터 전처리 과정 모으기
            transforms.Resize(resize),      # 짧은 변의 길이가 resize의 크기가 된다
            transforms.CenterCrop(resize),  # 이미지 중앙을 resize × resize으로 자르기
            transforms.ToTensor(),          #토치 텐서로 변환
            transforms.Normalize(mean, std) #색상 정보 표준화
        ])

    def __call__(self, img): 
        return self.base_transform(img)

class ILSVRC_class_predictor():
    """
    ILSVRC 데이터에 대한 모델의 출력에서 라벨을 구한다.

    Attributes
    ----------
    class_index : dictionary
            클래스 index와 라벨명을 대응시킨 사전형 변수.
    """

    def __init__(self, class_index):
        self.class_index = class_index
    
    def predict_max(self, out):
        """
        최대 확률의 ILSVRC 라벨명을 가져옵니다.

        Parameters
        ----------
        out : torch.Size([1, 1000])
            Net에서의 출력.

        Returns
        -------
        predicted_label_name : str
            가장 예측 확률이 높은 라벨명
        """
        maxid = np.argmax(out.detach().numpy()) #out값을 net에서 분리한다 - autograd 방지
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

if __name__ == "__main__":
    # VGG-16 모델의 인스턴스를 생성
    use_pretrained = True  # 학습된 파라미터를 사용
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()  # 추론 모드(평가 모드)로 설정

    # ILSVRC 라벨 정보를 읽어 사전형 변수를 생성합니다
    ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
    
    # ILSVRCPredictor 인스턴스 생성
    predictor = ILSVRC_class_predictor(ILSVRC_class_index)




    #화상 전처리 확인
    #1. 화상 읽기
    image_file_path = './data/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file_path)   #[높이][너비][색RGB]
    #print(img)  #아직 PIL이다

    #2. 원본 화상 표시
    plt.subplot(2, 1, 1)
    plt.imshow(img)

    # 3. 화상 전처리 및 처리된 이미지의 표시
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)    #ToTensor 덕분에 torch.Size([3, 224, 224])로 나타난다
    
    # (색상, 높이, 너비)을(높이, 너비, 색상)으로 변환하고 0-1로 값을 제한하여 표시
    #img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_display = img_transformed.permute(1, 2, 0)
    
    img_display = img_display.clamp(0, 1)
    #img_transformed = np.clip(img_transformed, 0, 1)

    #imshow에는 numpy만 들어갈 수 있다
    img_display = img_display.numpy()
    plt.subplot(2, 1, 2)
    plt.imshow(img_display)
    plt.show()


    #전처리 후 배치 크기의 차원 추가
    inputs = img_transformed.unsqueeze_(0)  #torch.Size([1, 3, 224, 224]) - 0번째 axis에 추가

    #모델에 입력하고 모델 출력을 라벨로 변환
    out = net(inputs)   #torch.Size([1, 1000])
    result = predictor.predict_max(out)

    #예측 결과 출력
    print(f"입력 화상의 예측 결과 : {result}")

