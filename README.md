# py-AdaIN
AdaIN은 Style Transfer의 일종으로 콘텐츠 이미지를 스타일 이미지의 스타일로 재구성하는 기술이다.
<br />(Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization)
<br />
<br />AdaIN은 기본적으로 Encoder, AdaIN Layer, Decoder로 구성되어 있다.

![structure](https://github.com/user-attachments/assets/bfcf1400-d075-4d32-956c-9806cfb19877)

<br />그림에서 왼쪽 위의 이미지를 콘텐츠 이미지, 왼쪽 아래의 이미지를 스타일 이미지라고 한다.
<br />콘텐츠 이미지는 스타일을 입힐 대상이고 스타일 이미지는 스타일을 추출할 대상이다.
<br />즉, 콘텐츠 이미지를 강아지 사진으로 하고 스타일 이미지를 반 고흐의 그림으로 한다면,
<br />반 고흐의 화풍으로 그려진 강아지 사진이 결과물이 된다.
<br />
<br />이를 위해 딥러닝 모델을 통한 처리 과정이 필요하다.
<br />먼저 VGG Encoder은 입력된 이미지들에서 특징을 추출한다.
<br />이 때, VGG는 Visual Geometry Group의 약자로 CNN 구조의 사전 학습된 모델을 사용한다.
<br />즉, AdaIN에서 Encoder은 학습을 시킬 필요가 없다. (ImageNet등에서 훌륭한 모델을 제공함)
<br />VGG Encoder을 거친 이미지들은 3차원 텐서 형태의 feature map이 된다.
<br />
<br />다음으로 AdaIN Layer에서는 VGG Encoder에서 추출한 콘텐츠/스타일 이미지의 정보를 기반으로 새로운 feature map을 구성한다.

$$\textrm{AdaIN}(x,y)=\sigma (y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)$$

이는 AdaIN Layer의 핵심 수식으로 $x$는 콘텐츠 특징, $y$는 스타일 특징을 뜻한다.
<br />출력값은 콘텐츠 특징의 평균($\mu$)과 분산($\sigma ^2$)을 스타일 특징의 평균과 분산으로 조정한다.
<br />$\left ( \frac{x-\mu(x)}{\sigma(x)} \right )$ 는 콘텐츠 이미지에서 콘텐츠 이미지의 스타일을 빼준 것이고,
<br />$\sigma (y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)$ 는 이에 스타일 이미지의 스타일을 입혀준 것이다.


https://drive.google.com/file/d/1TAk9eLtbAq0AFuak8GuTQfJa6zWB81Ib/view?usp=sharing
