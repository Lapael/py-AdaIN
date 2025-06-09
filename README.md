# py-AdaIN
AdaIN은 Style Transfer의 일종으로 콘텐츠 이미지를 스타일 이미지의 스타일로 재구성하는 기술이다.
<br />(Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization)
<br />
<br />AdaIN은 기본적으로 Encoder, AdaIN Layer, Decoder로 구성되어 있다.

![structure](https://github.com/user-attachments/assets/bfcf1400-d075-4d32-956c-9806cfb19877)


$$\textrm{AdaIN}(x,y)=\sigma (y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)$$

여기서 $x$는 콘텐츠 특징, $y$는 스타일 특징으로, 콘텐츠 특징의 평균과 분산을 스타일 특징의 평균과 분산으로 조정한다.


https://drive.google.com/file/d/1TAk9eLtbAq0AFuak8GuTQfJa6zWB81Ib/view?usp=sharing
