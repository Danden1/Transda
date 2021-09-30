# Transda

https://arxiv.org/abs/2105.14138

transformer(vit)를 domain adaptation 분야에 적용시킨 논문이다.

object에 대해 attention이 더 잘 일어날수록 정확도가 더 높아진다.

그리고 attention을 잘 하기 위해서 DINO 논문의 knowledge distillation 기법도 들고 왔다.

~~근데 이상하게도 내 코드로 target domain을 학습시키는데, 정확도가 너무 낮게 나온다.~~
2021.08.24 해결완료

## test accuracy

|   | paper |  me |
|---|-------|-----|   
|AD | 97.2  |97.66|
|AW |  95   |95.80|
|WD | 99.6  |100.00|
|WA | 79.3  |80.82|
|DW | 99.3  |99.22|
|DA |73.7   |80.99|



## my accuracy

|   | train |  test |
|---|-------|-----|   
|AD | 80.58  |97.66|
|AW |  87.93|95.80|
|WD | 85.49 |100.00|
|WA | 82.93  |80.82|
|DW | 89.77  |99.22|
|DA |83.13  |80.99|

DA 할 때, train accuracy 가 test accuracy보다 낮게 나옴.

하지만, source domain 학습할 때는 그렇지 않음.

이 문제는 해결하지 못했음.
