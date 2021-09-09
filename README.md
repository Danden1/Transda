# Transda

https://arxiv.org/abs/2105.14138

transformer(vit)를 domain adaptation 분야에 적용시킨 논문이다.

object에 대해 attention이 더 잘 일어날수록 정확도가 더 높아진다.

그리고 attention을 잘 하기 위해서 DINO 논문의 knowledge distillation 기법도 들고 왔다.

~~근데 이상하게도 내 코드로 target domain을 학습시키는데, 정확도가 너무 낮게 나온다.~~
2021.08.24 해결완료

|   | paper |  me |
|---|-------|-----|   
|AD | 97.2  |90.28|
|AW |  95   |94.79|
|WD | 99.6  |98.61|
|WA | 79.3  |70.59|
|DW | 99.3  |96.88|
|DA |73.7   |71.05|
