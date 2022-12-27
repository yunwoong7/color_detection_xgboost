# color_detection_xgboost
지금은 Deep Learning을 기반으로 하는 Object Detection 알고리즘이 많고 성능 역시 뛰어나지만 더 간단하게 M/L 방법을 이용하여 객체를 찾고 분류하는 것이 가능합니다. 물론 정확도 측면에서는 부족하지만 속도면에서는 월등히 앞서고 쉽게 응용하여 다양한 시도를 할 수 있습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/lwy50/btrSkGbfV4m/nTP0ybovTFc8OlNbYPbQC1/img.png" width="50%">
</div>

[이전 글에서 소개했던 Color Detection](https://github.com/yunwoong7/color_detection) 을 이미지 연산이 아닌 XGBoost Model을 이용하여 수행하려고 합니다. 

------

### **1. XGBoost Classification Model 생성**

#### **XGBoost 패키지 설치**

```shell
pip install xgboost
```

#### **Import packages**

```python
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
```

```python
XGBoostError: 
XGBoost Library (libxgboost.dylib) could not be loaded.
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/Gij6z/btrSgaDJIGj/8oYYhdTYYnyPtKNrxcklmk/img.png" width="50%">
</div>

만일 위와 같은 오류가 발생한다면 아래 명령어를 수행 후 진행하시면 됩니다.

```shell
brew install libomp
```

#### **Load data**

이미지의 분류는 Red, Green, Blue 3가지 색상으로 분류하도록 합니다. 색상 정보는 HTML color 정보를 이용하여 CSV 파일로 만들었습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/GuLk4/btrSjua6TEG/Pa1LNRCkrYG3EGjmFzVMv0/img.png" width="50%">
</div>

```python
color = pd.read_csv("data/color.csv")
color.head()
```

| R    | G    | B    | COLOR      | CLASS |
| ---- | ---- | ---- | ---------- | ----- |
| 136  | 8    | 8    | Blood Red  | Red   |
| 170  | 74   | 68   | Brick Red  | Red   |
| 238  | 75   | 43   | Bright Red | Red   |
| 165  | 42   | 42   | Brown      | Red   |
| 128  | 0    | 32   | Burgundy   | Red   |
| ...  | ...  | ...  | ...        | ...   |

CSV 파일에서 R, G, B는 X값으로 CLASS는 Y값으로 사용합니다.

```python
cols = list(color.columns)
x_col = cols[:3]
y_col = cols[-1]
 
print('x colum : {}'.format(x_col))
print('y colum : {}'.format(y_col))
```

Output:

```shell
x colum : ['R', 'G', 'B']
y colum : CLASS
```

#### **Train / Test split**

```python
color_train, color_test = train_test_split(color, test_size=0.2, random_state=123)
print(color_train.shape, color_test.shape)
```

Output:

```shell
(103, 5) (26, 5)
```

#### **Model 생성**

```python
xgb_model = XGBClassifier(num_class=3,
                          n_estimators=500, 
                          learning_rate=0.2, 
                          max_depth=4,
                          eval_metric='mlogloss')
                          
xgb_model.fit(X=color_train[x_col], y=color_train[y_col])
```

#### **Model 평가**

```python
xgb_pred = xgb.predict(color_test[x_col])
 
y_pred = xgb.predict(color_test[x_col]) # 예측치
y_true = color_test[y_col]
acc = accuracy_score(y_true, y_pred)
 
print('accuracy : [{}]'.format(acc))
plot_importance(xgb)
```

Output:

```shell
accuracy : [1.0]
<AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/qgwxa/btrSm0sMbwO/OiXF23x3E5rozjR5L0wUwK/img.png" width="50%">
</div>

------

### **2. Color Detection using XGBoost Model**

위에서 만든 XGBoost Model 을 이용하여 이미지의 객체의 색상을 분류해보도록 하겠습니다.

#### **Import packages**

```python
import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

XGBoost Model을 이용하여 예측 Color값을 반환하는 Function 정의 (contour로 찾은 영역의 색상을 cv2.mean을 이용하여 평균 rgb 값을 구함)

```python
def color_label(image, c):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(image, mask=mask)[:3]
    
    rgb_value = pd.DataFrame({'R': [mean[2]], 'G': [mean[1]], 'B': [mean[0]]})
    xgb_pred = xgb.predict(rgb_value)
    
    return xgb_pred[0]
```

#### **Load Image**

```python
cv2_image = cv2.imread('asset/images/color.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bhCdG8/btrSkDS8hA2/KKSoH49BRobYqgqjPnMSa1/img.png" width="50%">
</div>

#### **Color Detection**

```python
resized = imutils.resize(cv2_image, width=640)
ratio = cv2_image.shape[0] / float(resized.shape[0])
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
 
img_show(['GaussianBlur', 'Threshold'], [blurred, thresh])
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/tkpDw/btrSeioWYMu/MnHX5nNwekxymsMJK9TGr0/img.png" width="50%">
</div>

```python
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 
vis = cv2_image.copy()
 
for c in cnts:
    # cv2.moments를 이용하여 객체의 중심을 계산
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    
    # 이미지에서 객체의 윤곽선과 Color를 표시
    color = color_label(resized, c)
 
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 10)
    cv2.circle(vis, (cX, cY), 20, (0, 255, 0), -1); 
    cv2.putText(vis, color, (cX-80, cY-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
```

Color를 표현한 이미지를 확인합니다.

```python
img_show('Color Detection', vis, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/dY3PfT/btrSmsiLxUb/aIJZ8d50V1S48QCuIzlQHk/img.png" width="50%">
</div>

------

이미지 연산만을 이용한 Color Detection 보다 정확도가 높습니다. Deep learning을 이용하면 이전에 labeling 된 데이타가 있어야 히지만 없는 경우가 있습니다. 그런 경우 이미지의 다른 정보를 활용할 수 있는데 그 중 하나가 색상 정보입니다.

XGBoost가 아닌 Random Forest를 이용해도 꽤 좋은 수준의 결과가 나옵니다. 다만 이미지의 음영이나 명암으로 이로 인해 밝은 부분의 픽셀과 어두운 부분의 색상이 잘못 인식 될 수 있지만 이 방법은 이미지 내에서 추출하고자 하는 대상의 구조나 모양에 영향을 받지 않기 때문에 Deep learning의 방식 보다 더 나은 품질의 결과를 가져오는 경우도 있습니다.
