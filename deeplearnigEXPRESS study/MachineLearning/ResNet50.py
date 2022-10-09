from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#기존 모델(ResNet50) 객체 생성
model=ResNet50(weights='imagenet')

#강아지 사진 전처리
img=image.load_img('dog.jpg', target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)

#기존 모델을 이용해 예측
preds=model.predict(x)
print('예측:', decode_predictions(preds, top=1)[0])