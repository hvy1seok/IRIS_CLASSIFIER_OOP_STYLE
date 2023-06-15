# Iris 분류기

이 저장소에는 Iris 분류기 코드가 포함되어 있습니다. 이 분류기는 Iris 꽃의 꽃받침과 꽃잎의 측정값을 기반으로 꽃을 서로 다른 종으로 분류할 수 있는 머신러닝 모델입니다. 이 분류기는 K-최근접 이웃(K-Nearest Neighbors) 알고리즘을 사용하며 Iris 데이터셋을 활용합니다.

## 데이터셋

Iris 데이터셋은 머신러닝에서 인기 있는 데이터셋으로, 세피드 길이, 세피드 폭, 꽃잎 길이, 꽃잎 폭의 측정값을 포함하고 있으며 세 가지 다른 종류의 Iris 꽃인 세토사(Setosa), 버시컬러(Versicolor), 버지니카(Virginica)에 대한 정보를 담고 있습니다.

## 요구사항

다음 패키지들이 설치되어 있는지 확인하세요:

requirements
- NumPy
- Matplotlib
- scikit-learn

다음 명령을 사용하여 패키지를 설치할 수 있습니다:

```bash
pip install numpy matplotlib scikit-learn
```

## 사용법

1. 이 저장소를 클론합니다:

```bash
git clone https://github.com/hvy1seok/IRIS_CLASSIFIER_OOP_STYLE.git
```

2. 클론한 저장소로 이동합니다:

```bash
cd iris-classifier
```

3. `maison.py` 스크립트를 실행합니다:

```bash
python maison.py
```

이 스크립트는 다음과 같은 단계를 수행합니다:

- Iris 데이터셋을 로드합니다.
- 데이터셋을 훈련 세트와 테스트 세트로 분할합니다.
- K-최근접 이웃 분류기를 훈련시킵니다.
- 분류기를 테스트하고 테스트 세트의 예측값과 정확도를 출력합니다.
- 그리드 서치(Grid Search)와 교차 검증(Cross-Validation)을 사용하여 모델을 튜닝합니다.
- 최적의 매개변수와 튜닝 후의 테스트 세트 정확도를 출력합니다.
- 모델의 성능을 평가하기 위해 혼동 행렬(Confusion Matrix)을 그립니다.
- 각 클래스에 대한 정밀도(Precision), 재현율(Recall) 및 F1-점수(F1-Score)를 포함한 분류 보고서를 생성합니다.

## 기여하기

기여하지 마세요 기말고사 과제 코드입니다. PULL REQUEST에 응답하지 않습니다.

## 라이선스
x

## 소감
"기술의 진보는 추상화를 함유한다"라는 말을 어디선가 들었던 기억이 납니다.
OOP는 높은 생산성의 개발자가 되기 위해서 반드시 알아야하는 개념이지만 그 개념 자체가 추상적이라 진입장벽이 높다고 생각합니다. 이후에도 코딩을할떄 OOP의 개념을 항상 떠올리며 익숙해지기 위한 노력이 필수적으로 수반되어야 할것 같습니다.