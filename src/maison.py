아래는 주어진 코드를 데코레이터를 사용하여 리팩토링한 결과입니다:

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


def load_dataset(func):
    def wrapper(self, *args, **kwargs):
        self.iris = datasets.load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        return func(self, *args, **kwargs)

    return wrapper


def split_dataset(func):
    def wrapper(self, *args, **kwargs):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.X, self.y, test_size=0.2, shuffle=True
        )
        return func(self, *args, **kwargs)

    return wrapper


class IrisClassifier:
    def __init__(self):
        self.iris = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.knn = None
        self.grid = None

    @load_dataset
    def train(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.X_train, self.y_train)

    def test(self):
        y_pred = self.knn.predict(self.X_test)
        print("테스트 결과 예측값:\n", y_pred)
        print("테스트 세트 정확도: {:.2f}".format(np.mean(y_pred == self.y_test)))
        print("테스트 세트 정확도: {:.2f}".format(self.knn.score(self.X_test, self.y_test)))

    def tune(self):
        param_grid = {"n_neighbors": np.arange(1, 10)}
        self.grid = GridSearchCV(self.knn, param_grid, cv=5)
        self.grid.fit(self.X_train, self.y_train)
        print("최적의 교차 검증 점수: {:.2f}".format(self.grid.best_score_))
        print("최적의 매개변수: {}".format(self.grid.best_params_))
        print("테스트 세트 정확도: {:.2f}".format(self.grid.score(self.X_test, self.y_test)))

    def plot_confusion_matrix(self):
        y_pred = self.knn.predict(self.X_test)
        confusion = confusion_matrix(self.y_test, y_pred)
        print("혼동 행렬:\n{}".format(confusion))
        plt.imshow(confusion, cmap="Blues")
        plt.title("혼동 행렬")
        plt.colorbar()
        tick_marks = np.arange(len(self.iris.target_names))
        plt.xticks(tick_marks, self.iris.target_names, rotation=45)
        plt.yticks(tick_marks, self.iris.target_names)
        plt.xlabel("예측된 레이블")
        plt.ylabel("실제 레이블")
        plt.show()

    def generate_classification_report(self):
        y_pred = self.knn.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, target_names=self.iris.target_names)
        print("분류 보고서:\n", report)


def main():
    classifier = IrisClassifier