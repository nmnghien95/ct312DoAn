# Tập dữ liệu hoa Iris của sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
import sklearn.model_selection
import sklearn.datasets
X, y = sklearn.datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
# Sử dụng máy học tự động
from autoPyTorch import AutoNetClassification
# khởi tạo autopytorch
autoPyTorch = AutoNetClassification("tiny_cs", log_level='info', max_runtime=900, min_budget=30, max_budget=90, cuda=True, use_pynisher=False)
# Huấn luyện với tập dữ liệu
autoPyTorch.fit(x_train, y_train, 
                validation_split=0.3)

# Kiểm tra độ chính xác với dữ liệu kiểm thử
y_pred = autoPyTorch.predict(x_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
