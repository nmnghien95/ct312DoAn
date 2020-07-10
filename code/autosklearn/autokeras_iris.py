# Tập dữ liệu hoa Iris của sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
import sklearn.model_selection
import sklearn.datasets
X, y = sklearn.datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
# Nạp thư viện autosklearn
import autosklearn.classification
import sklearn
# khởi tạo auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=900, # điều chỉnh thời gian huấn luyện
          per_run_time_limit=30, # dành tối đa 30 giây cho mỗi mô hình đào tạo
          )

# huấn luyện mô hình
automl.fit(x_train, y_train)

# evaluate
y_hat = automl.predict(x_test)
test_acc = sklearn.metrics.accuracy_score(y_test, y_hat)
print("Test Accuracy score {0}".format(test_acc))