# Bộ ảnh chử viết tay của tensorflow
# https://www.tensorflow.org/quantum/tutorials/mnist
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# chuyển đổi dữ liệu 3D sang  2D
nsamples, nx, ny = X_train.shape
d2_X_train = X_train.reshape((nsamples,nx*ny))
nsamples1, nx1, ny1 = X_test.shape
d2_X_test = X_test.reshape((nsamples1,nx1*ny1))
# Nạp thư viện autosklearn
import autosklearn.classification
import sklearn
# khởi tạo auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=900, # điều chỉnh thời gian huấn luyện
          per_run_time_limit=30, # dành tối đa 30 giây cho mỗi mô hình đào tạo
          )

# huấn luyện mô hình
automl.fit(d2_X_train, y_train)

# evaluate
y_hat = automl.predict(d2_X_test)
test_acc = sklearn.metrics.accuracy_score(y_test, y_hat)
print("Test Accuracy score {0}".format(test_acc))
