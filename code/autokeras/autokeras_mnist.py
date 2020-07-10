# Bộ ảnh chử viết tay của tensorflow
# https://www.tensorflow.org/quantum/tutorials/mnist
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import thư viện autokeras
import autokeras as ak
# Khởi tạo trình phân loại dữ liệu có cấu trúc
clf = ak.ImageClassifier(max_trials=1)
# huấn luyện mô hình với dữ liệu đào tạo
clf.fit(x_train, y_train)
# Dự đoán với mô hình tốt nhất.
predicted_y = clf.predict(x_test)
# Đánh giá mô hình tốt nhất với dữ liệu thử nghiệm.
print(clf.evaluate(x_test, y_test))
