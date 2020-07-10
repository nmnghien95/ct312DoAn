# Tập dữ liệu nhận dạng quang học của bộ dữ liệu chữ số viết tay của sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
import sklearn.model_selection
import sklearn.datasets
X, y = sklearn.datasets.load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

#import thư viện autokeras
import autokeras as ak
# Khởi tạo trình phân loại dữ liệu có cấu trúc
clf = ak.StructuredDataClassifier(max_trials=10)
# huấn luyện mô hình với dữ liệu đào tạo
clf.fit(x_train, y_train)
# Dự đoán với mô hình tốt nhất.
predicted_y = clf.predict(x_test)
# Đánh giá mô hình tốt nhất với dữ liệu thử nghiệm.
print(clf.evaluate(x_test, y_test))
