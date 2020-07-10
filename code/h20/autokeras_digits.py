# Tập dữ liệu nhận dạng quang học của bộ dữ liệu chữ số viết tay của sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
import sklearn.model_selection
import sklearn.datasets
X, y = sklearn.datasets.load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

#import thư viện h20
import h2o
from h2o.automl import H2OAutoML

train = X_train
test = X_test
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
#Tap train la tap co nhan
y_train = h2o.H2OFrame( y_train)
train['C65'] = y_train
#Tap test la tap co nhan
y_test = h2o.H2OFrame( y_test)
test['C65'] = y_test
#
h2o.init()
y = 'C65'
aml = H2OAutoML(max_runtime_secs=300, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)