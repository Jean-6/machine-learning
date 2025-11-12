from flask import Flask
from sklearn import datasets
import pandas as pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/health')
def index():
    return 'Hello word'

if __name__ == '__main__':
    app = Flask(__name__)
    iris = datasets.load_iris()
    data = pandas.DataFrame(data=iris.data, columns= iris.feature_names)

    # features/data , target/label
    X, y = iris['data'], iris['target']


    sample_X = X[:60]
    sample_y = y[:60]

    print(sample_X)
    print(sample_y)


    X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2, random_state=42)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)


    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    app.run(debug=True)

