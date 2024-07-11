from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("dataset/Iris.csv")

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Preprocess the data
X = df.drop('Species', axis=1)
y = df['Species']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
grid = {'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': np.arange(2, 8),
        'min_samples_leaf': np.arange(10, 20)}

model = DecisionTreeClassifier()
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)
grid_search = GridSearchCV(model, grid, cv=rskf)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
dt_model2 = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                   min_samples_leaf=best_params['min_samples_leaf'],
                                   min_samples_split=best_params['min_samples_split'],
                                   random_state=20)
dt_model2.fit(x_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    x_rf = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    x_rf_prediction = dt_model2.predict(x_rf)
    category_rf = label_encoder.inverse_transform(x_rf_prediction)

    image_url = ''
    if category_rf[0] == 'Iris-setosa':
        image_url = url_for('static', filename='setosa.jpg')
    elif category_rf[0] == 'Iris-versicolor':
        image_url = url_for('static', filename='versicolor.jpg')
    elif category_rf[0] == 'Iris-virginica':
        image_url = url_for('static', filename='virginica.jpg')

    return render_template('index.html', prediction=category_rf[0], image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
