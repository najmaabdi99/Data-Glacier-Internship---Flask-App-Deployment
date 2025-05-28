from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/iris_model.pkl")  

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        data = np.array([[sl, sw, pl, pw]])
        class_index = model.predict(data)[0]

        # Map numeric class to flower name
        flower_names = ['Setosa 🌸', 'Versicolor 🌺', 'Virginica 🌼']
        prediction = flower_names[class_index]

        return render_template("index.html", prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)