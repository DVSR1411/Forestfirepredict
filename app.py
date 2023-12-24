from flask import Flask, render_template, request
import os
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('model.pkl')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form data
        temperature = int(request.form['temperature'])
        humidity = int(request.form['humidity'])
        oxygen = int(request.form['oxygen'])
        input_data = np.array([[temperature, humidity, oxygen]])
        prediction = model.predict(input_data)
        res = True if prediction[0] > 0.5 else False
        result = f"The Prediction is: {res}"
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
