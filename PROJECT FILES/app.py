from flask import Flask, render_template, request
from utils.predictor_module import predict_traffic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        date = request.form['date']
        time = request.form['time']

        prediction = predict_traffic(holiday, temp, rain, snow, weather, date, time)
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
