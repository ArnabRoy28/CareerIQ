
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("saved_models/career_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame([request.json])
    result = model.predict(data)[0]
    return jsonify({'career': result})

if __name__ == '__main__':
    app.run(debug=True)
