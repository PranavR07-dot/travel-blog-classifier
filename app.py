# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/tourism_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('blog', '').strip()
    if not text:
        return render_template('index.html', prediction=None, input_text='')

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return render_template(
        'index.html',
        prediction=f"Predicted Category: {prediction}",
        input_text=text
    )

if __name__ == '__main__':
    app.run(debug=True)
