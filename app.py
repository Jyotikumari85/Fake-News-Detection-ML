# app.py
from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Load your trained model & vectorizer
model      = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    # show the blank form
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # only process when form is submitted
    if request.method == 'POST':
        news_text = request.form['news']
        vect      = vectorizer.transform([news_text])
        pred      = model.predict(vect)[0]
        result    = "Real News" if pred == 1 else "Fake News"
        return render_template('index.html', prediction=result)
    # if someone browses to /predict via GET, redirect them back
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
