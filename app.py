from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)

    # Make a prediction using the model
    prediction = model.predict(final_input)
    output = prediction[0]

    # Map the prediction to the iris species
    species = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species[output]

    return render_template('index.html', prediction_text=f'Iris species: {predicted_species}')

if __name__ == "__main__":
    app.run(debug=True)
