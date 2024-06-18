from flask import Flask, render_template, request
import pickle
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
model_path = '/home/osama/Documents/frist_task/best_xgb.pkl'
try:
    model = pickle.load(open(model_path, 'rb'))
except AttributeError as e:
    logging.error(f"AttributeError: {e}")
    # Handle the error appropriately here
    # For example, you could set model to None or re-raise the error

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        # Extract data from form
        lead_time = int(request.form.get('lead_time', 0))
        avg_price = float(request.form.get('avg_price', 0.0))
        #special_requests = int(request.form.get('special_requests', 0))
        day_of_week = int(request.form.get('day_of_week', 1))
        month = int(request.form.get('month', 1))
        total_nights = int(request.form.get('total_nights', 1))

        # Predict cancellation
        prediction = model.predict([[lead_time, avg_price, day_of_week, month,total_nights]])
        result = "Cancelled" if prediction[0] == 1 else "Not Cancelled"
    except AttributeError as e:
        logging.error(f"AttributeError during prediction: {e}")
        result = "Error occurred during prediction"

    # Render the result on the home page
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
