import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend for plotting
from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load trained models and scaler
try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('xgb_classifier.pkl', 'rb') as xgb_classifier_file:
        xgb_classifier = pickle.load(xgb_classifier_file)

    with open('xgb_regressor.pkl', 'rb') as xgb_regressor_file:
        xgb_regressor = pickle.load(xgb_regressor_file)

    with open('classifier.pkl', 'rb') as rf_classifier_file:  # Load Random Forest classifier
        classifier = pickle.load(rf_classifier_file)

except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit(1)

# File paths for temporary plot images
RF_CM_PATH = "static/confusion_matrix_rf.png"
RF_BAR_PATH = "static/bar_graph_rf.png"
XGB_CM_PATH = "static/confusion_matrix_xgb.png"
XGB_BAR_PATH = "static/bar_graph_xgb.png"

# Home route
@app.route('/')
def home():
    return render_template('homepage.html')

# Route for rendering graphs
@app.route('/graphs')
def graphs():
    try:
        # Load dataset
        data = pd.read_csv('dataset.csv')
        X = data[['Latitude', 'Longitude', 'Depth']]
        y = pd.Categorical(data['Magnitude'].apply(
            lambda x: "Low" if x < 4 else "Medium" if x < 6 else "High"
        )).codes

        # Scale features
        X_scaled = scaler.transform(X)

        ## --- RANDOM FOREST ANALYSIS ---
        # Random Forest Predictions
        y_rf_pred = classifier.predict(X)  # Using Random Forest without scaling
        rf_accuracy = accuracy_score(y, y_rf_pred) * 100

        # Random Forest Confusion Matrix
        cm_rf = confusion_matrix(y, y_rf_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"],
                    yticklabels=["Low", "Medium", "High"])
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(RF_CM_PATH)
        plt.close()

        # Random Forest Feature Importance Bar Graph
        feature_importances_rf = pd.Series(classifier.feature_importances_, index=X.columns)
        plt.figure(figsize=(5, 3))
        feature_importances_rf.sort_values().plot(kind='barh', color='skyblue')
        plt.title('Random Forest Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(RF_BAR_PATH)
        plt.close()

        ## --- XGBOOST ANALYSIS ---
        # XGBoost Predictions
        y_xgb_pred = xgb_classifier.predict(X_scaled)
        xgb_accuracy = accuracy_score(y, y_xgb_pred) * 100

        # XGBoost Confusion Matrix
        cm_xgb = confusion_matrix(y, y_xgb_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', xticklabels=["Low", "Medium", "High"],
                    yticklabels=["Low", "Medium", "High"])
        plt.title('XGBoost Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(XGB_CM_PATH)
        plt.close()

        # XGBoost Feature Importance Bar Graph
        feature_importances_xgb = pd.Series(xgb_classifier.feature_importances_, index=X.columns)
        plt.figure(figsize=(5, 3))
        feature_importances_xgb.sort_values().plot(kind='barh', color='coral')
        plt.title('XGBoost Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(XGB_BAR_PATH)
        plt.close()

        # Render the graphs page with metrics and file paths for the graphs
        return render_template(
            'graphs.html',
            rf_accuracy=rf_accuracy,
            xgb_accuracy=xgb_accuracy,
            rf_cm_url=url_for('static', filename='confusion_matrix_rf.png'),
            rf_bar_url=url_for('static', filename='bar_graph_rf.png'),
            xgb_cm_url=url_for('static', filename='confusion_matrix_xgb.png'),
            xgb_bar_url=url_for('static', filename='bar_graph_xgb.png')
        )
    
    except Exception as e:
        return f"Error generating graphs: {str(e)}"

# Route for prediction form and results
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Parse input data from the form
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            depth = float(request.form['depth'])

            # Prepare input for prediction
            input_data = pd.DataFrame([[latitude, longitude, depth]], columns=['Latitude', 'Longitude', 'Depth'])
            scaled_data = scaler.transform(input_data)

            # Perform predictions
            predicted_magnitude = xgb_regressor.predict(scaled_data)[0]  # Predict magnitude
            predicted_intensity_code = xgb_classifier.predict(scaled_data)[0]
            predicted_intensity = ['Low', 'Medium', 'High'][int(predicted_intensity_code)]

            # Render results
            return render_template(
                'prediction_result.html',
                magnitude=round(predicted_magnitude, 2),
                intensity=predicted_intensity
            )
        except Exception as e:
            return f"Error: {str(e)}"

    # Render the prediction form if request method is GET
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
