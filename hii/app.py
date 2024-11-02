from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
from flask import send_file
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import os

app = Flask(__name__)

class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LabelEncoderScratch:
    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        for i, col in enumerate(X.T):
            self.classes_[i] = {label: idx for idx, label in enumerate(np.unique(col))}
        return self

    def transform(self, X):
        X_encoded = np.zeros(X.shape, dtype=int)
        for i, col in enumerate(X.T):
            X_encoded[:, i] = [self.classes_[i][label] for label in col]
        return X_encoded

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]




scaler_path = 'scaler.pkl'
encoder_path = 'label_encoder_scratch.pkl'
model_path = 'knn_best_model.pkl'

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(encoder_path, 'rb') as file:
    label_encoder_scratch = pickle.load(file)

with open(model_path, 'rb') as file:
    model = pickle.load(file)


numerical_columns = [
    'Loan_Amount_Requested', 'Annual_Income', 'Debt_To_Income',
    'Inquiries_Last_6Mo', 'Number_Open_Accounts', 'Total_Accounts',
    'loan_income_ratio', 'total_income_ratio', 'debt_income_ratio'
]
categorical_columns = ['Length_Employed', 'Income_Verified', 'Purpose_Of_Loan']


df = pd.read_csv('processed_test_data.csv')


dropdown_options = {}
for column in categorical_columns:
    if column in df.columns:
        unique_values = df[column].dropna().unique().tolist()
        dropdown_options[column] = unique_values[:1000]  


numerical_ranges = {}
for column in numerical_columns:
    if column in df.columns:
        numerical_ranges[column] = {
            "min": df[column].min(),
            "max": df[column].max()
        }

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        
        test_df = pd.DataFrame([data])

        
        for column in numerical_columns + categorical_columns:
            if column not in test_df.columns:
                return jsonify({'error': f'Missing column: {column}'}), 400

        
        for col in numerical_columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

        
        test_df[categorical_columns] = test_df[categorical_columns].astype(str)

        
        test_df = test_df.dropna(subset=numerical_columns + categorical_columns)

        
        X_test_encoded = label_encoder_scratch.transform(test_df[categorical_columns].values)

        
        X_test_scaled = scaler.transform(test_df[numerical_columns].values)

        
        X_test_processed = np.hstack([X_test_scaled, X_test_encoded])

        
        predictions = model.predict(X_test_processed)
        print("Raw Predictions:", predictions)

        
        predictions = predictions + 1
        print("Predictions:", predictions)

        
        predicted_interest_rate = float(predictions[0])

        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))  

        
        ax[0].hist(test_df['Loan_Amount_Requested'], bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax[0].set_title('Distribution of Loan Amount Requested')
        ax[0].set_xlabel('Loan Amount Requested')
        ax[0].set_ylabel('Frequency')
        ax[0].grid(True, linestyle='--', alpha=0.6)

        
        x_values = np.linspace(test_df['Annual_Income'].min(), test_df['Annual_Income'].max(), 500)
        y_values = np.full_like(x_values, predicted_interest_rate)
        ax[1].plot(x_values, y_values, color='orange', linestyle='-', linewidth=2, label='Predicted Interest Rate')
        ax[1].scatter(test_df['Annual_Income'], predictions, color='blue', alpha=0.5, edgecolor='black', label='Predicted Data Points')
        ax[1].set_title('Predicted Interest Rate vs Annual Income')
        ax[1].set_xlabel('Annual Income')
        ax[1].set_ylabel('Predicted Interest Rate')
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        
        plt.tight_layout()

        
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")

        
        return jsonify({
            'predicted_interest_rate': predicted_interest_rate,
            'plot_url': request.host_url + 'static/plot.png'
        })
    
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    predicted_interest_rate = request.args.get('predicted_interest_rate')
    plot_url = request.args.get('plot_url')

    return render_template('results.html', 
                           predicted_interest_rate=predicted_interest_rate, 
                           plot_url=plot_url)



@app.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    return jsonify({
        'categorical_options': dropdown_options,
        'numerical_ranges': numerical_ranges
    })

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True,port=5001)
