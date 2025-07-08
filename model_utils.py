import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

class DiabetesPredictor:
    def __init__(self, n_neighbors=5):
        """
        Initialize the Diabetes Predictor with K-NN classifier
        
        Args:
            n_neighbors (int): Number of neighbors for K-NN algorithm
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None
        
    def load_data(self, file_path='diabetes.csv'):
        """
        Load diabetes dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def preprocess_data(self, df):
        """
        Preprocess the diabetes dataset
        
        Args:
            df (pandas.DataFrame): Raw dataset
            
        Returns:
            tuple: (X, y) - Features and target variables
        """
        # Handle missing values represented as 0 in certain columns
        # These columns should not have 0 values in reality
        zero_not_acceptable = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
        
        for column in zero_not_acceptable:
            df[column] = df[column].replace(0, np.nan)
            # Fill with median values
            df[column] = df[column].fillna(df[column].median())
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['Outcome']
        
        return X, y
    
    def train_model(self, df):
        """
        Train the K-NN model on the diabetes dataset
        
        Args:
            df (pandas.DataFrame): Diabetes dataset
            
        Returns:
            dict: Training results including accuracy and metrics
        """
        try:
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            self.accuracy = accuracy_score(y_test, y_pred)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            self.classification_report = classification_report(y_test, y_pred)
            
            self.is_trained = True
            
            return {
                'accuracy': self.accuracy,
                'confusion_matrix': self.confusion_matrix,
                'classification_report': self.classification_report,
                'test_size': len(X_test),
                'train_size': len(X_train)
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, input_data):
        """
        Make diabetes prediction for given input data
        
        Args:
            input_data (dict or list): Input features for prediction
            
        Returns:
            dict: Prediction results with probabilities
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model first.")
        
        try:
            # Convert input to numpy array if it's a dictionary
            if isinstance(input_data, dict):
                input_array = np.array([[
                    input_data['Pregnancies'],
                    input_data['Glucose'],
                    input_data['BloodPressure'],
                    input_data['SkinThickness'],
                    input_data['Insulin'],
                    input_data['BMI'],
                    input_data['DiabetesPedigreeFunction'],
                    input_data['Age']
                ]])
            else:
                input_array = np.array([input_data])
            
            # Scale input data
            input_scaled = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probabilities = self.model.predict_proba(input_scaled)[0]
            
            return {
                'prediction': int(prediction),
                'probability_no_diabetes': float(probabilities[0]) * 100,
                'probability_diabetes': float(probabilities[1]) * 100,
                'prediction_label': 'Diabetes' if prediction == 1 else 'No Diabetes'
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def get_model_info(self):
        """
        Get information about the trained model
        
        Returns:
            dict: Model information
        """
        if not self.is_trained:
            return {"message": "Model is not trained yet."}
        
        return {
            'model_type': 'K-Nearest Neighbors',
            'n_neighbors': self.model.n_neighbors,
            'accuracy': self.accuracy,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
    
    def save_model(self, file_path='diabetes_model.pkl'):
        """
        Save the trained model to a file
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Cannot save untrained model.")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'accuracy': self.accuracy
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self, file_path='diabetes_model.pkl'):
        """
        Load a pre-trained model from a file
        
        Args:
            file_path (str): Path to the saved model
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file '{file_path}' not found.")
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            self.is_trained = True
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

def validate_input(input_data):
    """
    Validate user input for diabetes prediction
    
    Args:
        input_data (dict): Input data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in input_data:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    validations = {
        'Pregnancies': (0, 20),
        'Glucose': (0, 300),
        'BloodPressure': (0, 200),
        'SkinThickness': (0, 100),
        'Insulin': (0, 1000),
        'BMI': (10, 70),
        'DiabetesPedigreeFunction': (0, 3),
        'Age': (18, 120)
    }
    
    for field, (min_val, max_val) in validations.items():
        try:
            value = float(input_data[field])
            if value < min_val or value > max_val:
                return False, f"{field} must be between {min_val} and {max_val}"
        except (ValueError, TypeError):
            return False, f"{field} must be a valid number"
    
    return True, ""

def get_feature_descriptions():
    """
    Get descriptions for each feature used in the model
    
    Returns:
        dict: Feature descriptions
    """
    return {
        'Pregnancies': 'Number of times pregnant (0-20)',
        'Glucose': 'Plasma glucose concentration (mg/dL) (0-300)',
        'BloodPressure': 'Diastolic blood pressure (mm Hg) (0-200)',
        'SkinThickness': 'Triceps skin fold thickness (mm) (0-100)',
        'Insulin': '2-Hour serum insulin (mu U/ml) (0-1000)',
        'BMI': 'Body mass index (weight in kg/(height in m)²) (10-70)',
        'DiabetesPedigreeFunction': 'Diabetes pedigree function (0-3)',
        'Age': 'Age in years (18-120)'
    }
