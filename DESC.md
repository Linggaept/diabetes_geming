# Diabetes Prediction App

## Overview

This is a machine learning-powered web application built with Streamlit that predicts diabetes risk using a K-Nearest Neighbors (KNN) classifier. The application provides an interactive interface for users to input health metrics and receive diabetes risk predictions with visualizations and detailed analysis.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for rapid prototyping and easy deployment of ML applications
- **Styling**: Custom CSS embedded within the Streamlit app for enhanced user experience
- **Layout**: Wide layout with expandable sidebar for input controls
- **Visualization**: Multiple libraries integrated (Matplotlib, Seaborn, Plotly) for comprehensive data visualization

### Backend Architecture
- **ML Framework**: Scikit-learn with K-Nearest Neighbors (KNN) classifier
- **Data Processing**: Pandas and NumPy for data manipulation and numerical operations
- **Model Management**: Custom DiabetesPredictor class encapsulating the entire ML pipeline
- **Preprocessing**: StandardScaler for feature normalization

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Primary Streamlit application interface
- **Features**: 
  - Interactive input forms for health metrics
  - Real-time predictions with confidence scores
  - Data visualizations and analytics
  - Responsive design with custom CSS styling

### 2. Model Utilities (model_utils.py)
- **Purpose**: Machine learning pipeline and utilities
- **Components**:
  - `DiabetesPredictor` class: Complete ML workflow management
  - Data loading and preprocessing functions
  - Model training and evaluation methods
  - Input validation utilities

### 3. Core ML Pipeline
- **Algorithm**: K-Nearest Neighbors (KNN) classifier
- **Rationale**: Simple, interpretable algorithm suitable for small-to-medium datasets
- **Features**: 8 health metrics (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Preprocessing**: StandardScaler for feature normalization

## Data Flow

1. **Input Collection**: User provides health metrics through Streamlit interface
2. **Validation**: Input data validated using `validate_input` function
3. **Preprocessing**: Features scaled using pre-trained StandardScaler
4. **Prediction**: KNN model generates diabetes risk prediction
5. **Visualization**: Results displayed with interactive charts and metrics
6. **Analysis**: Additional insights and feature importance provided

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities

### Visualization Libraries
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

### Model Persistence
- **pickle**: Model serialization and deserialization

## Deployment Strategy

### Local Development
- Standard Python environment with pip-installable dependencies
- Streamlit's built-in development server for local testing
- No external database required - uses CSV data files

### Production Considerations
- Streamlit Cloud or similar platform deployment
- Model files persisted using pickle for state management
- Stateless application design for scalability

## Changelog

- July 08, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.

## Technical Notes

### Model Architecture Decisions
- **KNN Selection**: Chosen for interpretability and effectiveness with small datasets
- **Feature Scaling**: StandardScaler applied to handle different feature scales
- **No Database**: Simple CSV-based data storage for rapid prototyping

### Visualization Strategy
- Multiple visualization libraries integrated to provide comprehensive analysis
- Interactive Plotly charts for enhanced user engagement
- Static matplotlib/seaborn charts for detailed statistical analysis

### Code Organization
- Modular design with separate utility classes
- Custom CSS for professional appearance
- Error handling and input validation throughout