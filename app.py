import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_utils import DiabetesPredictor, validate_input, get_feature_descriptions
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive-prediction {
        background-color: #FFE5E5;
        border-left: 5px solid #FF6B6B;
    }
    .negative-prediction {
        background-color: #E5F7F5;
        border-left: 5px solid #4ECDC4;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = DiabetesPredictor()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

def load_and_train_model():
    """Load dataset and train the model"""
    try:
        with st.spinner("Loading dataset and training model..."):
            # Load dataset
            df = st.session_state.predictor.load_data('diabetes.csv')
            st.session_state.dataset = df
            
            # Train model
            training_results = st.session_state.predictor.train_model(df)
            st.session_state.model_trained = True
            
            return training_results
    except Exception as e:
        st.error(f"Error loading dataset or training model: {str(e)}")
        return None

def show_dataset_info():
    """Display dataset information"""
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Diabetes Cases", df['Outcome'].sum())
        with col4:
            st.metric("Non-Diabetes Cases", len(df) - df['Outcome'].sum())
        
        # Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))
        
        # Dataset statistics
        st.subheader("📈 Dataset Statistics")
        st.dataframe(df.describe())

def show_data_visualizations():
    """Display data visualizations"""
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        st.subheader("📊 Data Visualizations")
        
        # Outcome distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Diabetes Distribution**")
            fig_pie = px.pie(
                values=df['Outcome'].value_counts().values,
                names=['No Diabetes', 'Diabetes'],
                title="Distribution of Diabetes Cases",
                color_discrete_sequence=['#4ECDC4', '#FF6B6B']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.write("**Age Distribution by Outcome**")
            fig_hist = px.histogram(
                df, x='Age', color='Outcome',
                title="Age Distribution by Diabetes Status",
                color_discrete_sequence=['#4ECDC4', '#FF6B6B']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation heatmap
        st.write("**Feature Correlation Matrix**")
        fig_corr = px.imshow(
            df.corr(),
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions
        st.write("**Feature Distributions by Diabetes Status**")
        
        # Select features for visualization
        features_to_plot = ['Glucose', 'BMI', 'Age', 'Pregnancies']
        
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=features_to_plot,
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(features_to_plot):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # No diabetes
            fig_dist.add_trace(
                go.Histogram(
                    x=df[df['Outcome'] == 0][feature],
                    name=f'No Diabetes - {feature}',
                    marker_color='#4ECDC4',
                    opacity=0.7,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Diabetes
            fig_dist.add_trace(
                go.Histogram(
                    x=df[df['Outcome'] == 1][feature],
                    name=f'Diabetes - {feature}',
                    marker_color='#FF6B6B',
                    opacity=0.7,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
        
        fig_dist.update_layout(
            title="Feature Distributions by Diabetes Status",
            height=600,
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

def show_model_performance():
    """Display model performance metrics"""
    if st.session_state.model_trained:
        predictor = st.session_state.predictor
        
        st.subheader("🎯 Model Performance")
        
        # Model info
        model_info = predictor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_info['model_type'])
        with col2:
            st.metric("K Value", model_info['n_neighbors'])
        with col3:
            st.metric("Accuracy", f"{model_info['accuracy']:.2%}")
        
        # Confusion matrix
        if predictor.confusion_matrix is not None:
            st.write("**Confusion Matrix**")
            
            cm = predictor.confusion_matrix
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['No Diabetes', 'Diabetes'],
                y=['No Diabetes', 'Diabetes'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        if predictor.classification_report is not None:
            st.write("**Classification Report**")
            st.text(predictor.classification_report)

def prediction_interface():
    """Create the prediction interface"""
    st.subheader("🔮 Diabetes Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first by clicking 'Train Model' in the sidebar.")
        return
    
    # Get feature descriptions
    descriptions = get_feature_descriptions()
    
    # Create input form
    with st.form("prediction_form"):
        st.write("**Enter Patient Information:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20, value=0,
                help=descriptions['Pregnancies']
            )
            
            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0, max_value=300, value=100,
                help=descriptions['Glucose']
            )
            
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=200, value=70,
                help=descriptions['BloodPressure']
            )
            
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20,
                help=descriptions['SkinThickness']
            )
        
        with col2:
            insulin = st.number_input(
                "Insulin Level (mu U/ml)",
                min_value=0, max_value=1000, value=0,
                help=descriptions['Insulin']
            )
            
            bmi = st.number_input(
                "BMI",
                min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                help=descriptions['BMI']
            )
            
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                help=descriptions['DiabetesPedigreeFunction']
            )
            
            age = st.number_input(
                "Age",
                min_value=18, max_value=120, value=30,
                help=descriptions['Age']
            )
        
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            }
            
            # Validate input
            is_valid, error_message = validate_input(input_data)
            
            if not is_valid:
                st.error(f"Invalid input: {error_message}")
                return
            
            # Make prediction
            try:
                prediction_result = st.session_state.predictor.predict(input_data)
                
                # Display results
                st.subheader("📋 Prediction Results")
                
                # Main prediction
                if prediction_result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-box positive-prediction">
                        <h3>⚠️ High Risk of Diabetes</h3>
                        <p>The model predicts that this patient has a <strong>high risk</strong> of having diabetes.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box negative-prediction">
                        <h3>✅ Low Risk of Diabetes</h3>
                        <p>The model predicts that this patient has a <strong>low risk</strong> of having diabetes.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("📊 Probability Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Probability of No Diabetes",
                        f"{prediction_result['probability_no_diabetes']:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Probability of Diabetes",
                        f"{prediction_result['probability_diabetes']:.1f}%"
                    )
                
                # Probability visualization
                prob_data = pd.DataFrame({
                    'Outcome': ['No Diabetes', 'Diabetes'],
                    'Probability': [
                        prediction_result['probability_no_diabetes'],
                        prediction_result['probability_diabetes']
                    ]
                })
                
                fig_prob = px.bar(
                    prob_data, x='Outcome', y='Probability',
                    title="Prediction Probabilities",
                    color='Outcome',
                    color_discrete_sequence=['#4ECDC4', '#FF6B6B']
                )
                fig_prob.update_layout(showlegend=False)
                fig_prob.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Feature importance display
                st.subheader("📈 Input Feature Values")
                
                # Create radar chart for feature visualization
                features = list(input_data.keys())
                values = list(input_data.values())
                
                # Normalize values for radar chart (0-100 scale)
                normalized_values = []
                for feature, value in zip(features, values):
                    if feature == 'Pregnancies':
                        normalized_values.append(min(value / 20 * 100, 100))
                    elif feature == 'Glucose':
                        normalized_values.append(min(value / 300 * 100, 100))
                    elif feature == 'BloodPressure':
                        normalized_values.append(min(value / 200 * 100, 100))
                    elif feature == 'SkinThickness':
                        normalized_values.append(min(value / 100 * 100, 100))
                    elif feature == 'Insulin':
                        normalized_values.append(min(value / 1000 * 100, 100))
                    elif feature == 'BMI':
                        normalized_values.append(min(value / 70 * 100, 100))
                    elif feature == 'DiabetesPedigreeFunction':
                        normalized_values.append(min(value / 3 * 100, 100))
                    elif feature == 'Age':
                        normalized_values.append(min(value / 120 * 100, 100))
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values + [normalized_values[0]],  # Close the polygon
                    theta=features + [features[0]],  # Close the polygon
                    fill='toself',
                    name='Patient Profile',
                    line_color='#FF6B6B'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title="Patient Feature Profile (Normalized)"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Health Recommendations")
                
                recommendations = []
                
                if glucose > 140:
                    recommendations.append("🔸 High glucose level detected. Consider monitoring blood sugar regularly.")
                
                if bmi > 30:
                    recommendations.append("🔸 BMI indicates obesity. Consider weight management through diet and exercise.")
                elif bmi > 25:
                    recommendations.append("🔸 BMI indicates overweight. Consider lifestyle modifications.")
                
                if blood_pressure > 140:
                    recommendations.append("🔸 High blood pressure detected. Monitor blood pressure regularly.")
                
                if age > 45:
                    recommendations.append("🔸 Age is a risk factor. Regular health check-ups are recommended.")
                
                if diabetes_pedigree > 0.5:
                    recommendations.append("🔸 Family history of diabetes detected. Regular screening is important.")
                
                if not recommendations:
                    recommendations.append("🔸 Current parameters appear within normal ranges. Maintain healthy lifestyle.")
                
                for rec in recommendations:
                    st.write(rec)
                
                st.info("**Disclaimer:** This prediction is for educational purposes only. Please consult with a healthcare professional for proper medical diagnosis and treatment.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Prediction App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Aplikasi ini menggunakan algoritma pembelajaran mesin **K-Nearest Neighbors (K-NN)** untuk memprediksi risiko diabetes
 berdasarkan parameter medis. Model ini dilatih pada Pima Indian Diabetes Dataset.
    """)
    
    # Sidebar
    st.sidebar.title("🛠️ Model Control")
    
    # Model training section
    if st.sidebar.button("🚀 Train Model"):
        training_results = load_and_train_model()
        if training_results:
            st.sidebar.success("✅ Model trained successfully!")
            st.sidebar.write(f"**Accuracy:** {training_results['accuracy']:.2%}")
            st.sidebar.write(f"**Training samples:** {training_results['train_size']}")
            st.sidebar.write(f"**Testing samples:** {training_results['test_size']}")
    
    # Model status
    if st.session_state.model_trained:
        st.sidebar.success("🎯 Model is ready for predictions!")
    else:
        st.sidebar.warning("⚠️ Model not trained yet")
    
    # Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔮 Prediction", "📊 Dataset Info", "📈 Visualizations", "🎯 Model Performance"]
    )
    
    # Page routing
    if page == "🔮 Prediction":
        prediction_interface()
    elif page == "📊 Dataset Info":
        show_dataset_info()
    elif page == "📈 Visualizations":
        show_data_visualizations()
    elif page == "🎯 Model Performance":
        show_model_performance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>K-NN Diabetes Prediction Model</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
