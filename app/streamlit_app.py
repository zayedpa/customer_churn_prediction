"""
Streamlit app for customer churn prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('../src')

from preprocessing import feature_engineering


# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('../models/best_churn_pipeline.pkl')
    except:
        return None


def main():
    st.set_page_config(
        page_title="Churn Predictor", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Header
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Predict whether a customer will churn based on their profile")
    st.markdown("---")
    
    # Load model
    pipeline = load_model()
    
    if pipeline is None:
        st.error("❌ **Failed to load model**")
        st.info("Please run the following commands first:")
        st.code("cd src\npython train_models.py")
        return
    else:
        st.success("✅ Model loaded successfully!")
    
    # Sidebar for input
    st.sidebar.header("📋 Customer Information")
    st.sidebar.markdown("Enter customer details below:")
    
    # Input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 40)
        balance = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
        products_number = st.slider("Number of Products", 1, 4, 2)
        credit_card = st.selectbox("Has Credit Card?", [1, 0], 
                                   format_func=lambda x: "Yes" if x else "No")
    
    with col2:
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        estimated_salary = st.number_input("Estimated Salary ($)", 
                                          0.0, 200000.0, 60000.0)
        active_member = st.selectbox("Active Member?", [1, 0], 
                                     format_func=lambda x: "Yes" if x else "No")
    
    # Create sample dataframe
    sample = pd.DataFrame([{
        'customer_id': 999999,
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Customer Profile Summary")
        
        # Display profile
        profile_df = sample.drop(columns=['customer_id']).T
        profile_df.columns = ['Value']
        st.dataframe(profile_df, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Churn Prediction")
        
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            # Apply feature engineering
            sample_fe = feature_engineering(sample)
            sample_fe = sample_fe.drop(columns=['customer_id'])
            
            # Make prediction
            pred = pipeline.predict(sample_fe)[0]
            prob = pipeline.predict_proba(sample_fe)[0, 1]
            
            # Display result
            st.markdown("---")
            
            if pred == 1:
                st.error("### ⚠️ **HIGH RISK OF CHURN**")
                st.metric("Churn Probability", f"{prob:.1%}", delta=None)
                
                # Risk level
                if prob > 0.7:
                    risk_level = "🔴 Critical"
                elif prob > 0.5:
                    risk_level = "🟠 High"
                else:
                    risk_level = "🟡 Moderate"
                
                st.markdown(f"**Risk Level:** {risk_level}")
                st.markdown("**Recommendation:** Immediate retention strategies needed")
                
                with st.expander("💡 Suggested Actions"):
                    st.markdown("""
                    - Contact customer immediately
                    - Offer personalized retention incentives
                    - Schedule account review meeting
                    - Provide premium customer service
                    """)
            else:
                st.success("### ✅ **LOW RISK OF CHURN**")
                st.metric("Churn Probability", f"{prob:.1%}", delta=None)
                st.markdown("**Risk Level:** 🟢 Low")
                st.markdown("**Recommendation:** Continue standard engagement")
                
                with st.expander("💡 Suggested Actions"):
                    st.markdown("""
                    - Maintain regular communication
                    - Monitor account activity
                    - Cross-sell opportunities
                    - Gather feedback for improvement
                    """)
            
            # Progress bar
            st.markdown("---")
            st.markdown("**Churn Risk Meter**")
            st.progress(prob)
            
            # Probability breakdown
            st.markdown(f"**No Churn:** {(1-prob)*100:.1f}% | **Churn:** {prob*100:.1f}%")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Model:** Gradient Boosting")
    with col2:
        st.markdown("**Accuracy:** ~86%")
    with col3:
        st.markdown("**Framework:** scikit-learn")
    
    # Additional info
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        This application uses machine learning to predict customer churn probability.
        
        **Features:**
        - Real-time prediction based on customer profile
        - Risk assessment and recommendations
        - Built with Gradient Boosting Classifier
        
        **How to use:**
        1. Enter customer information in the sidebar
        2. Click "Predict Churn" button
        3. Review the prediction and recommendations
        """)


if __name__ == "__main__":
    main()



