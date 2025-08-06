import streamlit as st
import json
import numpy as np
import faiss
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from ml_models import FraudDetectionML

# Load environment variables
load_dotenv(dotenv_path="/Users/dhruv/Documents/Internship/UC2/.env")

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .normal-transaction {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def create_simple_embedding(text):
    # Create a hash-based vector representation
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert hex to numerical values
    vector = []
    for i in range(0, len(hash_hex), 2):
        if i + 1 < len(hash_hex):
            vector.append(int(hash_hex[i:i+2], 16) / 255.0)
    
    # Pad or truncate to 1536 dimensions (standard embedding size)
    while len(vector) < 1536:
        vector.extend(vector[:min(1536 - len(vector), len(vector))])
    
    return vector[:1536]

def transaction_to_text(txn):
    """Convert transaction to text representation"""
    text_parts = []
    
    if 'amount' in txn:
        text_parts.append(f"amount_{txn['amount']}")
    if 'location' in txn:
        text_parts.append(f"location_{txn['location']}")
    if 'timestamp' in txn:
        text_parts.append(f"timestamp_{txn['timestamp']}")
    elif 'time' in txn:
        text_parts.append(f"time_{txn['time']}")
    if 'user_id' in txn:
        text_parts.append(f"user_{txn['user_id']}")
    if 'merchant' in txn:
        text_parts.append(f"merchant_{txn['merchant']}")
    if 'category' in txn:
        text_parts.append(f"category_{txn['category']}")
    if 'frequency' in txn:
        text_parts.append(f"frequency_{txn['frequency']}")
    
    return " ".join(text_parts)

@st.cache_data
def load_fraud_vectors():
    """Load known fraud vectors and build FAISS index"""
    try:
        with open("/Users/dhruv/Documents/Internship/UC2/fraud_detection/fraud_vectors.json") as f:
            known_fraud_txns = json.load(f)
        
        fraud_vectors = []
        fraud_meta = []
        
        for txn in known_fraud_txns:
            text = transaction_to_text(txn)
            # Use simple embedding instead of OpenAI
            vector = create_simple_embedding(text)
            fraud_vectors.append(vector)
            fraud_meta.append(txn)
        
        if fraud_vectors:
            dim = len(fraud_vectors[0])
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(fraud_vectors).astype("float32"))
            return index, fraud_meta, known_fraud_txns
        else:
            return None, [], []
    except FileNotFoundError:
        st.warning("fraud_vectors.json not found. Starting with empty fraud database.")
        return None, [], []
    except Exception as e:
        st.error(f"Error loading fraud vectors: {e}")
        return None, [], []

def analyze_transaction_features(txn):
    """Analyze transaction features to provide insights about potential fraud"""
    insights = []
    
    # Extract common fraud indicators
    if 'amount' in txn:
        try:
            amount = float(txn['amount'])
            if amount > 10000:
                insights.append(f"High transaction amount: ${amount:,.2f}")
            elif amount < 1:
                insights.append(f"Suspiciously low amount: ${amount:.2f}")
        except (ValueError, TypeError):
            insights.append("Invalid amount format")
    
    if 'location' in txn:
        location = str(txn['location'])
        suspicious_locations = ['unknown', 'offshore', 'tax_haven']
        if any(susp in location.lower() for susp in suspicious_locations):
            insights.append(f"Suspicious location: {location}")
    
    # Handle both timestamp and time fields
    time_field = None
    if 'timestamp' in txn:
        time_field = txn['timestamp']
    elif 'time' in txn:
        time_field = txn['time']
    
    if time_field:
        time_str = str(time_field)
        if '23:' in time_str or '00:' in time_str or '01:' in time_str:
            insights.append("Unusual transaction time (late night)")
    
    # Analyze merchant information
    if 'merchant' in txn:
        merchant = str(txn['merchant']).lower()
        suspicious_merchants = ['unknown', 'offshore', 'unregistered']
        if any(susp in merchant for susp in suspicious_merchants):
            insights.append(f"Suspicious merchant: {txn['merchant']}")
    
    # Analyze category information
    if 'category' in txn:
        category = str(txn['category']).lower()
        high_risk_categories = ['gambling', 'crypto', 'adult', 'illegal']
        if any(risk in category for risk in high_risk_categories):
            insights.append(f"High-risk category: {txn['category']}")
    
    if 'frequency' in txn:
        try:
            freq = int(txn.get('frequency', 0))
            if freq > 10:
                insights.append(f"High transaction frequency: {freq} transactions")
        except (ValueError, TypeError):
            pass
    
    # Check for missing critical information
    missing_fields = []
    required_fields = ['amount', 'location', 'user_id']
    if 'timestamp' not in txn and 'time' not in txn:
        missing_fields.append('timestamp/time')
    
    for field in required_fields:
        if field not in txn or not txn[field]:
            missing_fields.append(field)
    
    if missing_fields:
        insights.append(f"Missing critical information: {', '.join(missing_fields)}")
    
    return insights

def get_fraud_explanation(txn, similarity_score, is_fraud):
    """Generate fraud explanation using statistical analysis"""
    insights = analyze_transaction_features(txn)
    
    if is_fraud:
        explanation = f"This transaction shows {similarity_score:.1%} similarity to known fraud patterns.\n"
        
        if insights:
            explanation += "\nSuspicious indicators detected:\n"
            for insight in insights:
                explanation += f"• {insight}\n"
        else:
            explanation += "\nNo obvious suspicious indicators, but pattern matches known fraud cases."
    else:
        explanation = "This transaction appears to be normal based on current fraud patterns."
        if insights:
            explanation += "\n\nNote: Some minor concerns detected:\n"
            for insight in insights:
                explanation += f"• {insight}\n"
    
    return explanation

def is_similar(new_vec, index, threshold=0.85):
    """Check if a vector is similar to known fraud patterns"""
    if index is None:
        return 0.0, False
    
    D, I = index.search(np.array([new_vec]).astype("float32"), 1)
    similarity = 1 - D[0][0]
    return similarity, similarity > threshold

def main():
    # Header
    st.markdown('<h1 class="main-header">🕵️ Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Fraud Detection Threshold", 0.0, 1.0, 0.85, 0.05)
    
    # Load fraud vectors
    with st.spinner("Loading fraud detection system..."):
        index, fraud_meta, known_frauds = load_fraud_vectors()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Upload & Analyze", "🤖 ML Models", "📈 Analytics"])
    
    with tab1:
        st.header("Transaction Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload transactions JSON file",
            type=['json'],
            help="Upload a JSON file containing transactions to analyze"
        )
        
        if uploaded_file is not None:
            try:
                transactions = json.load(uploaded_file)
                st.success(f"Successfully loaded {len(transactions)} transactions")
                
                # Analysis button
                if st.button("🔍 Analyze Transactions", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        results = []
                        fraud_count = 0
                        normal_count = 0
                        
                        progress_bar = st.progress(0)
                        
                        for i, txn in enumerate(transactions):
                            # Update progress
                            progress_bar.progress((i + 1) / len(transactions))
                            
                            txn_text = transaction_to_text(txn)
                            # Use simple embedding
                            txn_vec = create_simple_embedding(txn_text)
                            similarity, is_fraud = is_similar(txn_vec, index, threshold)
                            
                            result = {
                                'transaction': txn,
                                'similarity': similarity,
                                'is_fraud': is_fraud,
                                'explanation': get_fraud_explanation(txn, similarity, is_fraud)
                            }
                            results.append(result)
                            
                            if is_fraud:
                                fraud_count += 1
                            else:
                                normal_count += 1
                        
                        progress_bar.empty()
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", len(transactions))
                        with col2:
                            st.metric("Fraud Detected", fraud_count)
                        with col3:
                            st.metric("Normal Transactions", normal_count)
                        with col4:
                            fraud_rate = (fraud_count / len(transactions)) * 100 if transactions else 0
                            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                        
                        # Results table
                        st.subheader("Detailed Results")
                        
                        for i, result in enumerate(results):
                            if result['is_fraud']:
                                with st.expander(f"⚠️ Fraud Alert - Transaction {i+1} (Similarity: {result['similarity']:.3f})"):
                                    st.json(result['transaction'])
                                    st.markdown(f"**Analysis:** {result['explanation']}")
                                    
                                    # Add to fraud vectors option
                                    if st.button(f"Add to Fraud Database", key=f"add_fraud_{i}"):
                                        try:
                                            with open("fraud_detection/fraud_vectors.json", "r") as f:
                                                current_frauds = json.load(f)
                                        except FileNotFoundError:
                                            current_frauds = []
                                        
                                        current_frauds.append(result['transaction'])
                                        with open("fraud_detection/fraud_vectors.json", "w") as f:
                                            json.dump(current_frauds, f, indent=2)
                                        st.success("Transaction added to fraud database!")
                                        st.rerun()
                            else:
                                with st.expander(f"✅ Normal - Transaction {i+1} (Similarity: {result['similarity']:.3f})"):
                                    st.json(result['transaction'])
                                    st.info(result['explanation'])
                        
                        # Store results for ML tab
                        st.session_state['analysis_results'] = results
                        st.session_state['transactions'] = transactions
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab2:
        st.header("🤖 Machine Learning Models")
        
        # Initialize ML system
        if 'ml_system' not in st.session_state:
            st.session_state.ml_system = FraudDetectionML()
        
        ml_system = st.session_state.ml_system
        
        # Model training section
        st.subheader("Train ML Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Load training data
            if st.button("📊 Load Training Data"):
                try:
                    with open("sample_transactions.json", "r") as f:
                        training_transactions = json.load(f)
                    
                    st.session_state.training_transactions = training_transactions
                    st.success(f"Loaded {len(training_transactions)} transactions for training")
                except Exception as e:
                    st.error(f"Error loading training data: {e}")
        
        with col2:
            # Train models
            if st.button("🎯 Train Models") and 'training_transactions' in st.session_state:
                with st.spinner("Training ML models..."):
                    try:
                        performance = ml_system.train_models(st.session_state.training_transactions)
                        st.session_state.model_performance = performance
                        st.success("Models trained successfully!")
                        
                        # Save models
                        ml_system.save_models()
                        st.info("Models saved to disk")
                        
                    except Exception as e:
                        st.error(f"Error training models: {e}")
        
        # Model performance display
        if 'model_performance' in st.session_state:
            st.subheader("Model Performance")
            
            performance_df = pd.DataFrame([
                {
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'AUC': metrics['auc']
                }
                for name, metrics in st.session_state.model_performance.items()
            ])
            
            st.dataframe(performance_df, use_container_width=True)
            
            # Performance visualization
            fig = go.Figure()
            
            models = list(st.session_state.model_performance.keys())
            accuracies = [st.session_state.model_performance[m]['accuracy'] for m in models]
            aucs = [st.session_state.model_performance[m]['auc'] for m in models]
            
            fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracies, marker_color='lightblue'))
            fig.add_trace(go.Bar(name='AUC', x=models, y=aucs, marker_color='lightcoral'))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if hasattr(ml_system, 'models') and ml_system.models:
            st.subheader("Feature Importance")
            
            model_choice = st.selectbox(
                "Select model for feature importance:",
                list(ml_system.models.keys())
            )
            
            if st.button("Show Feature Importance"):
                try:
                    importance_df = ml_system.get_feature_importance(model_choice)
                    if importance_df is not None:
                        # Show top 15 features
                        top_features = importance_df.head(15)
                        
                        fig = px.bar(
                            top_features, 
                            x='importance', 
                            y='feature',
                            orientation='h',
                            title=f"Top 15 Features - {model_choice.replace('_', ' ').title()}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(importance_df, use_container_width=True)
                    else:
                        st.warning("Feature importance not available for this model")
                except Exception as e:
                    st.error(f"Error getting feature importance: {e}")
        
        # ML-based prediction
        st.subheader("ML-Based Fraud Detection")
        
        if 'transactions' in st.session_state and ml_system.models:
            model_choice = st.selectbox(
                "Select ML model for prediction:",
                list(ml_system.models.keys()),
                key="ml_model_choice"
            )
            
            threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.05, key="ml_threshold")
            
            if st.button("🔍 Run ML Analysis"):
                with st.spinner("Running ML analysis..."):
                    try:
                        predictions, probabilities = ml_system.predict_fraud(
                            st.session_state.transactions, 
                            model_choice
                        )
                        
                        # Count frauds
                        fraud_count_ml = sum(predictions)
                        normal_count_ml = len(predictions) - fraud_count_ml
                        
                        # Display ML results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", len(predictions))
                        with col2:
                            st.metric("ML Fraud Detected", fraud_count_ml)
                        with col3:
                            st.metric("ML Normal Transactions", normal_count_ml)
                        with col4:
                            fraud_rate_ml = (fraud_count_ml / len(predictions)) * 100 if predictions.size > 0 else 0
                            st.metric("ML Fraud Rate", f"{fraud_rate_ml:.1f}%")
                        
                        # Detailed ML results
                        st.subheader("ML Analysis Results")
                        
                        for i, (txn, pred, prob) in enumerate(zip(st.session_state.transactions, predictions, probabilities)):
                            is_fraud_ml = pred == 1
                            
                            if is_fraud_ml:
                                with st.expander(f"🤖 ML Fraud Alert - Transaction {i+1} (Probability: {prob:.3f})"):
                                    st.json(txn)
                                    st.markdown(f"**ML Analysis:** This transaction has a {prob:.1%} probability of being fraudulent according to the {model_choice} model.")
                                    
                                    # Compare with rule-based
                                    if 'analysis_results' in st.session_state:
                                        rule_based_result = st.session_state.analysis_results[i]
                                        st.markdown(f"**Rule-based result:** {'Fraud' if rule_based_result['is_fraud'] else 'Normal'} (Similarity: {rule_based_result['similarity']:.3f})")
                            else:
                                with st.expander(f"🤖 ML Normal - Transaction {i+1} (Probability: {prob:.3f})"):
                                    st.json(txn)
                                    st.info(f"This transaction has a {prob:.1%} probability of being fraudulent according to the {model_choice} model, which is below the threshold of {threshold:.1%}.")
                        
                    except Exception as e:
                        st.error(f"Error in ML analysis: {e}")
        else:
            st.info("Please upload transactions and train models first")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if index is not None and fraud_meta:
            # Fraud patterns analysis
            st.subheader("Known Fraud Patterns")
            
            # Convert fraud data to DataFrame for analysis
            fraud_df = pd.DataFrame(fraud_meta)
            
            if not fraud_df.empty:
                # Amount distribution
                if 'amount' in fraud_df.columns:
                    st.subheader("Fraud Amount Distribution")
                    try:
                        amounts = pd.to_numeric(fraud_df['amount'], errors='coerce').dropna()
                        if not amounts.empty:
                            fig = px.histogram(amounts, title="Distribution of Fraud Amounts")
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not analyze amount distribution")
                
                # Location analysis
                if 'location' in fraud_df.columns:
                    st.subheader("Fraud by Location")
                    location_counts = fraud_df['location'].value_counts()
                    fig = px.bar(x=location_counts.index, y=location_counts.values, 
                               title="Fraud Cases by Location")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time analysis
                time_column = None
                if 'timestamp' in fraud_df.columns:
                    time_column = 'timestamp'
                elif 'time' in fraud_df.columns:
                    time_column = 'time'
                
                if time_column:
                    st.subheader("Fraud by Time")
                    try:
                        fraud_df['hour'] = pd.to_datetime(fraud_df[time_column]).dt.hour
                        hour_counts = fraud_df['hour'].value_counts().sort_index()
                        fig = px.line(x=hour_counts.index, y=hour_counts.values, 
                                    title="Fraud Cases by Hour of Day")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not analyze time patterns")
                
                # Merchant analysis
                if 'merchant' in fraud_df.columns:
                    st.subheader("Fraud by Merchant")
                    merchant_counts = fraud_df['merchant'].value_counts().head(10)
                    fig = px.bar(x=merchant_counts.index, y=merchant_counts.values, 
                               title="Top 10 Fraudulent Merchants")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category analysis
                if 'category' in fraud_df.columns:
                    st.subheader("Fraud by Category")
                    category_counts = fraud_df['category'].value_counts()
                    fig = px.pie(values=category_counts.values, names=category_counts.index, 
                               title="Fraud Distribution by Category")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud data available for analytics. Upload some fraud cases first.")

if __name__ == "__main__":
    main() 