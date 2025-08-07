# 🛡️ FraudShield

**Advanced AI-Powered Fraud Detection & Analytics Platform**

FraudShield is a comprehensive fraud detection system that combines rule-based pattern matching with machine learning models to identify fraudulent transactions in real-time.

## 🚀 Features

- **Rule-Based Detection**: Pattern matching against known fraud vectors
- **ML Models**: Random Forest, Logistic Regression, SVM, Isolation Forest
- **Real-Time Analytics**: Interactive dashboards and visualizations
- **Comprehensive Analysis**: Detailed fraud explanations and risk indicators
- **User-Friendly Interface**: Streamlit-based web application

## 📋 Requirements

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- Plotly
- FAISS
- NumPy

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd UC2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   cd fraud_detection
   streamlit run app.py
   ```

## 📊 Usage

### 1. Upload & Analyze
- Upload transaction JSON files
- View rule-based fraud detection results
- See detailed similarity scores and explanations

### 2. ML Models
- Train machine learning models on your data
- Compare model performance
- Run ML-based fraud predictions

### 3. Analytics Dashboard
- View fraud patterns by location, time, merchant, and category
- Interactive charts and visualizations
- Comprehensive fraud analytics

## 📁 Project Structure

```
UC2/
├── fraud_detection/
│   ├── app.py              # Main Streamlit application
│   ├── ml_models.py        # ML system implementation
│   ├── sample_transactions.json
│   ├── fraud_vectors.json  # Known fraud patterns
│   └── models/             # Saved ML models
├── backend/                # Backend services
├── data_generator.py       # Sample data generator
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

- **Fraud Detection Threshold**: Adjustable sensitivity (0.0-1.0)
- **Model Training**: Customizable ML model parameters
- **Analytics**: Configurable chart sizes and styling

---

**FraudShield** - Protecting transactions with advanced ML Models 🛡️ 