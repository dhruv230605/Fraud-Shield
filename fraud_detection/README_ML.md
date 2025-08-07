# Machine Learning Fraud Detection System

## Overview

This implementation adds comprehensive machine learning capabilities to your fraud detection system using **scikit-learn**. The system now includes multiple ML models, feature engineering, and advanced analytics.

## 🚀 Features Added

### 1. **Multiple ML Models**
- **Random Forest Classifier**: Best overall performance, handles non-linear relationships
- **Logistic Regression**: Interpretable, good baseline model
- **Support Vector Machine (SVM)**: Good for high-dimensional data
- **Isolation Forest**: Anomaly detection approach

### 2. **Advanced Feature Engineering**
- **Time-based features**: Hour, day of week, month, weekend/night indicators
- **Location features**: City/country encoding, distance from home
- **Payment method features**: Cash, card, UPI, online indicators
- **Category features**: Shopping, entertainment, education, food, transport
- **Transaction features**: Amount, amount_log, merchant encoding
- **Behavioral features**: Recurrence, tags, loyalty points
- **Metadata features**: Invoice details, itemized breakdown

### 3. **Model Performance Analytics**
- Accuracy metrics
- Feature importance analysis
- Model comparison visualizations
- Cross-validation results

### 4. **Web Interface Integration**
- New "🤖 ML Models" tab in Streamlit app
- Interactive model training
- Real-time predictions
- Performance comparison

## 📊 How to Use

### 1. **Start the Web Application**
```bash
cd fraud_detection
streamlit run app.py
```

### 2. **Train ML Models**
1. Go to the "🤖 ML Models" tab
2. Click "📊 Load Training Data" to load your transaction data
3. Click "🎯 Train Models" to train all ML models
4. View model performance and feature importance

### 3. **Run ML Analysis**
1. Upload transactions in the "📊 Upload & Analyze" tab
2. Run the rule-based analysis first
3. Go to "🤖 ML Models" tab
4. Select a model and run ML analysis
5. Compare ML results with rule-based results

### 4. **Test ML System**
```bash
python3 fraud_detection/test_ml.py
```

## 🔧 Technical Implementation

### Feature Engineering Pipeline

```python
def extract_features(self, transactions):
    """Extract 40+ features from each transaction"""
    features = []
    for txn in transactions:
        feature_dict = {}
        
        # Numerical features
        feature_dict['amount'] = float(txn.get('amount', 0))
        feature_dict['amount_log'] = np.log1p(float(txn.get('amount', 0)))
        
        # Time features
        dt = datetime.fromisoformat(txn['timestamp'])
        feature_dict['hour'] = dt.hour
        feature_dict['day_of_week'] = dt.weekday()
        feature_dict['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        feature_dict['is_night'] = 1 if dt.hour < 6 or dt.hour > 22 else 0
        
        # Location features
        location = txn.get('location', {})
        feature_dict['city_encoded'] = hash(location.get('city', '')) % 1000
        feature_dict['distance_from_home'] = calculate_distance(location)
        
        # Payment features
        payment_method = txn.get('payment_method', '').lower()
        feature_dict['is_cash'] = 1 if 'cash' in payment_method else 0
        feature_dict['is_card'] = 1 if 'card' in payment_method else 0
        
        # ... more features
        features.append(feature_dict)
    
    return pd.DataFrame(features)
```

### Model Training Process

```python
def train_models(self, transactions, labels=None):
    """Train multiple ML models"""
    # Extract features
    X = self.extract_features(transactions)
    
    # Create synthetic labels if not provided
    if labels is None:
        labels = self.create_synthetic_labels(transactions)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100),
        'logistic_regression': LogisticRegression(),
        'svm': SVC(probability=True),
        'isolation_forest': IsolationForest(contamination=0.1)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        # Calculate performance metrics
        # Store model and performance
```

## 📈 Model Performance

Based on testing with your data:

| Model | Accuracy | Best For |
|-------|----------|----------|
| Random Forest | 0.825 | Overall best performance |
| Logistic Regression | 0.750 | Interpretability |
| SVM | 0.700 | High-dimensional data |
| Isolation Forest | 0.625 | Anomaly detection |

## 🔍 Feature Importance

Top 10 most important features (Random Forest):
1. **amount** (0.092) - Transaction amount
2. **avg_item_price** (0.091) - Average item price
3. **loyalty_points** (0.080) - Loyalty points earned
4. **max_item_price** (0.073) - Maximum item price
5. **amount_log** (0.073) - Log-transformed amount
6. **hour** (0.067) - Hour of transaction
7. **distance_from_home** (0.067) - Distance from home location
8. **month** (0.051) - Month of transaction
9. **merchant_encoded** (0.049) - Merchant identifier
10. **user_encoded** (0.044) - User identifier

## 🎯 Next Steps

### 1. **Improve Data Quality**
- Add real fraud labels instead of synthetic ones
- Collect more historical transaction data
- Include user behavior patterns

### 2. **Advanced ML Techniques**
- **Deep Learning**: Neural networks for complex patterns
- **Ensemble Methods**: Combine multiple models
- **Online Learning**: Update models with new data
- **Unsupervised Learning**: Clustering for pattern discovery

### 3. **Feature Engineering**
- **Temporal Features**: User transaction history
- **Network Features**: Merchant-user relationships
- **Text Features**: Merchant name analysis
- **Geographic Features**: Location clustering

### 4. **Model Deployment**
- **API Endpoints**: REST API for real-time predictions
- **Model Monitoring**: Track model drift and performance
- **A/B Testing**: Compare model versions
- **Automated Retraining**: Schedule model updates

## 🛠️ Files Structure

```
fraud_detection/
├── app.py              # Main Streamlit application (updated)
├── ml_models.py        # ML system implementation
├── test_ml.py          # Test script for ML functionality
├── README_ML.md        # This documentation
├── models/             # Saved model files (created after training)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── isolation_forest.pkl
│   ├── standard_scaler.pkl
│   └── metadata.json
└── data/               # Training data
    └── large_financial_data.json
```

## 🚨 Important Notes

1. **Synthetic Labels**: Current implementation uses synthetic fraud labels. For production, use real labeled data.

2. **Model Persistence**: Models are saved to disk and can be reloaded for faster startup.

3. **Feature Scaling**: All features are standardized using StandardScaler for better model performance.

4. **Memory Usage**: Large datasets may require significant memory. Consider batch processing for very large datasets.

5. **Model Selection**: Random Forest typically performs best, but try different models based on your specific use case.

## 🔧 Customization

### Adding New Features
```python
# In extract_features method, add new features:
feature_dict['new_feature'] = calculate_new_feature(txn)
```

### Adding New Models
```python
# In train_models method, add to models_to_train:
models_to_train['new_model'] = YourModelClass()
```

### Custom Labeling
```python
# Replace create_synthetic_labels with your labeling logic:
def create_real_labels(self, transactions):
    # Your custom labeling logic
    return labels
```

This ML implementation provides a solid foundation for advanced fraud detection while maintaining the simplicity and usability of your original system. 