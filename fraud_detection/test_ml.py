#!/usr/bin/env python3
"""
Test script for the ML-based fraud detection system
"""

import json
from ml_models import FraudDetectionML

def test_ml_system():
    """Test the ML fraud detection system"""
    print("🤖 Testing ML Fraud Detection System")
    print("=" * 50)
    
    # Initialize ML system
    ml_system = FraudDetectionML()
    
    # Load sample data
    print("📊 Loading training data...")
    try:
        with open("sample_transactions.json", "r") as f:
            transactions = json.load(f)
        
        print(f"Loaded {len(transactions)} transactions")
        
        # Use all transactions for testing (sample_transactions.json is already small)
        sample_transactions = transactions
        print(f"Using {len(sample_transactions)} transactions for testing")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train models
    print("\n🎯 Training ML models...")
    try:
        performance = ml_system.train_models(sample_transactions)
        
        print("\n📈 Model Performance:")
        for model_name, metrics in performance.items():
            print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}")
        
    except Exception as e:
        print(f"Error training models: {e}")
        return
    
    # Test predictions
    print("\n🔍 Testing predictions...")
    try:
        # Use a small sample for prediction
        test_transactions = sample_transactions[:10]
        
        for model_name in ['random_forest', 'logistic_regression']:
            predictions, probabilities = ml_system.predict_fraud(test_transactions, model_name)
            
            print(f"\n{model_name.upper()} Results:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                status = "FRAUD" if pred == 1 else "NORMAL"
                print(f"Transaction {i+1}: {status} (Probability: {prob:.3f})")
    
    except Exception as e:
        print(f"Error in predictions: {e}")
    
    # Feature importance
    print("\n📊 Feature Importance (Random Forest):")
    try:
        importance_df = ml_system.get_feature_importance('random_forest')
        if importance_df is not None:
            print("Top 10 most important features:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    # Save models
    print("\n💾 Saving models...")
    try:
        ml_system.save_models()
        print("Models saved successfully!")
    except Exception as e:
        print(f"Error saving models: {e}")
    
    print("\n✅ ML system test completed!")

if __name__ == "__main__":
    test_ml_system() 