import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionML:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.model_performance = {}
        
    def extract_features(self, transactions):
        """Extract comprehensive features from transactions"""
        features = []
        
        for txn in transactions:
            feature_dict = {}
            
            # Basic numerical features
            feature_dict['amount'] = float(txn.get('amount', 0))
            feature_dict['amount_log'] = np.log1p(float(txn.get('amount', 0)))
            
            # Time-based features
            if 'timestamp' in txn:
                try:
                    dt = datetime.fromisoformat(txn['timestamp'].replace('Z', '+00:00'))
                    feature_dict['hour'] = dt.hour
                    feature_dict['day_of_week'] = dt.weekday()
                    feature_dict['month'] = dt.month
                    feature_dict['is_weekend'] = 1 if dt.weekday() >= 5 else 0
                    feature_dict['is_night'] = 1 if dt.hour < 6 or dt.hour > 22 else 0
                except:
                    feature_dict['hour'] = 12
                    feature_dict['day_of_week'] = 0
                    feature_dict['month'] = 1
                    feature_dict['is_weekend'] = 0
                    feature_dict['is_night'] = 0
            
            # Location features
            if 'location' in txn and isinstance(txn['location'], dict):
                location = txn['location']
                feature_dict['city_encoded'] = hash(location.get('city', '')) % 1000
                feature_dict['country_encoded'] = hash(location.get('country', '')) % 100
                
                # Distance from home (assuming Bangalore as home)
                if 'geocoordinates' in location:
                    coords = location['geocoordinates']
                    if len(coords) == 2:
                        # Simple distance calculation (you can use haversine for accuracy)
                        home_coords = [12.9716, 77.5946]  # Bangalore
                        distance = np.sqrt((coords[0] - home_coords[0])**2 + (coords[1] - home_coords[1])**2)
                        feature_dict['distance_from_home'] = distance
                    else:
                        feature_dict['distance_from_home'] = 0
                else:
                    feature_dict['distance_from_home'] = 0
            else:
                feature_dict['city_encoded'] = 0
                feature_dict['country_encoded'] = 0
                feature_dict['distance_from_home'] = 0
            
            # Payment method features
            payment_method = txn.get('payment_method', '').lower()
            feature_dict['is_cash'] = 1 if 'cash' in payment_method else 0
            feature_dict['is_card'] = 1 if 'card' in payment_method else 0
            feature_dict['is_upi'] = 1 if 'upi' in payment_method else 0
            feature_dict['is_online'] = 1 if 'online' in payment_method else 0
            
            # Category features
            category = txn.get('category', '').lower()
            feature_dict['is_shopping'] = 1 if 'shopping' in category else 0
            feature_dict['is_entertainment'] = 1 if 'entertainment' in category else 0
            feature_dict['is_education'] = 1 if 'education' in category else 0
            feature_dict['is_food'] = 1 if 'food' in category else 0
            feature_dict['is_transport'] = 1 if 'transport' in category else 0
            
            # Merchant features
            merchant = txn.get('merchant_name', '').lower()
            feature_dict['merchant_encoded'] = hash(merchant) % 1000
            
            # Recurrence features
            if 'recurrence' in txn:
                recurrence = txn['recurrence']
                feature_dict['is_recurring'] = 1 if recurrence.get('is_recurring', False) else 0
                frequency = recurrence.get('frequency', '').lower()
                feature_dict['is_monthly'] = 1 if 'monthly' in frequency else 0
                feature_dict['is_weekly'] = 1 if 'weekly' in frequency else 0
            else:
                feature_dict['is_recurring'] = 0
                feature_dict['is_monthly'] = 0
                feature_dict['is_weekly'] = 0
            
            # Tags features
            tags = txn.get('tags', [])
            feature_dict['has_urgent_tag'] = 1 if 'urgent' in tags else 0
            feature_dict['has_business_tag'] = 1 if 'business' in tags else 0
            feature_dict['has_personal_tag'] = 1 if 'personal' in tags else 0
            feature_dict['has_gift_tag'] = 1 if 'gift' in tags else 0
            feature_dict['num_tags'] = len(tags)
            
            # Metadata features
            if 'metadata' in txn:
                metadata = txn['metadata']
                feature_dict['loyalty_points'] = metadata.get('loyalty_points_earned', 0)
                feature_dict['has_invoice'] = 1 if metadata.get('invoice_details') else 0
                
                # Itemized breakdown features
                itemized = metadata.get('itemized_breakdown', [])
                feature_dict['num_items'] = len(itemized)
                if itemized:
                    prices = [item.get('price', 0) for item in itemized]
                    feature_dict['avg_item_price'] = np.mean(prices)
                    feature_dict['max_item_price'] = np.max(prices)
                else:
                    feature_dict['avg_item_price'] = 0
                    feature_dict['max_item_price'] = 0
            else:
                feature_dict['loyalty_points'] = 0
                feature_dict['has_invoice'] = 0
                feature_dict['num_items'] = 0
                feature_dict['avg_item_price'] = 0
                feature_dict['max_item_price'] = 0
            
            # Currency features
            currency = txn.get('currency', '').upper()
            feature_dict['is_usd'] = 1 if currency == 'USD' else 0
            feature_dict['is_eur'] = 1 if currency == 'EUR' else 0
            feature_dict['is_inr'] = 1 if currency == 'INR' else 0
            
            # User behavior features (placeholder - would need user history)
            feature_dict['user_encoded'] = hash(txn.get('user_id', '')) % 1000
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def create_synthetic_labels(self, transactions, fraud_rate=0.1):
        """Create synthetic fraud labels based on heuristics"""
        labels = []
        
        for txn in transactions:
            is_fraud = False
            
            # High amount transactions
            amount = float(txn.get('amount', 0))
            if amount > 5000:
                is_fraud = np.random.random() < 0.3
            
            # Night transactions
            if 'timestamp' in txn:
                try:
                    dt = datetime.fromisoformat(txn['timestamp'].replace('Z', '+00:00'))
                    if dt.hour < 6 or dt.hour > 22:
                        is_fraud = is_fraud or np.random.random() < 0.2
                except:
                    pass
            
            # Suspicious locations
            if 'location' in txn and isinstance(txn['location'], dict):
                location = txn['location']
                city = location.get('city', '').lower()
                if any(susp in city for susp in ['unknown', 'offshore', 'tax']):
                    is_fraud = True
            
            # Cash payments with high amounts
            payment_method = txn.get('payment_method', '').lower()
            if 'cash' in payment_method and amount > 1000:
                is_fraud = is_fraud or np.random.random() < 0.4
            
            # Random fraud injection
            if np.random.random() < fraud_rate:
                is_fraud = True
            
            labels.append(1 if is_fraud else 0)
        
        return np.array(labels)
    
    def train_models(self, transactions, labels=None):
        """Train multiple ML models for fraud detection"""
        # Extract features
        X = self.extract_features(transactions)
        self.feature_names = X.columns.tolist()
        
        # Create labels if not provided
        if labels is None:
            labels = self.create_synthetic_labels(transactions)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Train models
        models_to_train = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42),
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            if name == 'isolation_forest':
                # Isolation Forest doesn't use labels
                model.fit(X_train_scaled)
                # Convert predictions: -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0
                y_pred = (model.predict(X_test_scaled) == -1).astype(int)
                y_pred_proba = model.decision_function(X_test_scaled)
                # Convert to probability-like scores
                y_pred_proba = 1 - (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.models[name] = model
            self.model_performance[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        return self.model_performance
    
    def predict_fraud(self, transactions, model_name='random_forest'):
        """Predict fraud for new transactions"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Extract features
        X = self.extract_features(transactions)
        
        # Scale features
        if 'standard' in self.scalers:
            X_scaled = self.scalers['standard'].transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        model = self.models[model_name]
        
        if model_name == 'isolation_forest':
            # Isolation Forest predictions
            predictions = (model.predict(X_scaled) == -1).astype(int)
            scores = model.decision_function(X_scaled)
            # Convert to probability-like scores
            probabilities = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # Standard classifier predictions
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for the specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_models(self, filepath='fraud_detection/models/'):
        """Save trained models"""
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(f"{filepath}{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            with open(f"{filepath}{name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_performance': {
                name: {
                    'accuracy': float(metrics['accuracy']),
                    'auc': float(metrics['auc'])
                }
                for name, metrics in self.model_performance.items()
            }
        }
        with open(f"{filepath}metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_models(self, filepath='fraud_detection/models/'):
        """Load trained models"""
        # Load models
        for name in ['random_forest', 'logistic_regression', 'svm', 'isolation_forest']:
            try:
                with open(f"{filepath}{name}.pkl", 'rb') as f:
                    self.models[name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Model {name} not found")
        
        # Load scalers
        try:
            with open(f"{filepath}standard_scaler.pkl", 'rb') as f:
                self.scalers['standard'] = pickle.load(f)
        except FileNotFoundError:
            print("Scaler not found")
        
        # Load metadata
        try:
            with open(f"{filepath}metadata.json", 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.model_performance = metadata.get('model_performance', {})
        except FileNotFoundError:
            print("Metadata not found") 