"""
ASD (Autism Spectrum Disorder) Prediction Machine Learning Model
This script loads, preprocesses, and trains ML models to predict ASD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ASDPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        return df
    
    def explore_data(self, df):
        """Explore the dataset"""
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Info:")
        print(df.info())
        
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
        # Handle different target column names
        target_col = None
        if 'Class/ASD' in df.columns:
            target_col = 'Class/ASD'
        elif 'Class' in df.columns:
            target_col = 'Class'
        
        if target_col:
            print(f"\nTarget Variable Distribution:")
            print(df[target_col].value_counts())
            print(df[target_col].value_counts(normalize=True) * 100)
        
        print(f"\nNumerical Features Statistics:")
        print(df.describe())
        
        print(f"\nCategorical Features:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:  # Don't print target as categorical feature
                print(f"\n{col}:")
                print(df[col].value_counts())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle different column names - normalize to common format
        # Map old column names to new format if needed
        column_mapping = {}
        if 'A1_Score' in df_processed.columns:
            # Old format - map to new format
            for i in range(1, 11):
                old_col = f'A{i}_Score'
                new_col = f'A{i}'
                if old_col in df_processed.columns:
                    column_mapping[old_col] = new_col
        
        if 'age' in df_processed.columns:
            column_mapping['age'] = 'Age'
        if 'gender' in df_processed.columns:
            column_mapping['gender'] = 'Sex'
        if 'jundice' in df_processed.columns:
            column_mapping['jundice'] = 'Jauundice'
        if 'austim' in df_processed.columns:
            column_mapping['austim'] = 'Family_ASD'
        if 'Class/ASD' in df_processed.columns:
            column_mapping['Class/ASD'] = 'Class'
        
        # Rename columns
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Remove unnecessary columns
        columns_to_drop = ['id', 'age_desc', 'result', 'contry_of_res', 'used_app_before', 'relation']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        
        # Handle missing values - replace '?' with mode or most frequent value
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Replace '?' with the most frequent value
                mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col] = df_processed[col].replace('?', mode_value)
        
        # Separate features and target
        target_col = 'Class' if 'Class' in df_processed.columns else 'Class/ASD'
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        # Encode target variable (handle both YES/NO and 0/1 formats)
        if y.dtype == 'object':
            y_encoded = (y.str.upper() == 'YES').astype(int)
        else:
            y_encoded = y.astype(int)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"\nProcessed features: {len(self.feature_names)}")
        print(f"Features: {self.feature_names}")
        print(f"\nTarget distribution:")
        print(f"NO: {(y_encoded == 0).sum()} ({(y_encoded == 0).mean()*100:.2f}%)")
        print(f"YES: {(y_encoded == 1).sum()} ({(y_encoded == 1).mean()*100:.2f}%)")
        
        return X_encoded, y_encoded
    
    def train_models(self, X, y):
        """Train multiple ML models and select the best one"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to test with stronger regularization to prevent overfitting
        # Note: Dataset has strong separation - using conservative parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=5, 
                min_samples_split=20, min_samples_leaf=10, max_features='sqrt'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50, random_state=42, max_depth=3,
                learning_rate=0.05, min_samples_split=20, min_samples_leaf=10,
                subsample=0.8  # Add subsampling for regularization
            ),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=0.5, gamma='scale'),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, C=0.5, penalty='l2'
            )
        }
        
        results = {}
        
        print("\nTraining and evaluating models...")
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train if name not in ['SVM', 'Logistic Regression'] else X_train_scaled, 
                                       y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Select best model based on ROC-AUC
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.model = results[best_model_name]['model']
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        print(f"CV Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']*2:.4f})")
        
        # Detailed classification report for best model
        print(f"\nClassification Report:")
        print(classification_report(y_test, results[best_model_name]['y_pred'], 
                                   target_names=['NO', 'YES']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nTop 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))
        
        return X_test, y_test, results[best_model_name], best_model_name
    
    def save_model(self, filepath='asd_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='asd_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def predict(self, X_new):
        """Make predictions on new data"""
        # Preprocess new data
        X_processed = X_new.copy()
        
        # Encode categorical variables
        for col in self.label_encoders:
            if col in X_processed.columns:
                # Handle unseen categories
                le = self.label_encoders[col]
                X_processed[col] = X_processed[col].astype(str)
                # Replace unseen values with most common value
                known_classes = set(le.classes_)
                X_processed[col] = X_processed[col].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                X_processed[col] = le.transform(X_processed[col])
        
        # Ensure all features are present and in correct order
        X_processed = X_processed[self.feature_names]
        
        # Scale if needed
        if hasattr(self.model, 'decision_function'):  # SVM or Logistic Regression
            X_processed = self.scaler.transform(X_processed)
        
        # Make prediction
        prediction = self.model.predict(X_processed)
        prediction_proba = self.model.predict_proba(X_processed)
        
        return prediction, prediction_proba


def main():
    """Main function to run the complete pipeline"""
    # Initialize predictor
    predictor = ASDPredictor()
    
    # Load data - use the combined dataset
    df = predictor.load_data('Autism_Screening_Data_Combined.csv')
    
    # Explore data
    predictor.explore_data(df)
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    
    # Train models
    X_test, y_test, best_results, best_model_name = predictor.train_models(X, y)
    
    # Save model
    predictor.save_model('asd_model.pkl')
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE!")
    print("="*50)
    print(f"Best model: {best_model_name}")
    print(f"Model saved as: asd_model.pkl")
    print(f"You can now use the model for predictions!")


if __name__ == "__main__":
    main()

