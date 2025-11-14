"""
ASD Prediction Script
Use this script to make predictions on new data using the trained model
"""

import pandas as pd
import joblib
import sys

def predict_asd(input_data):
    """
    Predict ASD for given input data
    
    Parameters:
    input_data: dict or pandas DataFrame with the following features:
        - A1 to A10 (or A1_Score to A10_Score): 0 or 1
        - Age (or age): numeric
        - Sex (or gender): 'm' or 'f'
        - Jauundice (or jundice): 'yes' or 'no'
        - Family_ASD (or austim): 'yes' or 'no'
        - Optional: ethnicity, contry_of_res, used_app_before, relation
    
    Returns:
        List of dictionaries with 'ASD_Prediction' and 'Probability'
    """
    # Load model
    try:
        model_data = joblib.load('asd_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        feature_names = model_data['feature_names']
    except FileNotFoundError:
        print("Error: Model file 'asd_model.pkl' not found. Please train the model first.")
        return None
    
    # Convert input to DataFrame if it's a dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Normalize column names to match training data format
    column_mapping = {}
    # Map old format to new format if needed
    for i in range(1, 11):
        old_col = f'A{i}_Score'
        new_col = f'A{i}'
        if old_col in df.columns:
            column_mapping[old_col] = new_col
    
    if 'age' in df.columns:
        column_mapping['age'] = 'Age'
    if 'gender' in df.columns:
        column_mapping['gender'] = 'Sex'
    if 'jundice' in df.columns:
        column_mapping['jundice'] = 'Jauundice'
    if 'austim' in df.columns:
        column_mapping['austim'] = 'Family_ASD'
    
    df = df.rename(columns=column_mapping)
    
    # Remove target column if present
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    if 'Class/ASD' in df.columns:
        df = df.drop('Class/ASD', axis=1)
    
    # Preprocess
    X_processed = df.copy()
    
    # Encode categorical variables
    for col in label_encoders:
        if col in X_processed.columns:
            le = label_encoders[col]
            X_processed[col] = X_processed[col].astype(str)
            # Handle unseen categories
            known_classes = set(le.classes_)
            X_processed[col] = X_processed[col].apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            X_processed[col] = le.transform(X_processed[col])
    
    # Ensure all features are present and in correct order
    X_processed = X_processed[feature_names]
    
    # Convert to numpy array but preserve feature names for model
    X_array = X_processed.values
    
    # Scale if needed (for SVM or Logistic Regression)
    if hasattr(model, 'decision_function'):
        X_array = scaler.transform(X_array)
    
    # Convert back to DataFrame with feature names for model compatibility
    X_processed = pd.DataFrame(X_array, columns=feature_names)
    
    # Make prediction
    prediction = model.predict(X_processed)
    prediction_proba = model.predict_proba(X_processed)
    
    # Convert to readable format
    results = []
    for i in range(len(prediction)):
        asd_status = 'YES' if prediction[i] == 1 else 'NO'
        probability = prediction_proba[i][1] * 100  # Probability of YES
        results.append({
            'ASD_Prediction': asd_status,
            'Probability': f"{probability:.2f}%"
        })
    
    return results


def example_usage():
    """Example of how to use the prediction function"""
    # Example 1: Single prediction
    example_data = {
        'A1_Score': 1,
        'A2_Score': 1,
        'A3_Score': 1,
        'A4_Score': 1,
        'A5_Score': 1,
        'A6_Score': 1,
        'A7_Score': 1,
        'A8_Score': 1,
        'A9_Score': 1,
        'A10_Score': 1,
        'age': 25,
        'gender': 'm',
        'ethnicity': 'White-European',
        'jundice': 'no',
        'austim': 'no',
        'contry_of_res': 'United States',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    
    results = predict_asd(example_data)
    if results:
        print("Example Prediction:")
        print(f"ASD Prediction: {results[0]['ASD_Prediction']}")
        print(f"Probability: {results[0]['Probability']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If CSV file provided, predict on all rows
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        results = predict_asd(df)
        if results:
            df_results = pd.DataFrame(results)
            output_df = pd.concat([df, df_results], axis=1)
            output_file = csv_file.replace('.csv', '_predictions.csv')
            output_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
    else:
        # Run example
        example_usage()

