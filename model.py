"""
model.py - Hybrid Deep Learning / Machine Learning Classification
==================================================================
Implements the hybrid architecture defined in the project abstract:
1. 1D CNN (TensorFlow/Keras) acts as a Deep Feature Extractor.
2. Random Forest acts as the primary, interpretable classifier.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

def build_cnn_feature_extractor(input_shape=(200, 1)):
    """
    Builds a shallow 1D CNN to learn deep physiological features.
    We remove the final prediction layer so it outputs an array of features 
    instead of a final classification.
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional Block 1
    x = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Convolutional Block 2
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Flatten to a 1D feature vector
    features = Flatten(name="deep_features")(x)
    
    # Final classification layer (used ONLY for initial training)
    outputs = Dense(1, activation='sigmoid', name="classifier")(features)
    
    full_model = Model(inputs=inputs, outputs=outputs)
    full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # The Extractor Model: Outputs the flattened layer, skipping the final Dense layer
    feature_extractor = Model(inputs=full_model.input, outputs=full_model.get_layer("deep_features").output)
    
    return full_model, feature_extractor

def train_hybrid_system(X_raw_signals, y_labels):
    print("--- Training Hybrid DL/ML System ---")
    # 1. Reshape for CNN
    X_cnn = np.expand_dims(X_raw_signals, axis=2)
    # 2. Extract Deep Features from CNN
    full_cnn, feature_extractor = build_cnn_feature_extractor()
    full_cnn.fit(X_cnn, y_labels, epochs=30, batch_size=32, verbose=1) # Increased epochs to 30
    deep_features = feature_extractor.predict(X_cnn)
    # 3. Extract Hand-Crafted Features from features.py
    from features import extract_features_all
    hand_crafted_features, _ = extract_features_all(X_raw_signals)
    # 4. Combine both (The Hybrid Step)
    X_combined = np.hstack([deep_features, hand_crafted_features])
    # 5. Train Random Forest on the Combined Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_labels, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    # Save everything
    full_cnn.save("cnn_extractor.h5")
    joblib.dump({'ml_model': rf, 'scaler': scaler}, "ml_pipeline.joblib")
    print("✅ Hybrid models retrained with new features!")

def predict_single_heartbeat(raw_signal):
    """Predicts a single 200-sample heartbeat using the hybrid pipeline."""
    # 1. Load models
    cnn = tf.keras.models.load_model("cnn_extractor.h5")
    extractor = Model(inputs=cnn.input, outputs=cnn.get_layer("deep_features").output)
    pipeline = joblib.load("ml_pipeline.joblib")
    
    # 2. Extract Deep Features (from CNN)
    signal_cnn = raw_signal.reshape(1, 200, 1)
    deep_features = extractor.predict(signal_cnn, verbose=0) # Shape is usually (1, 3000)
    
    # 3. Extract Hand-crafted Features (from features.py)
    # We must match exactly what was used during training
    from features import extract_features_single
    hand_features = extract_features_single(raw_signal).reshape(1, -1) # Shape is (1, 8 or 10)
    
    # 4. COMBINE THEM (This fixes the 3008 vs 3018 error!)
    X_combined = np.hstack([deep_features, hand_features])
    
    # 5. Scale and Classify
    features_scaled = pipeline['scaler'].transform(X_combined)
    probs = pipeline['ml_model'].predict_proba(features_scaled)[0]
    
    # Sensitivity threshold (0.3 makes it easier to catch Abnormal beats)
    prediction = 1 if probs[1] > 0.5 else 0
    confidence = probs[prediction]
    
    label = "Normal Rhythm" if prediction == 0 else "Abnormal Rhythm Detected"
    return prediction, confidence, label