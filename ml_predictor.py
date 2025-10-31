"""
Machine Learning predictor using Enhanced LSTM with attention mechanism.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Machine Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    ML_AVAILABLE = True
except ImportError:
    class Model:
        pass
    class EarlyStopping:
        pass
    ML_AVAILABLE = False

from config import MLConfig


class EnhancedLSTMPredictor:
    """Enhanced LSTM model with attention mechanism for return prediction"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_history = None
    
    def _create_attention_layer(self, lstm_output: tf.Tensor) -> tf.Tensor:
        """Create attention mechanism for LSTM output"""
        attention_weights = layers.Dense(1, activation='tanh')(lstm_output)
        attention_weights = layers.Flatten()(attention_weights)
        attention_weights = layers.Activation('softmax')(attention_weights)
        attention_weights = layers.RepeatVector(lstm_output.shape[-1])(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)
        
        attended_output = layers.Multiply()([lstm_output, attention_weights])
        attended_output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_output)
        
        return attended_output
    
    def build_model(self) -> Model:
        """Build LSTM model with attention mechanism"""
        if not ML_AVAILABLE:
            return None
        
        inputs = keras.Input(shape=(self.config.sequence_length, 1), name='price_input')
        
        lstm1 = layers.LSTM(
            self.config.lstm_units[0],
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate,
            name='lstm_1'
        )(inputs)
        
        lstm1_norm = layers.BatchNormalization(name='batch_norm_1')(lstm1)
        
        lstm2 = layers.LSTM(
            self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else 64,
            return_sequences=True,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.dropout_rate,
            name='lstm_2'
        )(lstm1_norm)
        
        lstm2_norm = layers.BatchNormalization(name='batch_norm_2')(lstm2)
        
        attended_output = self._create_attention_layer(lstm2_norm)
        
        dropout_layer = layers.Dropout(self.config.dropout_rate, name='dropout_final')(attended_output)
        
        dense1 = layers.Dense(32, activation='relu', name='dense_prediction')(dropout_layer)
        dense1_norm = layers.BatchNormalization(name='batch_norm_prediction')(dense1)
        dense1_dropout = layers.Dropout(self.config.dropout_rate, name='dropout_prediction')(dense1_norm)
        
        outputs = layers.Dense(self.config.prediction_horizon, activation='linear', name='return_prediction')(dense1_dropout)
        
        model = Model(inputs=inputs, outputs=outputs, name='enhanced_lstm_predictor')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences for training/prediction"""
        values = data.values
        scaled_values = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(scaled_values) - self.config.sequence_length - self.config.prediction_horizon + 1):
            X.append(scaled_values[i:(i + self.config.sequence_length)])
            y.append(scaled_values[(i + self.config.sequence_length):(i + self.config.sequence_length + self.config.prediction_horizon)])
        
        X = np.array(X).reshape(-1, self.config.sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def train(self, returns_data: pd.DataFrame, validation_split: float = 0.2) -> dict:
        """Train the LSTM model"""
        if not ML_AVAILABLE:
            return {}
        
        if self.model is None:
            self.model = self.build_model()
        
        all_X, all_y = [], []
        
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            if len(asset_returns) < self.config.sequence_length + self.config.prediction_horizon:
                continue
            
            X, y = self._prepare_sequences(asset_returns)
            all_X.append(X)
            all_y.append(y)
        
        if not all_X:
            return {}
        
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_combined, y_combined,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        return self.training_history
    
    def predict(self, recent_returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict future returns for given assets"""
        if not ML_AVAILABLE or not self.is_trained or self.model is None:
            return {}
        
        predictions = {}
        
        for asset in recent_returns.columns:
            asset_returns = recent_returns[asset].dropna()
            
            if len(asset_returns) < self.config.sequence_length:
                continue
            
            recent_sequence = asset_returns.tail(self.config.sequence_length).values
            scaled_sequence = self.scaler.transform(recent_sequence.reshape(-1, 1)).flatten()
            X_pred = scaled_sequence.reshape(1, self.config.sequence_length, 1)
            
            pred_scaled = self.model.predict(X_pred, verbose=0)
            pred_returns = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            predictions[asset] = pred_returns
        
        return predictions