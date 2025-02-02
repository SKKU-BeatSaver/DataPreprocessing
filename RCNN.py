import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, 
                                   MaxPooling1D, Dropout, Dense, GRU, 
                                   Bidirectional, GlobalMaxPooling1D)
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping
from pathlib import Path
import re
import traceback

class ECGDataManager:
    def __init__(self, data_dir, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError("Split proportions must sum to 1")
            
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
    
    def get_subject_number(self, filename):
        """Extract subject number from filename"""
        match = re.match(r'S(\d{4})_labeled\.txt', filename.name)
        if match:
            return int(match.group(1))
        return None
    
    def load_and_preprocess_file(self, filename):
        """Load and preprocess a single file"""
        try:
            # Load data
            data = np.loadtxt(filename)
            
            # Separate features and label
            features = data[:-1]  # All but last value
            label = float(data[-1] >= 0.5)  # Convert to binary label
            
            # Normalize features
            mean = np.mean(features)
            std = np.std(features)
            features_normalized = (features - mean) / (std + 1e-10)
            
            # Reshape for model input
            features_reshaped = features_normalized.reshape(-1, 1)
            
            return features_reshaped, label
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            raise
    
    def split_files(self):
        """Split files into training, validation, and test sets"""
        # Get all labeled files and sort by subject number
        labeled_files = []
        pattern = re.compile(r'S\d{4}_labeled\.txt')
        
        for file in self.data_dir.glob("S*_labeled.txt"):
            if pattern.match(file.name):
                labeled_files.append((self.get_subject_number(file), file))
        
        if not labeled_files:
            raise ValueError(f"No labeled files found in {self.data_dir}")
        
        # Sort files by subject number
        labeled_files.sort(key=lambda x: x[0])
        files = [f[1] for f in labeled_files]
        
        # Calculate split indices
        total_files = len(files)
        train_size = int(total_files * self.train_split)
        val_size = int(total_files * self.val_split)
        
        # Split files
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]
        
        return train_files, val_files, test_files
    
    def create_dataset(self, files, fixed_input_length):
        """Create dataset from files with padding to handle variable lengths"""
        X_data = []
        y_data = []
        file_ids = []
        
        print("\nProcessing data...")
        for file in files:
            try:
                features, label = self.load_and_preprocess_file(file)
                X_data.append(features)
                y_data.append(label)
                file_ids.append(self.get_subject_number(file))
            except Exception as e:
                print(f"Skipping {file}: {str(e)}")
                continue
        
        if not X_data:
            raise ValueError("No valid data could be processed from files")
        
        # Pad sequences to fixed length
        X_padded = []
        for x in X_data:
            padded = np.zeros((fixed_input_length, 1))
            length = min(len(x), fixed_input_length)
            padded[:length] = x[:length]  # Fill with the original data or truncate
            X_padded.append(padded)
        
        try:
            # Convert to numpy arrays
            X_final = np.array(X_padded)
            y_final = np.array(y_data)
            
            print(f"\nFinal shapes:")
            print(f"X shape: {X_final.shape}")
            print(f"y shape: {y_final.shape}")
            
            return X_final, y_final, file_ids
            
        except Exception as e:
            print("\nError creating final arrays:")
            print(f"Attempted shapes: {[x.shape for x in X_data]}")
            raise ValueError(f"Error creating arrays: {str(e)}")

    def prepare_data(self):
        """Prepare training, validation, and test datasets"""
        # Split files
        train_files, val_files, test_files = self.split_files()
        
        fixed_input_length = 115200  # Set a fixed input length
        
        print(f"\nTotal files found: {len(train_files) + len(val_files) + len(test_files)}")
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")
        
        # Create datasets
        print("\nProcessing training data...")
        X_train, y_train, train_ids = self.create_dataset(train_files, fixed_input_length)
        
        print("\nProcessing validation data...")
        X_val, y_val, val_ids = self.create_dataset(val_files, fixed_input_length)
        
        print("\nProcessing test data...")
        X_test, y_test, test_ids = self.create_dataset(test_files, fixed_input_length)
        
        # Print summary
        print("\nData Split Summary:")
        print("=" * 50)
        print(f"Training set:")
        print(f"  Samples: {len(X_train)}")
        print(f"  Input shape: {X_train.shape}")
        print(f"  Positive class: {sum(y_train)} ({(sum(y_train)/len(y_train))*100:.2f}%)")
        
        print(f"\nValidation set:")
        print(f"  Samples: {len(X_val)}")
        print(f"  Input shape: {X_val.shape}")
        print(f"  Positive class: {sum(y_val)} ({(sum(y_val)/len(y_val))*100:.2f}%)")
        
        print(f"\nTest set:")
        print(f"  Samples: {len(X_test)}")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Positive class: {sum(y_test)} ({(sum(y_test)/len(y_test))*100:.2f}%)")
        print("=" * 50)
        
        return {
            'train': {
                'X': X_train,
                'y': y_train,
                'ids': train_ids
            },
            'val': {
                'X': X_val,
                'y': y_val,
                'ids': val_ids
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'ids': test_ids
            }
        }

class ECGClassifier:
    def __init__(self, input_length):
        self.input_length = input_length
        self.model = self.create_rcnn_model()
        
    def create_rcnn_model(self):
        # Input layer
        input_layer = Input(shape=(self.input_length, 1))
        
        # CNN Block 1
        conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2)(bn1)
        
        # CNN Block 2
        conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(pool1)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(pool_size=2)(bn2)
        
        # CNN Block 3
        conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(pool2)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling1D(pool_size=2)(bn3)
        
        # Bidirectional GRU layers
        gru1 = Bidirectional(GRU(64, return_sequences=True))(pool3)
        dropout1 = Dropout(0.3)(gru1)
        
        gru2 = Bidirectional(GRU(32, return_sequences=True))(dropout1)
        dropout2 = Dropout(0.3)(gru2)
        
        # Global pooling
        max_pool = GlobalMaxPooling1D()(dropout2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(max_pool)
        bn4 = BatchNormalization()(dense1)
        dropout3 = Dropout(0.2)(bn4)
        
        # Output layer
        output_layer = Dense(1, activation='sigmoid')(dropout3)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

def main(data_directory):
    # Initialize data manager
    data_manager = ECGDataManager(data_directory)
    
    # Prepare data
    datasets = data_manager.prepare_data()
    
    # Instantiate the model
    model = ECGClassifier(input_length=115200)
    
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the model with EarlyStopping
    print("\nTraining the model...")
    model.model.fit(
        datasets['train']['X'], datasets['train']['y'],
        validation_data=(datasets['val']['X'], datasets['val']['y']),
        epochs=100,  # You may want to increase the epochs to allow early stopping to take effect
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]  # Add the callback here
    )
    
    # Evaluate the model on the test set
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.model.evaluate(datasets['test']['X'], datasets['test']['y'], verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    data_dir = "/"  # Update this path to your data directory
    main(data_dir)