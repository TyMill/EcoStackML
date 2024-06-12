import sys
import pandas as pd
from preprocessing.data_preprocessing import preprocess_data
from stacking.stacking import train_stacking_model
from evaluation.evaluation_metrics import print_evaluation_metrics

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_stacking.py <data_file> <target_column>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    target_column = sys.argv[2]
    
    # Preprocess data
    df = preprocess_data(data_file, columns=[col for col in df.columns if col != target_column])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train stacking model
    model, val_predictions, y_val = train_stacking_model(X, y)
    
    # Evaluate model
    print_evaluation_metrics(y_val, val_predictions)

if __name__ == "__main__":
    main()
