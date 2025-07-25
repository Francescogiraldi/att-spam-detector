import os
import subprocess
import sys

def setup_models():
    """Setup models for Streamlit Cloud deployment"""
    models_dir = 'models'
    model_file = os.path.join(models_dir, 'sklearn_model.joblib')
    vectorizer_file = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    
    # Check if models already exist
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        print("Models already exist, skipping training.")
        return True
    
    print("Models not found, training new models...")
    try:
        # Run the training script
        result = subprocess.run([sys.executable, 'train_sklearn_only.py'], 
                              capture_output=True, text=True, check=True)
        print("Training completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    setup_models()