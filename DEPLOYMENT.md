# 🚀 Deployment Guide for AT&T Spam Detector on Streamlit Cloud

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Required Files**: Ensure all necessary files are committed to your repository

## 📁 Required Files for Deployment

Make sure your repository contains:

```
├── app_simple.py          # Main Streamlit application
├── requirements.txt       # Python dependencies
├── spam.csv              # Training dataset
├── train_sklearn_only.py  # Model training script
├── setup_models.py       # Auto-setup script for models
└── models/               # Pre-trained models (optional)
    ├── sklearn_model.joblib
    └── tfidf_vectorizer.joblib
```

## 🔧 Deployment Steps

### Step 1: Prepare Your Repository

1. **Commit all changes** to your GitHub repository:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Verify .gitignore**: Ensure `*.joblib` and `models/` are NOT in `.gitignore` if you want to include pre-trained models

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account if not already connected
4. Select your repository
5. Choose the branch (usually `main`)
6. Set the main file path: `app_simple.py`
7. Click **"Deploy!"**

### Step 3: Monitor Deployment

- The deployment process will:
  1. Install dependencies from `requirements.txt`
  2. Run your Streamlit app
  3. Auto-train models if they don't exist (thanks to `setup_models.py`)

## ⚙️ Configuration Details

### Requirements.txt
The app uses these key dependencies:
```
streamlit>=1.28.0,<1.41.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### Auto-Training Feature
The app includes automatic model training:
- If models are missing, the app will automatically train them
- Training happens on first load (may take 1-2 minutes)
- Models are saved for subsequent runs

## 🔍 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Check `requirements.txt` includes all dependencies
   - Verify package versions are compatible

2. **"File not found" errors**:
   - Ensure `spam.csv` is in the repository
   - Check file paths are relative to the app root

3. **Memory issues**:
   - Streamlit Cloud has memory limits
   - Consider reducing model complexity if needed

4. **Long loading times**:
   - First run may be slow due to model training
   - Subsequent runs should be faster

### Debug Information
The app includes debug output that shows:
- Current working directory
- Model file paths and existence
- Loading status and errors

## 📊 App Features

- **Real-time SMS spam detection**
- **Interactive text input**
- **Confidence scores**
- **Model performance metrics**
- **Automatic model training**
- **French language interface**

## 🔗 Access Your App

Once deployed, your app will be available at:
`https://[your-app-name].streamlit.app/`

## 📝 Notes

- Streamlit Cloud apps go to sleep after inactivity
- First access after sleep may take longer to load
- Free tier has usage limits
- Apps are public by default

## 🆘 Support

If you encounter issues:
1. Check Streamlit Cloud logs in the deployment interface
2. Review the debug information in the app
3. Verify all files are properly committed to GitHub
4. Check Streamlit Cloud documentation