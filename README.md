# 🏀 NBA All-Rookie Team Prediction

> **Predicting NBA All-Rookie Team selections using machine learning algorithms**

## Project hosted on GitHub Pages
### https://matthew4335.github.io/NBAPredictorMLProject/

## 📖 Description

This project leverages machine learning to predict NBA All-Rookie Team selections by analyzing player statistics and performance data. The goal is to provide an objective, data-driven approach to identify which rookie players are most likely to be selected for the prestigious All-Rookie teams, addressing the subjective nature of the current selection process.

**Who it's for:**
- Sports analytics enthusiasts
- NBA fans and analysts
- Machine learning practitioners
- Data science students

**Why it matters:**
The current NBA All-Rookie Team selection process relies heavily on human judgment, which can lead to inconsistencies and overlook valuable statistical contributions. Our ML models provide a transparent, data-driven alternative that identifies patterns correlating with successful rookie selections.

## ✨ Features

- **Multi-Model Approach**: Implements four different ML algorithms for comprehensive analysis
- **Advanced Data Preprocessing**: PCA dimensionality reduction and feature scaling
- **High Accuracy Predictions**: Achieves up to 97% accuracy in identifying All-Rookie selections
- **Interactive Visualizations**: Confusion matrices and performance metrics
- **Real-World Testing**: Validated against actual 2022 NBA All-Rookie selections
- **Web-Based Documentation**: Complete project documentation with Jekyll

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development and analysis
- **scikit-learn** - Machine learning algorithms and preprocessing
- **TensorFlow/Keras** - Neural network implementation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

### Web Technologies
- **Jekyll** - Static site generator for documentation
- **GitHub Pages** - Hosting and deployment
- **Cayman Theme** - Clean, responsive design

### Machine Learning Models
- **Logistic Regression** - Binary classification with class weighting
- **K-Nearest Neighbors (KNN)** - Distance-based classification
- **Neural Network** - Deep learning with TensorFlow/Keras
- **Gaussian Mixture Model (GMM)** - Unsupervised clustering


### Model Performance Examples

The models achieved impressive results on the 2022 NBA season:

**Logistic Regression Results:**
- Accuracy: 97%
- Precision (All-Rookie): 87%
- Recall (All-Rookie): 93%
- F1-Score (All-Rookie): 90%

**K-Nearest Neighbors Results:**
- Accuracy: 95.8%
- Precision (All-Rookie): 100%
- Recall (All-Rookie): 74%
- F1-Score (All-Rookie): 85%

## ⚙️ Configuration

### Data Sources
The project uses the [Kaggle NBA dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data) containing:
- Player statistics from 1947-2024
- 31,000+ player records
- Multiple CSV files with different stat categories

### Key Parameters
- **Training Period**: 2000-2021 seasons
- **Testing Period**: 2022 season
- **PCA Components**: 4 (retaining 95% variance)
- **Class Weights**: 1:3 (non-All-Rookie : All-Rookie)
- **KNN Optimal k**: 19 (based on F1-score optimization)

### Environment Variables
No environment variables are required for basic usage. The project uses publicly available datasets.

## 📁 Folder Structure

```
MLProject/
├── ProjectCode/                 # Main analysis code
│   ├── projectcode-final.ipynb  # Complete ML analysis
│   ├── data/                    # NBA dataset files
│   │   ├── PlayerPerGame.csv
│   │   ├── PlayerTotals.csv
│   │   └── ... (20+ CSV files)
│   └── *.jpg                    # Generated visualizations
├── _layouts/                    # Jekyll templates
├── _sass/                       # CSS styling
├── assets/                      # Static assets
├── docs/                        # Documentation
├── index.md                     # Main project page
├── ProjectProposal.md           # Initial project proposal
├── FinalReport.md               # Project summary
└── README.md                    # This file
```

### Key Files Explained
- **`projectcode-final.ipynb`**: Complete machine learning pipeline
- **`data/`**: Contains all NBA statistics CSV files
- **`index.md`**: Main project documentation and results
- **`ProjectProposal.md`**: Original project proposal and methodology

