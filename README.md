# Iris Flower Classification Project

## Overview
This project implements a machine learning model to classify iris flowers into three species (setosa, versicolor, and virginica) based on their physical measurements. The classification is performed using a Random Forest algorithm achieving **90% accuracy**.

## Dataset
- **Source**: Iris flower dataset with measurements of 150 flowers
- **Features**: 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)  
  - Petal Length (cm)
  - Petal Width (cm)
- **Target**: 3 species classes
  - Iris-setosa
  - Iris-versicolor  
  - Iris-virginica
- **Distribution**: Perfectly balanced with 50 samples per species

## Project Structure
```
iris-classification/
│
├── Iris.csv                 # Dataset file
├── iris_classification.ipynb # Main Jupyter notebook
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Key Results

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Overall Accuracy**: 90.0%
- **Training/Test Split**: 80/20 (120 training, 30 testing samples)

### Per-Class Performance
| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Iris-setosa | 1.00 | 1.00 | 1.00 | 10 |
| Iris-versicolor | 0.82 | 0.90 | 0.86 | 10 |
| Iris-virginica | 0.89 | 0.80 | 0.84 | 10 |

### Feature Importance
1. **Petal Width**: 43.72% - Most discriminative feature
2. **Petal Length**: 43.15% - Second most important
3. **Sepal Length**: 11.63% - Moderate importance
4. **Sepal Width**: 1.50% - Least important

## Key Insights
- **Perfect Separation**: Iris-setosa is perfectly distinguishable (100% accuracy)
- **Petal Features Dominate**: Petal measurements are far more important than sepal measurements
- **Species Overlap**: Some confusion between versicolor and virginica species
- **Robust Model**: 90% accuracy indicates reliable classification performance

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Quick Start
```python
# Load and prepare data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Iris.csv')
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # Sepal Length, Sepal Width, Petal Length, Petal Width
prediction = model.predict(sample)
print(f"Predicted species: {prediction[0]}")
```

### Sample Prediction
```python
# Example: Typical Iris-setosa measurements
sample_flower = [5.1, 3.5, 1.4, 0.2]
predicted_species = model.predict([sample_flower])
# Output: Iris-setosa
```

## Technical Details

### Data Preprocessing
- No missing values detected
- No feature scaling required (Random Forest is scale-invariant)
- Balanced dataset requires no sampling techniques

### Model Selection
- **Random Forest** chosen for:
  - High accuracy on tabular data
  - Built-in feature importance
  - Robustness to overfitting
  - No hyperparameter tuning required for good performance

### Evaluation Metrics
- **Accuracy**: Overall correctness (90%)
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall

## Confusion Matrix
```
                Predicted
Actual    setosa  versicolor  virginica
setosa      10        0          0
versicolor   0        9          1  
virginica    0        2          8
```

## Future Improvements
1. **Hyperparameter Tuning**: Grid search for optimal Random Forest parameters
2. **Feature Engineering**: Create ratio features (petal/sepal ratios)
3. **Model Ensemble**: Combine multiple algorithms for better accuracy
4. **Cross-Validation**: 5-fold CV for more robust performance estimation
5. **Deployment**: Create REST API for real-time predictions

## Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## License
This project is open source and available under the MIT License.

## Author
Created for machine learning practice and educational purposes.

---
*This project demonstrates fundamental machine learning concepts including data preprocessing, model training, evaluation, and interpretation.*
