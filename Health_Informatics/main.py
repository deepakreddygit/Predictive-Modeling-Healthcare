import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import time
from sklearn.metrics import confusion_matrix

def calculate_metrics_multiclass(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)  # Total correct predictions / Total predictions

    # Sensitivity and specificity for each class
    sensitivity = cm.diagonal() / cm.sum(axis=1)
    
    # Initialize specificity as an array
    specificity = np.zeros(cm.shape[0])
    
    # Check if the confusion matrix is 2D and has more than 1 class
    if cm.shape[0] > 1:
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum()  # True Negatives for class i
            fp = cm[:, i].sum() - cm[i, i]  # False Positives for class i
            specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:  
        tn = cm[0, 0]  # True Negatives
        fp = cm[0, 1]  # False Positives
        specificity[0] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Precision
    precision = cm.diagonal() / cm.sum(axis=0)

    return accuracy, sensitivity, specificity, precision, cm




# Function to load heart disease data with error handling for delimiters
def load_heart_disease_data(file_path):
    try:
        return pd.read_csv(file_path, header=None, sep=',', encoding='latin1')
    except pd.errors.ParserError:
        return pd.read_csv(file_path, header=None, sep=' ', encoding='latin1')

# Load Breast Cancer dataset
breast_cancer_data = pd.read_csv('dataset/breast_cancer/wdbc.data', header=None)
breast_cancer_data.columns = ['ID', 'Diagnosis', 'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 
                              'Area_Mean', 'Smoothness_Mean', 'Compactness_Mean', 'Concavity_Mean', 
                              'Concave_Points_Mean', 'Symmetry_Mean', 'Fractal_Dimension_Mean', 
                              'Radius_Se', 'Texture_Se', 'Perimeter_Se', 'Area_Se', 
                              'Smoothness_Se', 'Compactness_Se', 'Concavity_Se', 
                              'Concave_Points_Se', 'Symmetry_Se', 'Fractal_Dimension_Se', 
                              'Radius_Worst', 'Texture_Worst', 'Perimeter_Worst', 
                              'Area_Worst', 'Smoothness_Worst', 'Compactness_Worst', 
                              'Concavity_Worst', 'Concave_Points_Worst', 'Symmetry_Worst', 
                              'Fractal_Dimension_Worst']

# Load Cleveland Heart Disease dataset
cleveland_data = load_heart_disease_data('dataset/heart_disease/processed.cleveland.data')
if cleveland_data.shape[1] == 14:
    cleveland_data.columns = ['Age', 'Sex', 'Chest_Pain_Type', 'Resting_BP', 'Serum_Cholestoral',
                              'Fasting_Blood_Sugar', 'Resting_ECG', 'Max_Heart_Rate_Achieved',
                              'Exercise_Induced_Angina', 'ST_Depression', 'Slope_ST_Segment',
                              'Num_Major_Vessels', 'Thal', 'Diagnosis']

# Handle missing data
cleveland_data.replace('?', np.nan, inplace=True)
cleveland_data = cleveland_data.astype(float)  
cleveland_data.fillna(cleveland_data.mean(), inplace=True)

# Scaling features for Breast Cancer dataset
scaler = StandardScaler()
X_breast_cancer = breast_cancer_data.drop(columns=['ID', 'Diagnosis'])
y_breast_cancer = breast_cancer_data['Diagnosis'].map({'M': 1, 'B': 0}) 
X_breast_cancer_scaled = scaler.fit_transform(X_breast_cancer)

# Train-test split for Breast Cancer dataset
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_breast_cancer_scaled, y_breast_cancer, test_size=0.3, random_state=42)

# Classification Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Support Vector Machine': SVC(class_weight='balanced'),
    'k-Nearest Neighbors': KNeighborsClassifier()
}

#for Breast Cancer
performance_metrics_bc = []

# Evaluate each model on Breast Cancer dataset
print("\nEvaluating models on Breast Cancer dataset...")
performance_metrics_bc = []  

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_bc, y_train_bc)
    y_pred_bc = model.predict(X_test_bc)

    # Calculate metrics
    accuracy_bc, sensitivity_bc, specificity_bc, precision_bc, cm_bc = calculate_metrics_multiclass(y_test_bc, y_pred_bc)

    # Store performance metrics for Breast Cancer
    performance_metrics_bc.append({
        'Model': model_name,
        'BC Accuracy': accuracy_bc,
        'BC Precision': precision_bc.mean(), 
        'BC Recall': sensitivity_bc.mean(),  
        'BC Specificity': specificity_bc.mean(), 
    })

# Create DataFrame for performance metrics for Breast Cancer
metrics_df_bc = pd.DataFrame(performance_metrics_bc)
print("\nBreast Cancer Dataset Metrics:")
print(metrics_df_bc)


# Confusion Matrix for Breast Cancer dataset
cm_bc = confusion_matrix(y_test_bc, y_pred_bc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bc, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix for Breast Cancer')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Scaling features for Cleveland Heart Disease dataset
X_cleveland = cleveland_data.drop(columns=['Diagnosis'])
y_cleveland = cleveland_data['Diagnosis']

# Check class distribution before SMOTE
# print("\nOriginal class distribution in Cleveland dataset:")
# print(y_cleveland.value_counts())

# Apply SMOTE for class imbalance in Cleveland dataset
smote = SMOTE(random_state=42)
X_cleveland_resampled, y_cleveland_resampled = smote.fit_resample(X_cleveland, y_cleveland)


# print("Resampled class distribution in Cleveland dataset:")
# print(pd.Series(y_cleveland_resampled).value_counts())

X_cleveland_scaled = scaler.fit_transform(X_cleveland_resampled)

# Train-test split for Cleveland Heart Disease dataset
X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(X_cleveland_scaled, y_cleveland_resampled, test_size=0.3, random_state=42)

# for Heart Disease
performance_metrics_hd = []  

# Evaluate each model on Cleveland Heart Disease dataset
print("\nEvaluating models on Cleveland Heart Disease dataset...")
performance_metrics_hd = [] 

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_hd, y_train_hd)
    y_pred_hd = model.predict(X_test_hd)

    # Calculate metrics
    accuracy_hd, sensitivity_hd, specificity_hd, precision_hd, cm_hd = calculate_metrics_multiclass(y_test_hd, y_pred_hd)

    # Store performance metrics for Cleveland Heart Disease
    performance_metrics_hd.append({
        'Model': model_name,
        'HD Accuracy': accuracy_hd,
        'HD Precision': precision_hd.mean(),  
        'HD Recall': sensitivity_hd.mean(),  
        'HD Specificity': specificity_hd.mean(),  
    })

# Create DataFrame for performance metrics for Heart Disease
metrics_df_hd = pd.DataFrame(performance_metrics_hd)
print("\nCleveland Heart Disease Dataset Metrics:")
print(metrics_df_hd)

# Data Visualization for Breast Cancer Dataset
plt.figure(figsize=(10, 6))
sns.histplot(breast_cancer_data['Radius_Mean'], bins=30, kde=True)
plt.title('Radius Mean Distribution')
plt.xlabel('Radius Mean')
plt.ylabel('Frequency')
plt.show()

# Updated Data Visualization for Breast Cancer Correlation
plt.figure(figsize=(16, 14)) 
correlation_matrix_bc = breast_cancer_data.corr()

#mask to display only the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix_bc, dtype=bool))

# Set the color palette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# heatmap
sns.heatmap(correlation_matrix_bc, mask=mask, annot=True, fmt='.2f', cmap=cmap, cbar=True,
            square=False,
            linewidths=.5, linecolor='white', annot_kws={"size": 6}, 
            xticklabels=breast_cancer_data.columns[2:], yticklabels=breast_cancer_data.columns[2:],
            cbar_kws={"shrink": .8}) 

plt.title('Breast Cancer Feature Correlation', fontsize=24) 
plt.xticks(rotation=45, ha='right', fontsize=10) 
plt.yticks(fontsize=10)
plt.tight_layout(pad=2.0)
plt.show()


# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_breast_cancer_pca = pca.fit_transform(X_breast_cancer_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_breast_cancer_pca[:, 0], X_breast_cancer_pca[:, 1], c=y_breast_cancer, cmap='coolwarm')
plt.title('PCA of Breast Cancer Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Diagnosis (1 = M, 0 = B)')
plt.show()

# Data Visualization for Cleveland Heart Disease Dataset
plt.figure(figsize=(10, 6))
sns.histplot(cleveland_data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix_hd = cleveland_data.corr()
sns.heatmap(correlation_matrix_hd, annot=True, fmt='.1f', cmap='coolwarm', cbar=True, square=True, linewidths=.5, 
            annot_kws={"size": 12}, linecolor='white')
plt.title('Cleveland Heart Disease Feature Correlation', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(pad=2.0)
plt.show()

# Confusion Matrix for Cleveland Heart Disease dataset
cm_hd = confusion_matrix(y_test_hd, y_pred_hd)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_hd, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=np.unique(y_cleveland_resampled), yticklabels=np.unique(y_cleveland_resampled))
plt.title('Confusion Matrix for Cleveland Heart Disease')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
