
# **Project: Classification Models Comparison**

This project compares two classification models—a **Scikit-learn Logistic Regression model** and a **PyTorch Neural Network**—on a dataset related to a business domain. The goal is to evaluate their performance using various metrics and analyze how they behave under different scenarios, such as class imbalance and varying regularization strengths (`C` values).

---

## **Table of Contents**
1. [Business Context](#business-context)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Scikit-learn Model](#scikit-learn-model)
5. [PyTorch Neural Network](#pytorch-neural-network)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Bonus Tasks](#bonus-tasks)
8. [How to Run the Code](#how-to-run-the-code)

---

## **Business Context**
The **Telco Customer Churn** dataset contains information about customers of a telecom company, including their demographics, services subscribed, and whether they churned (stopped using the service). Predicting churn helps businesses take proactive measures to retain customers, such as offering discounts or personalized services.

---

## **Dataset**
We use the [Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) for this project. It includes features like `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`.

### **Code: Load Dataset**
```python
import pandas as pd

# Load Telco Customer Churn dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)

# Display the first few rows
print(data.head())
```

---

## **Preprocessing**
The dataset requires preprocessing, including encoding categorical variables and scaling numerical features.

### **Code: Preprocessing**
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode categorical variables
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService']
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(data[categorical_cols])

# Combine encoded features with numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = np.hstack([encoded_features, data[numerical_cols].values])
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
```

---

## **Scikit-learn Model**
We train a **Logistic Regression model** using Scikit-learn and evaluate its performance.

### **Code: Train Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize and train Logistic Regression
logreg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test_scaled)

# Evaluation metrics
def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print("Logistic Regression Performance:")
print_metrics(y_test, y_pred_logreg)
```

---

## **PyTorch Neural Network**
We train a simple neural network using PyTorch with one hidden layer and ReLU activation.

### **Code: Train Neural Network**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 16
num_classes = len(set(y_train))
model = SimpleNN(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
all_preds_nn = []
with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16, shuffle=False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds_nn.extend(preds.numpy())

print("\nNeural Network Performance:")
print_metrics(y_test, all_preds_nn)
```

---

## **Evaluation Metrics**
Both models are evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---

## **Bonus Tasks**
### **1. Analyze Class Imbalance**
Simulate class imbalance by undersampling one class and evaluate both models.

#### **Code: Simulate Class Imbalance**
```python
from sklearn.utils import resample

# Separate majority and minority classes
majority = data[data['Churn'] == 0]
minority = data[data['Churn'] == 1]

# Downsample majority class
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)

# Combine minority and downsampled majority
balanced_data = pd.concat([majority_downsampled, minority])

# Proceed with preprocessing and training as before
```

### **2. Train with Different `C` Values**
Train Logistic Regression with different `C` values to observe metric variations.

#### **Code: Train with Different `C` Values**
```python
C_values = [0.01, 0.1, 1, 10]

for C in C_values:
    logreg_c = LogisticRegression(C=C, max_iter=1000, random_state=42)
    logreg_c.fit(X_train_scaled, y_train)
    y_pred_c = logreg_c.predict(X_test_scaled)
    
    print(f"\nResults for C = {C}:")
    print_metrics(y_test, y_pred_c)
```

---

## **How to Run the Code**
1. Clone the repository:
   ```bash
   git clone https://github.com/Tesfalegnp/Neural-Network_in_ML.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn torch seaborn matplotlib
   ```
3. Run the notebook or Python script:
   ```bash
   jupyter notebook project.ipynb
   ```

---
