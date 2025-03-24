# FreshAI
FreshAI is an AI-powered app designed to combat food waste by intelligently tracking kitchen inventory, predicting expiration dates, and suggesting recipes. Leveraging computer vision for food recognition and machine learning for personalized recommendations, it empowers households and businesses to adopt sustainable consumption habits.
Here’s an enhanced, polished version of your README with expanded details, clearer structure, and actionable improvements:

---

# **FreshAI: Reducing Food Waste Through Smart Inventory Management**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
**GitHub Repo:** [FreshAI Project](https://github.com/starlinks123/FreshAI)  

---

## **Background**  
### **The Problem**  
- **Global Impact**: 1.3 billion tons of food is wasted annually, contributing to **8% of global greenhouse gas emissions** (FAO, 2023).  
- **Household Waste**: 28% of purchased food in households goes uneaten (USDA).  
- **Business Loss**: Supermarkets discard unsold perishables daily, incurring financial and environmental costs.  

### **Motivation**  
- **Sustainability**: Passion for reducing environmental footprints through technology.  
- **Economic Efficiency**: Helping users save money by optimizing food usage.  

### **Why It Matters**  
- Reducing waste preserves resources (water, energy) and lowers methane emissions from landfills.  

---

## **How It Works**  
### **User Groups**  
- **Households**: Manage groceries efficiently.  
- **Restaurants**: Track ingredient shelf life.  
- **Grocery Stores**: Redistribute surplus food.  

### **Workflow**  
1. **Scan & Log**: Users snap photos of groceries → AI identifies items and logs expiration dates using **CNN-based image recognition**.  
   <img src="https://example.com/freshai-scan-demo.gif" width="300" alt="Scanning groceries with FreshAI">  
2. **Smart Alerts**: Notifications for nearing-expiry items + **recipe suggestions** (e.g., "Your spinach expires tomorrow! Try a spinach smoothie.").  
3. **Inventory Dashboard**: Track usage patterns and generate optimized shopping lists.  
4. **Community Sharing**: Option to donate surplus food to local shelters via partner networks.  

### **Technical Demo**  
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import joblib
from datetime import datetime

# Enhanced data loading with temporal features
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['purchase_date', 'expiry_date'])
    
    # Feature engineering
    df['days_until_expiry'] = (df['expiry_date'] - df['purchase_date']).dt.days
    df['purchase_day_of_week'] = df['purchase_date'].dt.dayofweek
    df['purchase_month'] = df['purchase_date'].dt.month
    df['storage_temp_variation'] = df.groupby('item_type')['storage_temperature'].transform('std')
    
    # Historical spoilage rates
    spoilage_stats = df.groupby('item_type')['spoiled'].agg(['mean', 'std']).reset_index()
    spoilage_stats.columns = ['item_type', 'avg_spoilage_rate', 'spoilage_std']
    df = pd.merge(df, spoilage_stats, on='item_type', how='left')
    
    return df

# Load enhanced dataset
data = load_data('food_inventory.csv')

# Handle anomalies in storage conditions
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(data[['storage_temperature', 'humidity']])
data = data[anomalies == 1]

# Define features and target
features = data[[
    'storage_temperature', 
    'humidity',
    'item_type',
    'package_type',
    'purchase_day_of_week',
    'purchase_month',
    'storage_temp_variation',
    'avg_spoilage_rate'
]]

target = data['days_until_expiry']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['storage_temperature', 'humidity', 'storage_temp_variation', 'avg_spoilage_rate']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['item_type', 'package_type'])
    ])

# Advanced model pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50
    ))
])

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter grid
param_grid = {
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_child_weight': [1, 5, 10],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

# Randomized search with time-series validation
search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=2
)

# Train-test split with temporal ordering
train_size = int(0.8 * len(data))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Fit model
search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

# Evaluate
predictions = best_model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
print(f"MAE: {mean_absolute_error(y_test, predictions))}")

# Explainability with SHAP
explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
shap_values = explainer.shap_values(preprocessor.transform(X_test))
shap.summary_plot(shap_values, X_test)

# Save model pipeline
joblib.dump(best_model, 'expiry_predictor.pkl')

# Production prediction example
def predict_expiry(item_type, storage_temp, humidity, package_type):
    input_data = pd.DataFrame([{
        'item_type': item_type,
        'storage_temperature': storage_temp,
        'humidity': humidity,
        'package_type': package_type,
        'purchase_day_of_week': datetime.now().weekday(),
        'purchase_month': datetime.now().month,
        'storage_temp_variation': data[data['item_type'] == item_type]['storage_temperature'].std(),
        'avg_spoilage_rate': data[data['item_type'] == item_type]['spoilage_rate'].mean()
    }])
    
    return best_model.predict(input_data)[0]

# Example prediction with confidence interval
prediction = predict_expiry('dairy', 4, 65, 'vacuum_sealed')
confidence = 0.95  # 95% confidence interval
std_dev = np.std(y_test - predictions)
print(f"Predicted expiry: {prediction:.1f} days (±{1.96*std_dev:.1f} days at {confidence*100}% confidence)")

# API endpoint example (using Flask)
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    prediction = predict_expiry(
        data['item_type'],
        data['storage_temp'],
        data['humidity'],
        data['package_type']
    )
    return jsonify({'predicted_expiry_days': prediction})

if __name__ == '__main__':
    app.run(debug=True)
```
1. Install required packages:

```python
pip install pandas scikit-learn xgboost shap flask joblib
 ```

2. Prepare your dataset with additional columns:

humidity

package_type

spoiled (boolean flag)

spoilage_rate

3. Run the script and access the API:

```python
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"item_type": "dairy", "storage_temp": 4, "humidity": 65, "package_type": "vacuum_sealed"}'
```

---

## **Data Sources & AI Techniques**  
| **Component**               | **Data/Technology**                          | **Purpose**                          |  
|------------------------------|----------------------------------------------|--------------------------------------|  
| **Food Recognition**          | [Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41) (101k images, CC BY 4.0) | Train CNN model (ResNet-50) to classify food items |  
| **Recipe Suggestions**        | [Spoonacular API](https://spoonacular.com/food-api) + [Open Food Facts](https://world.openfoodfacts.org/) | NLP-based ingredient matching and collaborative filtering |  
| **Expiry Prediction**         | User-input storage conditions + [USDA Shelf Life Data](https://fdc.nal.usda.gov/) | Time-series forecasting (ARIMA/LSTM) |  
| **User Interface**            | React Native (mobile app) + TensorFlow Lite (on-device inference) | Cross-platform deployment |  

---

## **Challenges & Ethical Considerations**  
### **Technical Limitations**  
- **Image Recognition**: Struggles with obscured items (e.g., wrapped produce) or uncommon regional foods.  
- **Spoilage Variability**: Storage conditions (humidity, temperature) may require IoT sensors for accuracy.  

### **Ethical Concerns**  
- **Privacy**: Secure storage of user food logs; anonymize data for ML training.  
- **Bias**: Recipe suggestions must accommodate dietary restrictions (vegan, allergies) and cultural preferences.  

---

## **Future Roadmap**  
1. **Phase 1 (MVP)**: Launch mobile app with core features (scan, alerts, recipes).  
2. **Phase 2**: Partner with grocery chains to integrate surplus food redistribution.  
3. **Phase 3**: Deploy IoT sensors (smart fridges) for real-time tracking.  

### **Skills Needed for Growth**  
- IoT integration (Arduino/Raspberry Pi)  
- Cloud scaling (AWS/Azure)  
- UX/UI design for accessibility  

---

## **Acknowledgments**  
- [Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41) by ETH Zurich (CC BY 4.0).  
- [Olio](https://olioex.com/) and [Too Good To Go](https://toogoodtogo.com/) for inspiring food waste reduction models.  
- Mentorship from the [Building AI](https://buildingai.elementsofai.com/) course by Reaktor & University of Helsinki.  
