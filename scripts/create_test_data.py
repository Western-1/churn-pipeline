import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 100

df = pd.DataFrame({
    'customerID': [f'C{i:04d}' for i in range(n)],
    'gender': np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen': np.random.choice([0, 1], n),
    'Partner': np.random.choice(['Yes', 'No'], n),
    'Dependents': np.random.choice(['Yes', 'No'], n),
    'tenure': np.random.randint(0, 72, n),
    'PhoneService': np.random.choice(['Yes', 'No'], n),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
    'PaymentMethod': np.random.choice([
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n),
    'MonthlyCharges': np.random.uniform(18.0, 118.0, n),
    'TotalCharges': np.random.uniform(18.0, 8500.0, n),
    'Churn': np.random.choice(['Yes', 'No'], n)
})

Path('tests/fixtures').mkdir(parents=True, exist_ok=True)
df.to_csv('tests/fixtures/sample_data.csv', index=False)
print(f'✅ Created sample data: {len(df)} rows')
