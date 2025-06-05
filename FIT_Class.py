import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# Load and preprocess your data
data = pd.read_csv('C:\\Users\\Пользователь\\Desktop\\РГР_ML\\data\\balanced_diabetes_data.csv')
data = data[:5000]
X = data.drop('Diabetes_012', axis=1)
y = data['Diabetes_012']

# Model 1: kNN (классическая модель обучения с учителем)
ml1 = KNeighborsClassifier()
ml1.fit(X, y)
with open('models/ml1_model.pkl', 'wb') as f:
    pickle.dump(ml1, f)

# Model 2: XGBoost (ансамблевая модель - бустинг)
ml2 = xgb.XGBClassifier()
ml2.fit(X, y)
ml2.save_model('models/ml2_model.json')

# Model 3: CatBoost (продвинутый градиентный бустинг)
ml3 = CatBoostClassifier(verbose=0)
ml3.fit(X, y)
ml3.save_model('models/ml3_model.cbm')

# Model 4: Random Forest (ансамблевая модель - бэггинг)

ml4 = RandomForestClassifier(
    n_estimators=50,  # Уменьшите количество деревьев (по умолчанию 100)
    max_depth=10,     # Ограничьте глубину деревьев
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
ml4.fit(X, y)
with open('models/ml4_model.pkl', 'wb') as f:
    pickle.dump(ml4, f)

# Model 5: Stacking (ансамблевая модель - стэкинг)
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(random_state=42))
]
# Для Stacking
ml5 = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    stack_method='predict_proba',
    n_jobs=-1  # Параллелизация
)
ml5.fit(X, y)
with open('models/ml5_model.pkl', 'wb') as f:
    pickle.dump(ml5, f)