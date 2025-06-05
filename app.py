import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from PIL import Image
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os

# Конфигурация
st.set_page_config(layout="wide", page_title="Анализ диабета")

# Навигация
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти", 
    ["Информация", "Описание данных", "Визуализации", "Предсказание диабета"])

# Загрузка моделей (добавляем MLP модель)
@st.cache_resource
def load_models():
    models = {}
    with open('models/ml1_model.pkl', 'rb') as f:
        models['kNN'] = pickle.load(f)
    models['XGBoost'] = xgb.XGBClassifier()
    models['XGBoost'].load_model('models/ml2_model.json')
    models['CatBoost'] = CatBoostClassifier()
    models['CatBoost'].load_model('models/ml3_model.cbm')
    with open('models/ml4_model.pkl', 'rb') as f:
        models['Random Forest'] = pickle.load(f)
    with open('models/ml5_model.pkl', 'rb') as f:
        models['Stacking'] = pickle.load(f)
    
    # Загрузка MLP модели и scaler
    try:
        with open('mlp_adam_model.pkl', 'rb') as f:
            models['MLP (Adam)'] = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("MLP модель или scaler не найдены. Убедитесь, что файлы mlp_adam_model.pkl и scaler.pkl находятся в правильной директории.")
        models['MLP (Adam)'] = None
        scaler = None
    
    return models, scaler

def predict_with_mlp(model, scaler, input_data):
    """Функция для предсказания с помощью MLP модели"""
    if scaler is None:
        st.error("Scaler не загружен. Невозможно выполнить предсказание.")
        return None, None
    
    # Масштабирование данных
    input_data_scaled = scaler.transform(input_data)
    
    # Получение предсказания (MLPRegressor возвращает непрерывное значение)
    prediction_continuous = model.predict(input_data_scaled)
    
    # Преобразование в дискретные классы (0, 1, 2)
    predictions = np.round(prediction_continuous).astype(int)
    # Ограничим значения диапазоном 0-2
    predictions = np.clip(predictions, 0, 2)
    
    # Для MLP нет predict_proba, создадим фиктивные вероятности
    proba = np.zeros((len(predictions), 3))
    for i, pred in enumerate(predictions):
        proba[i, pred] = 1.0
    
    return predictions, proba

# Страница 1: Информация
if page == "Информация":
    st.title("Прогнозирование диабета")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        image = Image.open('assets/Фото 30х40.jpg')
        st.image(image, width=200)
    
    with col2:
        st.write("**ФИО:** Истомина Яна Юрьевна")
        st.write("**Группа:** ФИТ-231")
        st.write("**Тема:** Прогнозирование диабета")
        st.write("""
        Этот дашборд позволяет анализировать данные о пациентах
        и прогнозировать вероятность диабета с помощью различных ML моделей.
        """)

# Страница 2: Описание данных
elif page == "Описание данных":
    st.title("Описание набора данных")
    
    st.header("Данные о пациентах")
    st.write("""
    Набор данных содержит информацию о пациентах и их здоровье.
    Включает следующие характеристики:
    - HighBP: Высокое давление
    - HighChol: Высокий холестерин
    - BMI: Индекс массы тела
    - Smoker: Курильщик
    - Stroke: Инсульт в анамнезе
    - HeartDiseaseorAttack: Болезни сердца
    - PhysActivity: Физическая активность
    - Fruits: Употребление фруктов
    - Veggies: Употребление овощей
    - HvyAlcoholConsump: Употребление алкоголя
    - GenHlth: Общее состояние здоровья
    - MentHlth: Психическое здоровье
    - PhysHlth: Физическое здоровье
    - DiffWalk: Трудности при ходьбе
    - Sex: Пол
    - Age: Возраст
    - Education: Образование
    - Income: Доход
    - Diabetes_012: Наличие диабета (целевая переменная)
    """)
    
    st.header("Пример данных")
    data_path = os.path.join(os.path.dirname(__file__), 'diabetes_good.csv')
    data = pd.read_csv(data_path)
    st.dataframe(data.head())
    
    st.header("Предобработка")
    st.write("""
    Выполнены следующие шаги:
    - Нормализация числовых признаков
    - Проверка на выбросы
    """)

# Страница 3: Визуализации
elif page == "Визуализации":
    st.title("Визуализация данных")
    data = pd.read_csv(data_path)
    
    # 1. Распределение целевой переменной
    st.header("1. Распределение диабета")
    fig1, ax1 = plt.subplots()
    data['Diabetes_012'].value_counts().plot(kind='bar')
    plt.xlabel('Класс диабета')
    plt.ylabel('Количество')
    plt.xticks(ticks=[0, 1, 2], labels=['Нет диабета', 'Предиабет', 'Диабет'], rotation=0)
    st.pyplot(fig1)
    
    # 2. Зависимость от состояния здоровья
    st.header("2. Диабет vs Общее состояние здоровья")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Diabetes_012', y='GenHlth', data=data)
    plt.xticks(ticks=[0, 1, 2], labels=['Нет диабета', 'Предиабет', 'Диабет'])
    st.pyplot(fig2)
    
    # 3. Коробчатые диаграммы по категориям
    st.header("3. Распределение признаков по классам диабета")
    category = st.selectbox("Выберите признак", 
                          ['BMI', 'MentHlth', 'PhysHlth', 'Income'])
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Diabetes_012', y=category, data=data)
    plt.xticks(ticks=[0, 1, 2], labels=['Нет диабета', 'Предиабет', 'Диабет'])
    st.pyplot(fig3)
    
    # 4. Тепловая карта корреляций
    st.header("4. Корреляция признаков")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig4)

# Страница 4: Предсказание
elif page == "Предсказание диабета":
    st.title("Прогнозирование диабета")
    
    # Загрузка моделей и scaler
    models, scaler = load_models()
    
    # Вариант 1: Загрузка файла
    st.header("1. Прогнозирование по файлу")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:")
        st.dataframe(input_df.head())
        
        model_choice = st.selectbox("Выберите модель", list(models.keys()))
        
        if st.button("Спрогнозировать"):
            # Убедимся, что целевая переменная не входит в предикторы
            if 'Diabetes_012' in input_df.columns:
                X = input_df.drop('Diabetes_012', axis=1)
            else:
                X = input_df
            
            # Для MLP модели
            if model_choice == 'MLP (Adam)':
                predictions, proba = predict_with_mlp(models[model_choice], scaler, X)
            else:
                predictions = models[model_choice].predict(X)
                proba = models[model_choice].predict_proba(X)
            
            result_df = input_df.copy()
            result_df['Прогнозируемый класс'] = predictions
            result_df['Вероятность класса 0'] = proba[:, 0].round(3)
            result_df['Вероятность класса 1'] = proba[:, 1].round(3)
            result_df['Вероятность класса 2'] = proba[:, 2].round(3)
            
            st.write("Результаты:")
            st.dataframe(result_df)
            
            # Визуализация матрицы ошибок
            if 'Diabetes_012' in input_df.columns:
                st.header("Матрица ошибок")
                cm = confusion_matrix(input_df['Diabetes_012'], predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', 
                            xticklabels=['Нет', 'Предиабет', 'Диабет'],
                            yticklabels=['Нет', 'Предиабет', 'Диабет'])
                plt.xlabel('Предсказание')
                plt.ylabel('Факт')
                st.pyplot(fig)
            
            # Скачивание результатов
            csv = result_df.to_csv(index=False)
            st.download_button(
                "Скачать прогнозы",
                csv,
                "diabetes_predictions.csv",
                "text/csv"
            )
    
    # Вариант 2: Ручной ввод
    st.header("2. Ручной ввод параметров")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_bp = st.selectbox("Высокое давление", [0, 1])
        high_chol = st.selectbox("Высокий холестерин", [0, 1])
        chol_check = st.selectbox("Проверка холестерина за последние 5 лет", [0, 1])
        bmi = st.slider("Индекс массы тела (BMI)", 10.0, 50.0, 25.0)
        smoker = st.selectbox("Курильщик", [0, 1])
        stroke = st.selectbox("Инсульт в анамнезе", [0, 1])
        heart_disease = st.selectbox("Болезни сердца", [0, 1])
        any_healthcare = st.selectbox("Есть медстраховка", [0, 1])

    with col2:
        phys_activity = st.selectbox("Физическая активность", [0, 1])
        fruits = st.selectbox("Употребление фруктов", [0, 1])
        veggies = st.selectbox("Употребление овощей", [0, 1])
        hvy_alcohol = st.selectbox("Употребление алкоголя", [0, 1])
        no_doc_cost = st.selectbox("Отказ от визита к врачу из-за стоимости", [0, 1])
        gen_hlth = st.slider("Общее состояние здоровья (1-5)", 1, 5, 3)
        ment_hlth = st.slider("Дней с плохим психическим здоровьем (0-30)", 0, 30, 0)
        phys_hlth = st.slider("Дней с плохим физическим здоровьем (0-30)", 0, 30, 0)
        diff_walk = st.selectbox("Трудности при ходьбе", [0, 1])
        sex = st.selectbox("Пол (0-жен, 1-муж)", [0, 1])
        age = st.slider("Возраст", 1, 13, 6)
        education = st.selectbox("Уровень образования (1-6)", [1, 2, 3, 4, 5, 6])
        income = st.selectbox("Доход (1-8)", [1, 2, 3, 4, 5, 6, 7, 8])
    
    model_choice_manual = st.selectbox("Модель для прогноза", list(models.keys()))
    
    if st.button("Рассчитать вероятность диабета"):
        # Подготовка данных
        input_data = pd.DataFrame({
            'HighBP': [high_bp],
            'HighChol': [high_chol],
            'CholCheck': [chol_check],
            'BMI': [bmi],
            'Smoker': [smoker],
            'Stroke': [stroke],
            'HeartDiseaseorAttack': [heart_disease],
            'PhysActivity': [phys_activity],
            'Fruits': [fruits],
            'Veggies': [veggies],
            'HvyAlcoholConsump': [hvy_alcohol],
            'AnyHealthcare': [any_healthcare],
            'NoDocbcCost': [no_doc_cost],
            'GenHlth': [gen_hlth],
            'MentHlth': [ment_hlth],
            'PhysHlth': [phys_hlth],
            'DiffWalk': [diff_walk],
            'Sex': [sex],
            'Age': [age],
            'Education': [education],
            'Income': [income]
        })
        
        # Для MLP модели
        if model_choice_manual == 'MLP (Adam)':
            prediction_continuous = models[model_choice_manual].predict(scaler.transform(input_data))[0]
            prediction = int(np.round(prediction_continuous))
            prediction = max(0, min(2, prediction))
            proba = [0.0, 0.0, 0.0]
            proba[prediction] = 1.0
        else:
            prediction = int(models[model_choice_manual].predict(input_data)[0])
            proba = models[model_choice_manual].predict_proba(input_data)[0]
        
        class_names = ['Нет диабета', 'Предиабет', 'Диабет']
        
        st.success(f"Прогнозируемый класс: {class_names[prediction]}")
        
        st.write("Вероятности классов:")
        for i, prob in enumerate(proba):
            st.write(f"{class_names[i]}: {prob:.3f}")
        
        # Визуализация вероятностей
        fig, ax = plt.subplots()
        sns.barplot(x=class_names, y=proba)
        plt.ylabel("Вероятность")
        plt.title("Распределение вероятностей по классам")
        st.pyplot(fig)

if __name__ == "__main__":
    st.write("")
