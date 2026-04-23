import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Настройка страницы ---
st.set_page_config(page_title="Анализ данных: ПОЛ-МДА", layout="wide")

# --- Таблица критических значений для критерия Диксона (alpha=0.05) ---
# Для выборки от 3 до 30 значений
Q95 = {
    3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568,
    8: 0.526, 9: 0.493, 10: 0.466, 11: 0.444, 12: 0.426,
    13: 0.410, 14: 0.396, 15: 0.384, 16: 0.374, 17: 0.365,
    18: 0.356, 19: 0.349, 20: 0.342, 21: 0.337, 22: 0.331,
    23: 0.326, 24: 0.321, 25: 0.317, 26: 0.312, 27: 0.308,
    28: 0.304, 29: 0.300, 30: 0.296
}

# --- Функции ---

@st.cache_data
def parse_data(file, file_type):
    """
    Интеллектуальный парсинг данных. Ищет строку с названиями групп и формирует чистый DataFrame.
    """
    if file_type == "csv":
        try:
            # Пытаемся автоматически определить разделитель (запятая или точка с запятой)
            df_raw = pd.read_csv(file, header=None, sep=None, engine='python')
        except:
            df_raw = pd.read_csv(file, header=None)
    else:
        df_raw = pd.read_excel(file, header=None)
    
    # Ищем строку, где есть слово 'Контроль' или 'группы'
    header_idx = 0
    for i in range(min(15, len(df_raw))):
        # Максимально безопасная проверка: переводим каждое значение строго в Python-строку
        row_values = [str(val).lower() for val in df_raw.iloc[i].values]
        if any('контроль' in val for val in row_values) or any('группы' in val for val in row_values):
            header_idx = i
            break
            
    # Устанавливаем найденную строку как заголовок
    df_raw.columns = df_raw.iloc[header_idx]
    
    # Обрезаем все, что было до заголовка
    df_clean = df_raw.iloc[header_idx+1:].copy()
    
    # Сразу переводим всё в числа. Весь текст (например "рыба 1", "IК") безопасно станет NaN.
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Удаляем строки и столбцы, состоящие ЦЕЛИКОМ из NaN
    # Это автоматически уберет текстовые колонки с названиями рыб и лишние шапки
    df_clean = df_clean.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Очищаем названия колонок от пустых (NaN) заголовков
    valid_cols = [c for c in df_clean.columns if pd.notna(c) and str(c).strip() != '']
    df_clean = df_clean[valid_cols]
    
    return df_clean

def dixon_q_test(data):
    """
    Поиск выбросов по критерию Диксона.
    Возвращает список выбросов.
    """
    data = [x for x in data if not np.isnan(x)]
    n = len(data)
    if n < 3 or n > 30:
        return [] # Диксон работает лучше всего для малых выборок
    
    sorted_data = sorted(data)
    outliers = []
    
    # Проверяем минимальное значение
    gap_min = sorted_data[1] - sorted_data[0]
    range_min = sorted_data[-1] - sorted_data[0]
    if range_min > 0:
        q_min = gap_min / range_min
        if q_min > Q95.get(n, 1.0):
            outliers.append(sorted_data[0])
            
    # Проверяем максимальное значение
    gap_max = sorted_data[-1] - sorted_data[-2]
    range_max = sorted_data[-1] - sorted_data[0]
    if range_max > 0:
        q_max = gap_max / range_max
        if q_max > Q95.get(n, 1.0):
            outliers.append(sorted_data[-1])
            
    return outliers

def calculate_descriptive_stats(df):
    """Расчет описательной статистики"""
    stats_list = []
    for col in df.columns:
        data = df[col].dropna()
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n) if n > 0 else 0
        median = np.median(data)
        
        stats_list.append({
            "Группа": col,
            "N (кол-во)": n,
            "Среднее": round(mean, 3),
            "SD (ст. откл.)": round(std, 3),
            "SE (ст. ошибка)": round(se, 3),
            "Медиана": round(median, 3)
        })
    return pd.DataFrame(stats_list)

def calculate_ttest(df, control_group):
    """Расчет t-критерия Стьюдента"""
    results = []
    if control_group not in df.columns:
        return pd.DataFrame()
        
    control_data = df[control_group].dropna()
    
    for col in df.columns:
        if col == control_group:
            continue
        exp_data = df[col].dropna()
        if len(control_data) > 1 and len(exp_data) > 1:
            t_stat, p_val = stats.ttest_ind(control_data, exp_data, equal_var=False) # t-тест Уэлча (надежнее при неравных дисперсиях)
            
            # Определение уровня значимости
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            else: sig = "ns"
            
            results.append({
                "Группа": col,
                "Сравнение с": control_group,
                "t-статистика": round(t_stat, 3),
                "p-value": round(p_val, 4),
                "Значимость": sig
            })
    return pd.DataFrame(results)

def create_excel_download(df_clean, df_stats, df_ttest, outliers_info):
    """Создание Excel файла для скачивания"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_clean.to_excel(writer, sheet_name='Очищенные данные', index=False)
        df_stats.to_excel(writer, sheet_name='Описательная статистика', index=False)
        if not df_ttest.empty:
            df_ttest.to_excel(writer, sheet_name='T-тест Стьюдента', index=False)
        
        # Записываем информацию о выбросах
        if outliers_info:
            outliers_df = pd.DataFrame(outliers_info)
            outliers_df.to_excel(writer, sheet_name='Выбросы (Диксон)', index=False)
            
    processed_data = output.getvalue()
    return processed_data

# --- Интерфейс приложения ---

st.title("🧪 Анализ данных: ПОЛ-МДА (нормализованный на белок)")
st.markdown("""
Это приложение автоматически парсит ваши таблицы, находит выбросы по **критерию Диксона**, 
рассчитывает описательную статистику и проводит **t-тест Стьюдента**.
""")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Загрузите файл (CSV или Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    
    try:
        # 1. Парсинг данных
        df_clean = parse_data(uploaded_file, file_type)
        
        # Выбор контрольной группы
        groups = list(df_clean.columns)
        default_control = groups[0] if groups else None
        for g in groups:
            if 'контроль' in str(g).lower():
                default_control = g
                break
                
        st.sidebar.markdown("---")
        control_group = st.sidebar.selectbox("Выберите контрольную группу:", groups, index=groups.index(default_control) if default_control else 0)
        
        # Настройки удаления выбросов
        remove_outliers = st.sidebar.checkbox("Исключить выбросы из анализа", value=True)
        
        # 2. Поиск и обработка выбросов
        outliers_info = []
        df_processed = df_clean.copy()
        
        for col in df_processed.columns:
            outliers = dixon_q_test(df_processed[col].tolist())
            if outliers:
                for out in outliers:
                    outliers_info.append({"Группа": col, "Значение выброса": out})
                    if remove_outliers:
                        # Заменяем выброс на NaN
                        df_processed.loc[df_processed[col] == out, col] = np.nan
        
        # 3. Расчеты
        df_stats = calculate_descriptive_stats(df_processed)
        df_ttest = calculate_ttest(df_processed, control_group)
        
        # --- ВЫВОД РЕЗУЛЬТАТОВ ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Данные и Статистика", "📈 Графики", "🧪 T-тест", "⚠️ Выбросы"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Очищенные данные")
                st.dataframe(df_processed, use_container_width=True)
            with col2:
                st.subheader("Описательная статистика")
                st.dataframe(df_stats, use_container_width=True)
                
        with tab2:
            st.subheader("Визуализация данных")
            
            # Подготовка данных для Seaborn (melt)
            df_melted = df_processed.melt(var_name="Группа", value_name="МДА, нМ/мг белка").dropna()
            
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                st.markdown("**Boxplot (разброс данных)**")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.boxplot(data=df_melted, x="Группа", y="МДА, нМ/мг белка", ax=ax1, palette="Set2")
                sns.stripplot(data=df_melted, x="Группа", y="МДА, нМ/мг белка", ax=ax1, color=".3", size=6, alpha=0.6)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
                
            with col_plot2:
                st.markdown("**Столбчатая диаграмма (Среднее ± SE)**")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                # Строим barplot, estimator=mean по умолчанию, errorbar='se' для стандартной ошибки
                sns.barplot(data=df_melted, x="Группа", y="МДА, нМ/мг белка", ax=ax2, errorbar='se', capsize=.1, palette="pastel")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)
                
        with tab3:
            st.subheader("Сравнение с контролем (t-критерий Стьюдента Уэлча)")
            if not df_ttest.empty:
                st.dataframe(df_ttest, use_container_width=True)
                st.markdown("*ns: p > 0.05, \*: p < 0.05, \*\*: p < 0.01, \*\*\*: p < 0.001*")
                st.info("Используется t-тест Уэлча, который не предполагает равенства дисперсий, что является более надежным подходом для биологических данных.")
            else:
                st.warning("Недостаточно данных для проведения t-теста.")
                
        with tab4:
            st.subheader("Выбросы по Q-критерию Диксона")
            if outliers_info:
                st.table(pd.DataFrame(outliers_info))
                if remove_outliers:
                    st.success("Выбросы исключены из расчетов статистики и графиков.")
                else:
                    st.warning("Выбросы найдены, но НЕ исключены из расчетов (согласно настройке в боковом меню).")
            else:
                st.success("Выбросы в выборках не обнаружены.")

        # --- КНОПКА СКАЧИВАНИЯ ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Сохранение результатов")
        
        excel_data = create_excel_download(df_processed, df_stats, df_ttest, outliers_info)
        
        st.sidebar.download_button(
            label="📥 Скачать результаты (Excel)",
            data=excel_data,
            file_name="MDA_Analysis_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
            
    except Exception as e:
        st.error(f"Произошла ошибка при обработке файла. Убедитесь, что формат соответствует ожидаемому. Детали: {e}")
else:
    st.info("Пожалуйста, загрузите CSV или Excel файл через панель слева.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=100)
    st.markdown("""
    **Требования к файлу:**
    - Приложение найдет строку с названиями групп (например: Контроль, CeO2, Gd2O3 и т.д.).
    - Значения должны располагаться в столбцах под соответствующими группами.
    """)