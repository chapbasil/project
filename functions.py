import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay


# Функция для вывода текста на график\n",
def plot_text(ax: plt.Axes):

    "Вывод текста на графике barplot",

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            # координата xy
            (p.get_x() + p.get_width() / 2., p.get_height()),
            # центрирование
            ha='center',
            va='center',
            xytext=(0, 10),
            # точка смещения относительно координаты
            textcoords='offset points',
            fontsize=14)

        
def normalize_target(df: pd.Series):
    
    
    norm_target = (df.value_counts(
    normalize=True).mul(100).rename('percent').reset_index())

    # Вывод фигуры графика
    plt.figure(figsize=(15, 7))
    ax = sns.barplot(x=df, y='percent', data=norm_target, palette='Spectral')
    # Вывод надписей
    plot_text(ax)
    plt.title('Соотношение ожиданий трудоустройства', fontsize=20)
    plt.xlabel('0 - Будет безработный, 1 - Будетрудоустроен', fontsize=14)
    plt.ylabel('Проценты', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    
def fullfeat(data: pd.DataFrame):
    '''Функция для обработки предсказаний тестового набора'''

    # Удаляем неинформативные признаки: ID, Дату опроса и столбец, 
    # в котором практически все значения одинаковые.
    data = data.drop(["Person_id", "Survey_date", "Sa_citizen", "Round"], axis=1)

    # Часть пропусков можно заполнить нулями
    data['Tenure'] = np.where(data.Status == 'studying', 0.0, data.Tenure)
    data['Tenure'] = np.where(data.Status == 'other', 0.0, data.Tenure)

    return data
    
    
# Функция проверки двух категориальных переменных с помощью хи-квадрат и p-value
def check_chi2(x: pd.Series, y: pd.Series, alpha: float = 0.05):
   
    ct_table_ind = pd.crosstab(x, y)
    chi2_stat, p, dof, expected = stats.chi2_contingency(ct_table_ind)
    print(f"chi2 statistic: {chi2_stat:.5g}")
    print(f"p-value {p:.5g}")

    if p < alpha:
        print("Две категориальные переменные имеют значимую связь")
    else:
        print("Две категориальные переменные не имеют значимой связи")
    
    
def barplot_group(df_data: pd.DataFrame, col_main: str, col_group: str,
                  title: str) -> None:
    """
    Построение barplot с нормированными данными с выводом значений на графике
    """

    plt.figure(figsize=(15, 6))

    data = (df_data.groupby(
        [col_group])[col_main].value_counts(normalize=True).rename(
            'percentage').mul(100).reset_index().sort_values(col_group))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data,
                     palette="ch:start=.2,rot=-.3"
                     )

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            (p.get_x() + p.get_width() / 2., p.get_height()),  # координата xy
            ha='center',  # центрирование
            va='center',
            xytext=(0, 7),
            textcoords='offset points',  # точка смещения относительно координаты
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.show()
    
    
# Функция для проверки переобучения
def check_overfitting(metric_fun,
                      y_train,
                      y_test,
                      X_train=None,
                      X_test=None,
                      model=None,
                      y_train_proba=None,
                      y_test_proba=None):
    """
    Проверка на overfitting
    """
    if model is None:
        value_train = metric_fun(y_train, y_train_proba)
        value_test = metric_fun(y_test, y_test_proba)
    else:
        if metric_fun.__name__ == 'roc_auc_score':
            y_pred_train = model.predict_proba(X_train)[:, 1]
            y_pred_test = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        value_train = metric_fun(y_train, y_pred_train)
        value_test = metric_fun(y_test, y_pred_test)

    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} test: %.3f' % value_test)
    print(f'delta = {(abs(value_train - value_test)/value_test*100):.1f} %')
 

def plot_confusion_matrix(y_test, X_test, ax, model=None, prediction=None):
    """Визуализация ConfusionMatrix"""
    if prediction is None:
        prediction = model.predict(X_test)
        
    labels = list(set(prediction))
    cm_ovr = confusion_matrix(y_test, prediction, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ovr, display_labels=labels)
    
    if ax:
        disp.plot(ax=ax)
        
        
def duomatrix(y_test, X_test, title1: str, title2: str, model1=None, model2=None):        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    plt.rcParams.update({'font.size': 16})
    plot_confusion_matrix(y_test, X_test, model=model1, ax=ax[0])
    plt.rcParams.update({'font.size': 16})
    plot_confusion_matrix(y_test, X_test, model=model2, ax=ax[1])

    ax[0].title.set_text(title1)
    ax[1].title.set_text(title2)
    plt.tight_layout()  
    plt.show()
    
    
def get_metrics(X_train, X_test, y_test: np.array, y_pred: np.array, \
                y_proba: np.array, y_train: np.array, model=None, ) -> Dict:
    
    value_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    value_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    overfit = abs(value_train - value_test)/value_test*100
    dict_metrics = {
        'roc_auc': round(roc_auc_score(y_test, y_proba[:, 1]), 3),
        'precision': round(precision_score(y_test, y_pred), 3),
        'recall': round(recall_score(y_test, y_pred), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
        'logloss': round(log_loss(y_test, y_proba), 3),
        'overfitting': round((abs(value_train - value_test)/value_test*100), 1)
    }

    return dict_metrics

def replace_values(data: pd.DataFrame, map_change_columns: dict) -> pd.DataFrame:
    """
    Замена значений в датасете
    :param data: датасет
    :param map_change_columns: словарь с признаками и значениями
    :return: датасет
    """
    return data.replace(map_change_columns)