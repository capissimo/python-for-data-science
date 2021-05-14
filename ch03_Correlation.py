# [А. Л.], https://capissimo.github.io/pythonfordatascience, 2021

## Глава 03. Корреляция

### Содержание
'''
- [Обследование данных]
- [Визуализация данных]
- [Логнормальное распределение]
- [Ковариация]
- [Корреляция Пирсона]
- [Тестирование статистических гипотез]
- [Интервалы уверенности]
- [Регрессия]
- [Обычные наименьшие квадраты]
- [Качество подгонки и R-квадрат]
- [Множественная линейная регрессия и матрицы]
- [Нормальное уравнение]
- [Множественный R-квадрат]
- [Скорректированный матричный R-квадрат]
- [Коллинеарность]
- [Предсказание]
'''

# -*- coding: utf-8 -*-

# Системные библиотеки

import random
import numpy as np
from scipy import stats
import pandas as pd    
# для загрузки файлов excel требуется xlrd >= 0.9.0 
# ?pip install --upgrade xlrd

# Графические настройки 

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']     = 'sans-serif'
rcParams['font.sans-serif'] = ['Ubuntu Condensed']
rcParams['figure.figsize']  = (5, 4.1)
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9

def saveplot(dest):
    plt.tight_layout()
    plt.savefig('images/' + dest)

### Обследование данных

# Загрузка данных:

def load_data_excel(fname):
    '''Загрузить данные из файла Excel fname'''
    return pd.read_excel('data/ch03/' + fname)   

def load_data():
    '''Загрузить данные об олимпийских спортсменах'''
    return pd.read_csv('data/ch03/all-london-2012-athletes-ru.tsv', '\t') 

def ex_3_1():
    '''Загрузка данных об участниках 
       олимпийских игр в Лондоне 2012 г.'''
    return load_data().head(4)

ex_3_1()

### Визуализация данных

def ex_3_2():
    '''Визуализаия разброса значений 
       роста спортсменов на гистограмме'''
    df = load_data()
    df['Рост, см'].hist(bins=20)
    plt.xlabel('Рост, см.')
    plt.ylabel('Частота')
    saveplot('ex_3_2.png')  
    plt.show() 
    
ex_3_2()

def ex_3_3():
    '''Визуализаия разброса значений веса спортсменов'''
    df = load_data()
    df['Вес'].hist(bins=20)
    plt.xlabel('Вес')
    plt.ylabel('Частота')
    saveplot('ex_3_3.png')  
    plt.show() 
    
ex_3_3()

def ex_3_4():
    '''Вычисление ассиметрии веса спортсменов'''
    df = load_data()
    swimmers = df[ df['Вид спорта'] == 'Swimming']
    return swimmers['Вес'].skew()
    
ex_3_4()  

def ex_3_5():
    '''Визуализаия разброса значений веса спортсменов на
       полулогарифмической гистограмме с целью удаления 
       ассиметрии'''
    df = load_data()
    df['Вес'].apply(np.log).hist(bins=20)
    plt.xlabel('Логарифмический вес')
    plt.ylabel('Частота')
    saveplot('ex_3_5.png')  
    plt.show() 
    
ex_3_5()

### Логнормальное распределение

# Визуализация корреляции

def swimmer_data():
    '''Загрузка данных роста и веса только олимпийских пловцов'''
    df = load_data()
    return df[df['Вид спорта'] == 'Swimming'].dropna()

def ex_3_6():
    '''Визуализация корреляции между ростом и весом'''
    df = swimmer_data()
    xs = df['Рост, см']
    ys = df['Вес'].apply( np.log )
    pd.DataFrame(np.array([xs,ys]).T).plot.scatter(0, 1, s=3, grid=True)
    plt.xlabel('Рост, см.')
    plt.ylabel('Логарифмический вес')
    saveplot('ex_3_6.png')  
    plt.show()
    
ex_3_6()

# Генерирование джиттера

def jitter(limit):
    '''Генератор джиттера (произвольного сдвига точек данных)'''
    return lambda x: random.uniform(-limit, limit) + x

# как вариант: 
# jitter = lambda limit: lambda x: random.uniform(-limit, limit) + x
# пример вызова: jitter(0.5)(7)

def ex_3_7():
    '''Визуализация корреляции между ростом и весом с джиттером'''
    df = swimmer_data()
    xs = df['Рост, см'].apply(jitter(0.5))
    ys = df['Вес'].apply(jitter(0.5)).apply(np.log)
    pd.DataFrame(np.array([xs,ys]).T).plot.scatter(0, 1, s=3, grid=True)
    plt.xlabel('Рост, см.')
    plt.ylabel('Логарифмический вес')
    saveplot('ex_3_7.png')  
    plt.show()
    
ex_3_7()

### Ковариация

def covariance(xs, ys):
    '''Вычисление ковариации (несмещенная, т.е. n-1)'''
    dx = xs - xs.mean() 
    dy = ys - ys.mean()
    return (dx * dy).sum() / (dx.count() - 1)

def ex_3_custom():
    '''Вычисление ковариации 
       на примере данных роста и веса'''
    df = swimmer_data()
    return covariance(df['Рост, см'], df['Вес'].apply(np.log))

ex_3_custom()

def ex_3_pandas():
    '''Вычисление ковариации (несмещенная, т.е. n-1) в pandas
       на примере данных роста и веса'''
    df = swimmer_data()
    return df['Рост, см'].cov(df['Вес'].apply(np.log))

ex_3_pandas()

### Корреляция Пирсона

def variance(xs):
    '''Вычисление дисперсии,
       несмещенная дисперсия при n <= 30'''
    x_hat = xs.mean()
    n = xs.count()
    n = n - 1 if n in range( 1, 30 ) else n  
    return sum((xs - x_hat) ** 2) / n

def standard_deviation(xs):
    '''Вычисление стандартного отклонения'''
    return np.sqrt(variance(xs))

def correlation(xs, ys): 
    '''Вычисление корреляции'''
    return covariance(xs, ys) / (standard_deviation(xs) * 
                                 standard_deviation(ys))

def ex_3_8_custom():
    '''Вычисление корреляции в собственной имплементации
       на примере данных роста и веса'''
    df = swimmer_data()[['Рост, см', 'Вес']]
    return correlation( df['Рост, см'], df['Вес'].apply(np.log))

ex_3_8_custom()

def ex_3_8():
    '''Вычисление корреляции средствами Pandas
       на примере данных роста и веса'''
    df = swimmer_data()
    return df['Рост, см'].corr( df['Вес'].apply(np.log))
    
ex_3_8()

### Тестирование статистических гипотез

def t_statistic(xs, ys):
    '''Вычисление t-статистики
       t = corr * sqrt(df / (1 - corr**2))'''
    r = xs.corr(ys)  # как вариант, correlation(xs, ys)
    df = xs.count() - 2
    return r * np.sqrt(df / 1 - r ** 2)

def ex_3_9():
    '''Выполнение двухстороннего t-теста'''
    df = swimmer_data()
    xs = df['Рост, см']
    ys = df['Вес'].apply(np.log)
    t_value = t_statistic(xs, ys)
    df = xs.count() - 2 
    p = 2 * stats.t.sf(t_value, df)  # функция выживания (иногда лучше, чем 1-t.cdf)
    return {'t-значение':t_value, 'p-значение':p}
    
ex_3_9()

### Интервалы уверенности

def critical_value(confidence, ntails): # ДИ и число хвостов
    '''Расчет критического значения путем
       вычисления квантиля и получения 
       для него нормального значения'''
    lookup = 1 - ((1 - confidence) / ntails) 
    return stats.norm.ppf(lookup, 0, 1)  # mu=0, sigma=1

critical_value(0.95, 2)

def z_to_r(z):
    '''Преобразование z-оценки в r-значение'''
    return (np.exp(z*2) - 1) / (np.exp(z*2) + 1)

def r_confidence_interval(crit, xs, ys):
    '''Расчет интервала уверенности для критического значения и данных'''
    r   = xs.corr(ys)
    n   = xs.count()
    zr  = 0.5 * np.log((1 + r) / (1 - r)) 
    sez = 1 / np.sqrt(n - 3)
    return (z_to_r(zr - (crit * sez))), (z_to_r(zr + (crit * sez)))

def ex_3_10():
    '''Расчет интервала уверенности 
       на примере данных роста и веса'''
    df = swimmer_data()
    X = df['Рост, см']
    y = df['Вес'].apply(np.log)
    interval = r_confidence_interval(1.96, X, y) 
    print('Интервал уверенности (95%):', interval)
    
ex_3_10()

### Регрессия

# Линейные уравнения

'''Функция перевода из шкалы Цельсия в шкалу Фаренгейта'''
celsius_to_fahrenheit = lambda x: 32 + (x * 1.8)

def ex_3_11():
    '''График линейной зависимости температурных шкал'''
    s  = pd.Series(range(-10,40))
    df = pd.DataFrame({'C':s, 'F':s.map(celsius_to_fahrenheit)})
    df.plot('C', 'F', legend=False, grid=True)
    plt.xlabel('Градусы Цельсия')
    plt.ylabel('Градусы Фаренгейта')
    saveplot('ex_3_11.png')  
    plt.show()

ex_3_11()

### Обычные наименьшие квадраты

# Наклон и пересечение

def slope(xs, ys):
    '''Вычисление наклона линии (коэффициента наклона)'''
    return xs.cov(ys) / xs.var()

def intercept(xs, ys): 
    '''Вычисление точки пересечения с осью Y (коэффициента сдвига)'''
    return ys.mean() - (xs.mean() * slope(xs, ys))

def ex_3_12():
    '''Вычисление пересечения и наклона (углового коэффициента) 
       на примере данных роста и веса'''
    df = swimmer_data()
    X  = df['Рост, см']
    y  = df['Вес'].apply(np.log)
    a  = intercept(X, y)
    b  = slope(X, y) 
    print('Пересечение: %f, наклон: %f' % (a,b))
    
ex_3_12()

# Визуализация

'''Функция линии регрессии'''
regression_line = lambda a, b: lambda x: a + (b * x)  # вызовы fn(a,b)(x)

def ex_3_13():
    '''Визуализация линейного уравнения
       на примере данных роста и веса'''
    df = swimmer_data()
    X  = df['Рост, см'].apply( jitter(0.5) )
    y  = df['Вес'].apply(np.log)
    a, b = intercept(X, y), slope(X, y) 
    ax = pd.DataFrame(np.array([X, y]).T).plot.scatter(0, 1, s=3)
    s  = pd.Series(range(150,210))
    df = pd.DataFrame( {0:s, 1:s.map(regression_line(a, b))} )  
    df.plot(0, 1, legend=False, grid=True, ax=ax)
    plt.xlabel('Рост, см.')
    plt.ylabel('Логарифмический вес')
    saveplot('ex_3_13.png')  
    plt.show()
    
ex_3_13()

def residuals(a, b, xs, ys):
    '''Вычисление остатков (остаточных расстояний)'''
    estimate = regression_line(a, b)         
    return pd.Series( map(lambda x, y: y - estimate(x), xs, ys) )

constantly = lambda x: 0

def ex_3_14():
    '''Построение графика остатков на примере данных роста и веса'''
    df = swimmer_data()
    X  = df['Рост, см'].apply( jitter(0.5) )
    y  = df['Вес'].apply(np.log)
    a, b = intercept(X, y), slope(X, y) 
    y  = residuals(a, b, X, y)
    ax = pd.DataFrame(np.array([X, y]).T).plot.scatter(0, 1, s=3)
    s  = pd.Series(range(150,210))
    df = pd.DataFrame( {0:s, 1:s.map(constantly)} )      
    df.plot(0, 1, legend=False, grid=True, ax=ax)
    plt.xlabel('Рост, см.')
    plt.ylabel('Остатки')
    saveplot('ex_3_14.png')  
    plt.show()
    
ex_3_14()

### Качество подгонки и R-квадрат

def r_squared(a, b, xs, ys):
    '''Рассчитать коэффициент детерминации (R-квадрат)'''
    r_var = residuals(a, b, xs, ys).var() 
    y_var = ys.var()
    return 1 - (r_var / y_var)

def ex_3_15():
    '''Рассчитать коэффициент R-квадрат 
       на примере данных роста и веса'''
    df = swimmer_data()
    X  = df['Рост, см'].apply( jitter(0.5) )
    y  = df['Вес'].apply(np.log)
    a, b = intercept(X, y), slope(X, y) 
    return r_squared(a, b, X, y)

ex_3_15()

### Множественная линейная регрессия и матрицы

'''Конвертирование рядов данных Series и 
   таблиц данных DataFrame библиотеки pandas 
   в матрицы библиотеки numpy'''
pd.Series(random.sample(range(256), k=16)).values

def ex_3_16():
    '''Конвертирование в массив (матрицу) numpy 
       таблицы данных роста и веса'''
    df = swimmer_data()[['Рост, см', 'Вес']]
    return df.values

ex_3_16()

def ex_3_17():
    '''Конвертирование в массив (матрицу) numpy 
       данных числового ряда с данными о росте'''
    return swimmer_data()['Рост, см'].head(20).values

ex_3_17()[0:10]

'''Добавление столбца в таблицу данных (массив)'''
df = pd.DataFrame({'x':[2, 3, 6, 7],'y':[8, 7, 4, 3]})
df['константа'] = 1
df

### Операции над матрицами в pandas

# Конструирование

df1 = pd.DataFrame([[1,0],[2,5],[3,1]])
df2 = pd.DataFrame([[4,0.5],[2,5],[0,1]])
df1

df2

# Сложение и скалярное произведение

# Прибавление скаляра к матрице

df1 + 3

# Сложение матриц

df1 + df2

# Вычитание матриц

df1 - df2

# Умножение матрицы на скаляр

df1 * 3

# Матричное деление на скаляр
# эквивалентно df1.div(3, fill_value=None, axis=0, level=None)

df1 / 3

# Матрично-векторное умножение

df3 = pd.DataFrame([[1,3],[0,4],[2,1]])
vec = [1,5]
df3.dot(vec)

# Матрично-матричное умножение

df3 = pd.DataFrame([[1,3],[0,4],[2,1]])
df4 = pd.DataFrame([[1,0],[5,6]])     
df3.dot(df4)

# Вариант 2

np.matmul(df3,np.asarray(df4))

# Транспонирование

df3.T

# Нейтральная (единичная) матрица

pd.DataFrame(np.identity(5))

# Обратная матрица

# np.random.seed([3,1415])
df5 = pd.DataFrame(np.random.rand(3, 3), list('abc'), list('xyz'))
print(df5)
df_inv = pd.DataFrame(np.linalg.pinv(df5.values), df5.columns, df5.index)
df_inv

# проверяем
df_inv.dot(df5)

### Нормальное уравнение

# Имплементация нормального уравнения

# numpy.linalg.inv(A) фактически вызывает numpy.linalg.solve(A,I), 
# где I - это нейтральная матрица, и находит решение разложением 
# LU матрицы средствами динамической библиотеки lapack
# как вариант:
# numpy.linalg.pinv - вычисляет псевдообратную матрицу (А+) по методу 
# Мура-Пенроуза с использованием сингулярного разложения (SVD)
# np.linalg.pinv(x.values), x.columns, x.index 

def normal_equation(x, y):
    '''Имплементация нормального уравнения'''
    xtx  = np.matmul(x.T.values, x.values) 
    xtxi = np.matmul(np.linalg.inv(np.matmul(xtx.T,xtx)),xtx.T)  # вычислить мультипликативную инверсию матрицы
    xty  = np.matmul(x.T.values, y.values) 
    return np.matmul(xtxi, xty)  

def ex_3_18():
    '''Решение нормального уравнения 
       на примере данных роста и веса'''
    df = swimmer_data()
    X = df[['Рост, см']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply( np.log )
    return normal_equation(X, y)

ex_3_18()

# Дополнительные признаки

def ex_3_19():
    '''Пример создания матрицы признаков NumPy
       на примере данных роста и возраста'''
    X = swimmer_data()[['Рост, см', 'Возраст']]
    X.insert(0, 'константа', 1)
    return X.values

ex_3_19()

def ex_3_20():
    '''Решение нормального уравнения 
       для данных роста и возраста в качестве независимых и 
       веса в качестве зависимой переменной'''
    df = swimmer_data()
    X = df[['Рост, см', 'Возраст']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply(np.log)
    return normal_equation(X, y)

ex_3_20()

### Множественный R-квадрат

def matrix_r_squared(coefs, x, y):
    '''Вычислить матричный R-квадрат'''
    fitted      = x.dot(coefs) 
    residuals   = y - fitted 
    difference  = y - y.mean()  
    rss         = residuals.dot(residuals)  # сумма квадратов
    ess         = difference.dot(difference)
    return 1 - (rss / ess)

def ex_3_21():
    '''Вычислить матричный R-квадрат 
       на данных роста и возраста в качестве независимых и 
       веса в качестве зависимой переменной'''
    df = swimmer_data()
    X = df[['Рост, см', 'Возраст']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply(np.log)
    beta = normal_equation(X, y) 
    return matrix_r_squared(beta, X, y)

ex_3_21()

### Скорректированный матричный R-квадрат

def matrix_adj_r_squared(coefs, x, y):
    '''Вычислить скорректированный матричный R-квадрат'''
    r_squared = matrix_r_squared(coefs, x, y) 
    n = y.shape[0]  # строки
    p = coefs.shape[0]
    dec = lambda x: x-1
    return 1 - (1 - r_squared) * (dec(n) / dec(n-p))

def ex_3_22():
    '''Вычислить скорректированный матричный R-квадрат 
       на данных роста и возраста в качестве независимых и 
       веса в качестве зависимой переменной'''
    df = swimmer_data()
    X = df[['Рост, см', 'Возраст']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply(np.log)
    beta = normal_equation(X, y) 
    return matrix_adj_r_squared(beta, X, y)

ex_3_22()

# Линейная модель в NumPy и SciPy

'''
функция numpy.linalg.pinv аппроксимирует псевдоинверсию Мура-Пенроуза 
с использованием SVD (точнее, метод dgesdd динамической библиотеки lapack), 
тогда как scipy.linalg.pinv находит решение линейной системы с точки 
зрения наименьших квадратов, чтобы аппроксимировать псевдоинверсию 
(пользуясь dgelss). Отсюда и разница и в производительности.

Производительность функции scipy.linalg.pinv2 сравнима с   
numpy.linalg.pinv, т.к. в ней используется метод SVD, вместо 
аппроксимации методом наименьших квадратов'''

def numpy_scipy():
    '''Линейная регрессия в NumPy и SciPy'''
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.5,2.5))
    xi = np.arange(0,9)
    A  = np.array([xi, np.ones(9)])
    y  = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24] # линейная последовательность

    # имплементация в numpy
    (a, b) = np.linalg.lstsq(A.T, y, rcond=-1)[0] # получение параметров
    line = a * xi + b                   # линия регрессии 
    ax[0].plot(xi, line, 'r-', xi, y, '.')
    ax[0].grid(True)

    # имплементация в scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi, y)
    line = slope * xi + intercept
    #print('r-значение', r_value, 'p-значение', p_value, 'СТО', std_err)
    ax[1].plot(xi, line, 'r-', xi, y, '.')
    ax[1].grid(True)
    plt.show()
    
numpy_scipy()

def linear_model(x, y):
    '''Обертка вокруг библиотечной функции 
       линейной регрессии наименьшими квадратами, 
       вместо собственной имплементации нормального уравнения normal_equation'''
    return np.linalg.lstsq(x,y,rcond=-1)[0]
    #return stats.linregress(xi, y)  # наклон, пересечение, r-значение, p-значение, сош 

def ex_3_linear_model():
    '''Проверка функции linear_model'''
    df = swimmer_data()
    X = df[['Рост, см', 'Возраст']]
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log)

    beta = linear_model(X,y)
    return beta   # проверка модели np.dot(Xi,beta_hat)

ex_3_linear_model()

def f_test(fitted, x, y):
    '''F-тест коэффициентов регрессии'''
    difference = fitted - y.mean() 
    residuals  = y - fitted
    ess        = difference.dot(difference) # сумма квадратов
    rss        = residuals.dot(residuals)
    p          = x.shape[1]    # столбцы
    n          = y.shape[0]    # строки
    df1        = p - 1
    df2        = n - p
    msm        = ess / df1
    mse        = rss / df2
    f_stat     = msm / mse     # mse модели / mse остатков
    f_test     = 1-stats.f.cdf(f_stat, df1, df2) 
    return f_test

def ex_3_23():
    '''Проверка значимости модели на основе F-теста
       на примере данных роста, возраста и веса'''
    df = swimmer_data()
    X = df[['Рост, см', 'Возраст']]
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log)
    beta = linear_model(X, y)    
    fittedvalues = np.dot(X,beta) 

    #model = sm.OLS(y, X)
    #results = model.fit()
    #print(results.summary())
    #print(1-stats.f.cdf(results.fvalue, results.df_model, results.df_resid))
    
    # проверка коэффициентов модели
    return ('F-тест', f_test(fittedvalues, X, y))

ex_3_23()

def ex_3_24():
    '''Проверка значимости модели на основе F-теста
       на произвольной выборке из данных роста, возраста и веса'''
    df = swimmer_data().sample(5)  # произвольная выборка
    df.index = range(len(df))      # задать новый индекс
    X = df[['Рост, см', 'Возраст']]
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 
    beta = linear_model(X, y)    
    fittedvalues = np.dot(X,beta) 

    #model = sm.OLS(y, X)
    #results = model.fit()
    #print(results.summary())
    
    return ('F-тест', f_test(fittedvalues, X, y))

ex_3_24()

# Категориальные и фиктивные переменные

def ex_3_25():
    '''Обработка категориальных признаков 
       (создание двоичной переменной)'''
    df = swimmer_data()
    # как вариант, получить из категориального поля несколько 
    # прямокодированных двоичных полей
    # dummies = pd.get_dummies(df['Пол'], prefix='бин_')  
    # X = df[['Рост, см', 'Возраст']].join(dummies)
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int) # строковое --> числовое

    X = df[['Рост, см', 'Возраст', 'бин_Пол']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply(np.log)  
    
    beta = linear_model(X, y) 
    return matrix_adj_r_squared(beta, X, y)

ex_3_25()

# Относительная мощность

def beta_weight(coefs, x, y):
    '''Вычисление относительного вклада каждого признака'''
    sdx = x.std()
    sdy = y.std()
    return [x / sdy * c for x,c in zip(sdx,coefs)] 

def ex_3_26():
    '''Относительный вклад каждого признака в предсказании веса
       на примере данных роста, возраста и пола'''
    df = swimmer_data()
    # получить двоичное поле
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int)
    X = df[['Рост, см', 'Возраст', 'бин_Пол']] 
    X.insert(0, 'константа', 1)
    y = df['Вес'].apply(np.log) 
    beta = linear_model(X, y) 
    res = beta_weight(beta, X, y)
    #result = sm.OLS(y, X).fit()
    #print(result.summary())    
    return res

ex_3_26()

### Коллинеарность

'''Служебная функция приведения строкового 
   представления даты к типу DateTime и извлечение года'''
str_to_year = lambda x: pd.to_datetime(x).year

def ex_3_27():
    '''Относительный вклад признаков в предсказании веса
       с участием признака с датой (год)'''
    df = swimmer_data()
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int) 
    df['Год рождения'] = df['Дата рождения'].map(str_to_year)
    X = df[['Рост, см', 'Возраст', 'бин_Пол', 'Год рождения']] 
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 
    
    beta = linear_model(X, y) 
    #result = sm.OLS(y, X).fit()
    #print(result.summary())    
    return beta_weight(beta, X, y)

ex_3_27()  

def ex_3_28():
    '''График коллинеарности возраста спортсменов и даты их рождения'''
    df = swimmer_data()
    df['Год рождения'] = df['Дата рождения'].map(str_to_year)
    xs = df['Возраст'].apply(jitter(0.5))
    ys = df['Год рождения']
    pd.DataFrame(np.array([xs,ys]).T).plot.scatter(0, 1, s=3, grid=True)
    plt.xlabel('Возраст')
    plt.ylabel('Год рождения')
    saveplot('ex_3_28.png')
    plt.show()
    
ex_3_28()

### Предсказание

def predict(coefs, x): 
    '''функция предсказания'''
    return np.matmul(coefs, x.values) 

def ex_3_29():
    '''Вычисление ожидаемого веса спортсмена'''
    df = swimmer_data()
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int) 
    df['Год рождения'] = df['Дата рождения'].map(str_to_year)
    X = df[['Рост, см', 'бин_Пол', 'Год рождения']] 
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 
    beta = linear_model(X, y)
    xspitz = pd.Series([1.0, 183, 1, 1950]) # параметры Марка Шпитца
    return np.exp( predict(beta, xspitz) )  

ex_3_29()   

# Интервал уверенности для предсказания

def prediction_interval(x, y, xp):
    '''Вычисление интервала предсказания'''
    xtx    = np.matmul(x.T, np.asarray(x))
    xtxi   = np.linalg.inv(xtx)  
    xty    = np.matmul(x.T, np.asarray(y)) 
    coefs  = linear_model(x, y) 
    fitted = np.matmul(x, coefs)
    resid  = y - fitted
    rss    = resid.dot(resid)  
    n      = y.shape[0]  # строки
    p      = x.shape[1]  # столбцы
    dfe    = n - p 
    mse    = rss / dfe
    se_y   = np.matmul(np.matmul(xp.T, xtxi), xp)
    t_stat = np.sqrt(mse * (1 + se_y))         # t-статистика
    intl   = stats.t.ppf(0.975, dfe) * t_stat   
    yp     = np.matmul(coefs.T, xp)
    return np.array([yp - intl, yp + intl])

def ex_3_30():
    '''Интервал предсказания
       применительно к данным о Марке Шпитце'''
    df = swimmer_data()
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int) 
    df['Год рождения'] = df['Дата рождения'].map(str_to_year)
    X = df[['Рост, см', 'бин_Пол', 'Год рождения']] 
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 
    xspitz = pd.Series([1.0, 183, 1, 1950])  # данные М.Шпитца
    return np.exp( prediction_interval(X, y, xspitz) )

ex_3_30()  

def ex_3_CIPI():
    '''Сравнение интервалов уверенности и предсказания
       относительно значений независимой переменной 
       в заданном диапазоне'''
    df = swimmer_data()[['Рост, см', 'Вес']].sample(80) 
    X  = df[['Рост, см']].apply(jitter(0.5))
    X.insert(0, 'константа', 1.0)
    y  = df['Вес'].apply(jitter(0.5)).apply(np.log) 

    # вывести точки данных
    ax = pd.DataFrame(np.array([X['Рост, см'],y]).T).plot.scatter(0, 1, s=3, grid=True)  

    # прямая регрессии
    a  = intercept(X['Рост, см'], y)
    b  = slope(X['Рост, см'], y) 
    s  = pd.Series(range(150 ,210))
    df = pd.DataFrame( {0:s, 1:s.map(regression_line(a, b))} )  
    ax = df.plot(0, 1, color='r', legend=False, ax=ax)

    # интервал предсказания (нижняя и верхняя границы)
    bound = lambda i: lambda x: prediction_interval(X, y, pd.Series([1.0, x]))[i]
    defbound = lambda b: pd.DataFrame( {0:s, 1:s.map(bound(b))} )
    lo, hi = defbound(0), defbound(1)
    ax = lo.plot(0, 1, color='b', legend=False, ax=ax)
    ax = hi.plot(0, 1, color='b', legend=False, ax=ax)
    ax.fill_between(lo[0], lo[1], hi[1], alpha=0.2, facecolor='lightblue', interpolate=True) 
    
    # интервал уверенности
    n      = X.shape[0]
    alpha  = 0.05
    fit    = lambda xx: a + b*xx  
    dfr    = n-2
    tval   = stats.t.isf(alpha/2., dfr) # соответствующее t-значение
    se_fit = lambda x: np.sqrt(np.sum((y - fit(x))**2)/(n-2)) * \
                       np.sqrt(1./n + (x-np.mean(x))**2/(np.sum(x**2) - np.sum(x)**2/n))

    X = X['Рост, см'].sort_values()
    ax.plot(X, fit(X)-tval*se_fit(X), 'g')
    ax.plot(X, fit(X)+tval*se_fit(X), 'g')
    
    # настройка графика 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(min(y)-0.2, max(y)+0.2)
    ax.tick_params(labelbottom='off')
    ax.tick_params(labelleft='off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    saveplot('ex_3_CIPI.png')
    plt.show()

ex_3_CIPI()

def ex_3_31():
    '''График изменения интервала предсказания
       относительно значений независимой переменной 
       в заданном диапазоне '''
    df = swimmer_data().sample(5)  # произвольная выборка
    X = df[['Рост, см']]
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 

    bound = lambda i: lambda x: prediction_interval(X, y, pd.Series([1.0, x]))[i]
    s = pd.Series(range(150 ,210))
    defbound = lambda b: pd.DataFrame( {0:s, 1:s.map(bound(b))} )

    df = pd.DataFrame(np.array([X['Рост, см'],y]).T)
    ax = df.plot.scatter(0, 1, s=3, grid=True)  
    ax = defbound(0).plot(0, 1, legend=False, color='g', ax=ax)
    defbound(1).plot(0, 1, legend=False, grid=True, color='g', ax=ax)
    
    plt.xlabel('Рост, см')
    plt.ylabel('Логарифмический вес')
    saveplot('ex_3_31.png')
    plt.show()

ex_3_31()

# Окончательная модель

def ex_3_32():
    '''Окончательная модель для предсказания 
       соревновательного веса'''
    df = swimmer_data()
    df['бин_Пол'] = df['Пол'].map({'М': 1, 'Ж': 0}).astype(int) 
    X = df[['Рост, см', 'бин_Пол', 'Возраст']] 
    X.insert(0, 'константа', 1.0)
    y = df['Вес'].apply(np.log) 

    beta = linear_model(X, y)
    # предсказать вес для М.Шпитца
    xspitz = pd.Series([1.0, 185, 1, 22]) 
    return np.exp( predict(beta, xspitz) )

ex_3_32()   

