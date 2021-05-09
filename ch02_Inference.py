#!/usr/bin/env python
# coding: utf-8

# <link rel="stylesheet" href="fonts/css/font-awesome.min.css">
# [А. Л.](https://capissimo.github.io/pythonfordatascience), 2021
# 
# https://capissimo.github.io/pythonfordatascience

# ## Глава 2. Статистический вывод

# ### Содержание

# - [Обследование данных](#Обследование-данных)
# - [Визуализация времени задержки](#Визуализация-времени-задержки)
# - [Экспоненциальное распределение](#Экспоненциальное-распределение)
# - [Центральная предельная теорема](#Центральная-предельная-теорема)
# - [Стандартная ошибка](#Стандартная-ошибка)
# - [Выборки и популяции](#Выборки-и-популяции)
# - [Визуализация разных популяций](#Визуализация-разных-популяций)
# - [Проверка обновленного дизайна веб-сайта](#Проверка-обновленного-дизайна-веб-сайта)
# - [t-статистика](#t-статистика)
# - [t-тест](#t-тест)
# - [Одновыборочный t-тест](#Одновыборочный-t-тест)
# - [Многократные выборки](#Многократные-выборки)
# - [Проверка многочисленных вариантов дизайна веб-сайта](#Проверка-многочисленных-вариантов-дизайна-веб-сайта)
# - [Поправка Бонферрони](#Поправка-Бонферрони)
# - [F-распределение](#F-распределение)
# - [F-статистика](#F-статистика)
# - [F-тест](#F-тест)
# - [Размер эффекта](#Размер-эффекта)

# <a name="home"></a>

# In[19]:


# -*- coding: utf-8 -*-

# Системные библиотеки

import scipy as sp
import pandas as pd    
from scipy import stats

# Графические настройки 

import matplotlib.pyplot as plt
   
from matplotlib import rcParams
rcParams['font.family']     = 'sans-serif'
rcParams['font.sans-serif'] = ['Ubuntu Condensed']
rcParams['figure.figsize']  = (5, 4.05)
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9

def saveplot(dest):
    plt.tight_layout()
    plt.savefig('images/' + dest)  


# ### Обследование данных

# In[2]:


def load_data(fname):
    '''Загрузка данных из файла fname'''
    return pd.read_csv('data/ch02/' + fname, '\t')  


# In[3]:


def ex_2_1():
    '''Загрузка данных времени задержки на веб-сайте'''
    return load_data('dwell-times.tsv')[0:5:]  # как вариант, head()
    
ex_2_1()


# ### Визуализация времени задержки

# In[20]:


def ex_2_2():
    '''Визуализация времени задержки'''
    load_data('dwell-times.tsv')['dwell-time'].hist(bins=50)
    plt.xlabel('Время задержки, сек.')
    plt.ylabel('Частота')
    #saveplot('ex_2_2.png')  
    plt.show()    
    
ex_2_2()


# In[21]:


def ex_2_3():
    '''Визуализация времени задержки
       на полулогарифмическом графике'''
    load_data('dwell-times.tsv')['dwell-time'].plot.hist(bins=20, logy=True, grid=True)
    plt.xlabel('Время задержки, сек.')
    plt.ylabel('Логарифмическая частота')
    #saveplot('ex_2_3.png')  
    plt.show()    
    
ex_2_3()


# ### Экспоненциальное распределение

# In[6]:


def ex_2_4():
    '''Показать сводные статистики набора данных,
       подчиняющегося экспоненциональному распределению'''
    ts = load_data('dwell-times.tsv')['dwell-time']
    print('Среднее:                ', ts.mean())    
    print('Медиана:                ', ts.median())
    print('Стандартное отклонение: ', ts.std())
    
ex_2_4()


# *Распределение среднесуточных значений*

# In[8]:


# Определение служебных функций по обработке дат

def with_parsed_date(df):
    '''Привести поле date к типу date-time'''
    df['date'] = pd.to_datetime(df['date'], errors='ignore')
    return df

def filter_weekdays(df): 
    '''Отфильтровать по будним дням'''
    return df[df['date'].index.dayofweek < 5]  # понедельник..пятница

def mean_dwell_times_by_date(df):
    '''Среднесуточные времена задержки'''
    df.index = with_parsed_date(df)['date']
    return df.resample('D').mean()  # перегруппировать  

def daily_mean_dwell_times(df):
    '''Средние времена задержки с фильтрацией - только по будним дням'''
    df.index = with_parsed_date(df)['date']
    df = filter_weekdays(df)
    return df.resample('D').mean() 


# In[68]:


def ex_2_5():
    '''Распределение значений в будние дни'''
    df  = load_data('dwell-times.tsv')    
    means = daily_mean_dwell_times(df)
    print('Среднее:                ', float(means.mean()))    
    print('Медиана:                ', float(means.median()))
    print('Стандартное отклонение: ', float(means.std()))
    
ex_2_5()


# In[22]:


def ex_2_6():
    '''Построить гистограмму значений 
       времени задержки в будние дни'''
    df = load_data('dwell-times.tsv')
    daily_mean_dwell_times(df)['dwell-time'].hist(bins=20)
    plt.xlabel('Время задержки по будним дням, сек.')
    plt.ylabel('Частота')
    #saveplot('ex_2_6.png')  
    plt.show()    
    
ex_2_6()


# ### Центральная предельная теорема

# In[23]:


def ex_2_7():
    '''Подгонка нормальной кривой поверх гистограммы'''
    df = load_data('dwell-times.tsv')
    means = daily_mean_dwell_times(df)['dwell-time'].dropna() 
    ax = means.hist(bins=20, density=True)
    xs = sorted(means)    # корзины
    df = pd.DataFrame()
    df[0] = xs
    df[1] = stats.norm.pdf(xs, means.mean(), means.std())
    df.plot(0, 1, linewidth=2, color='r', legend=None, ax=ax)
    plt.xlabel('Время задержки по будним дням, сек.')
    plt.ylabel('Плотность')
    plt.grid(True)
    #saveplot('ex_2_7.png')  
    plt.show()
    
ex_2_7()


# ### Стандартная ошибка

# In[24]:


# Определение функции стандартной ошибки среднего

def variance(xs):
    '''Вычисление дисперсии,
       несмещенная дисперсия при n <= 30'''
    x_hat = xs.mean() 
    n = len(xs)
    n = n-1 if n in range(1, 30) else n  
    square_deviation = lambda x : (x - x_hat) ** 2 
    return sum( map(square_deviation, xs) ) / n

def standard_deviation(xs):
    '''Вычисление стандартного отклонения'''
    return sp.sqrt(variance(xs))

def standard_error(xs):
    '''Вычисление стандартной ошибки'''
    return standard_deviation(xs) / sp.sqrt(len(xs))


# ### Выборки и популяции

# In[69]:


def ex_2_8(): 
    '''Вычислить стандартную ошибку 
       средних значений за определенный день'''
    may_1 = '2015-05-01'
    df = with_parsed_date( load_data('dwell-times.tsv') )  
    filtered = df.set_index( ['date'] )[may_1]
    se = standard_error( filtered['dwell-time'] )
    print('Стандартная ошибка:', se)

ex_2_8()


# In[26]:


def confidence_interval(p, xs):
    '''Интервал уверенности'''
    x_hat = xs.mean()
    se = standard_error(xs)
    '''критическое значение z
    Критическое значение z - это число стандартных отклонений, на которые 
    нужно отойти от среднего значения нормального распределения, чтобы захватить 
    долю данных, связанную с нужным интервалом уверенности.'''
    z_crit = stats.norm.ppf(1 - (1-p) / 2)  #q=0.975 -> 1.96
    return [x_hat - z_crit * se, x_hat + z_crit * se]


# In[27]:


def ex_2_9():
    '''Вычислить интервал уверенности 
       для данных за определенный день'''
    may_1 = '2015-05-01'
    df    = with_parsed_date( load_data('dwell-times.tsv') )  
    filtered = df.set_index( ['date'] )[may_1]
    ci = confidence_interval(0.95, filtered['dwell-time'])
    print('Интервал уверенности: ', ci)
    
ex_2_9()


# *Сравнение выборок*

# In[28]:


def ex_2_10():
    '''Сводные статистики данных, полученных 
       в результате вирусной кампании'''
    ts = load_data('campaign-sample.tsv')['dwell-time']     
    print('n:                      ', ts.count())
    print('Среднее:                ', ts.mean())
    print('Медиана:                ', ts.median())
    print('Стандартное отклонение: ', ts.std())
    print('Стандартная ошибка:     ', standard_error(ts))    

ex_2_10()


# In[29]:


def ex_2_11():
    '''Интервал уверенности для данных,
       полученных в результате вирусной кампании'''
    ts = load_data('campaign-sample.tsv')['dwell-time']     
    print('Интервал увернности:', confidence_interval(0.95, ts))

ex_2_11()    


# *Смещение*

# In[30]:


'''Проверка даты''' 
# будние дни: 0..4, выходные дни: 5-6
d = pd.to_datetime('2015 6 6') 
d.weekday() in [5,6]


# ### Визуализация разных популяций

# In[31]:


def ex_2_12(): 
    '''Построить график времени ожидания 
       по всем дням без фильтра'''
    df = load_data('dwell-times.tsv')
    means = mean_dwell_times_by_date(df)['dwell-time']
    means.hist(bins=20)
    plt.xlabel('Ежедневное время ожидания без фильтра, сек.')
    plt.ylabel('Частота')
    #saveplot('ex_2_12.png') 
    plt.show()     

ex_2_12()


# In[32]:


def ex_2_13():
    '''Сводные статистики данных,
       отфильтрованных только по выходным дням'''
    df = with_parsed_date( load_data('dwell-times.tsv') )
    df.index = df['date']
    df = df[df['date'].index.dayofweek > 4]   # суббота-воскресенье
    weekend_times = df['dwell-time']
  
    print('n:                      ', weekend_times.count())
    print('Среднее:                ', weekend_times.mean())
    print('Медиана:                ', weekend_times.median())
    print('Стандартное отклонение: ', weekend_times.std())
    print('Стандартная ошибка:     ', standard_error(weekend_times))        
    
ex_2_13()


# ### Проверка обновленного дизайна веб-сайта 

# *Выполнение z-теста*

# In[33]:


def pooled_standard_error(a, b, unbias=False):
    '''Объединенная стандартная ошибка'''
    std1 = a.std(ddof=0) if unbias==False else a.std()  # ddof=0 = смещенное значение
    std2 = b.std(ddof=0) if unbias==False else b.std()
    x = std1 ** 2 / a.count()
    y = std2 ** 2 / b.count()
    return sp.sqrt(x + y)    

def z_stat(a, b, unbias=False):
    '''z-статистика'''
    return (a.mean() - b.mean()) / pooled_standard_error(a, b, unbias)    

def z_test(a, b): 
    '''z-тест'''
    # ИФР нормального распределения
    return stats.norm.cdf([ z_stat(a, b) ])


# In[35]:


def ex_2_14():
    '''Сравнение результативности двух вариантов
       дизайна веб-сайта на основе z-теста'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    a = groups.get_group(0)
    b = groups.get_group(1) 
    
    print('a n:         ', a.count())
    print('b n:         ', b.count())
    print('z-статистика:', z_stat(a, b))
    print('p-значение:  ', z_test(a, b))
    
ex_2_14()


# ### t-статистика

# In[36]:


def pooled_standard_error_t(a, b):
    '''Объединенная стандартная ошибка для t-теста'''
    return sp.sqrt(standard_error(a) ** 2 + 
                   standard_error(b) ** 2) 


# In[37]:


t_stat = z_stat

def ex_2_15():
    '''Вычисление t-статистики 
       двух вариантов дизайна веб-сайта'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    a = groups.get_group(0)
    b = groups.get_group(1)    
    return t_stat(a, b)
    
ex_2_15()


# ### t-тест

# In[38]:


def t_test(a, b):
    '''Выполнение проверки на основе t-теста'''
    df = len(a) + len(b) - 2
    # http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.t.html
    return stats.t.sf([ abs(t_stat(a, b)) ], df)      # функция выживания (1-cdf иногда точнее)


# In[39]:


def ex_2_16():
    '''Сравнение результативности двух вариантов 
       дизайна веб-сайта на основе t-теста'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    a = groups.get_group(0)
    b = groups.get_group(1)   
    return t_test(a, b)

ex_2_16()


# In[40]:


def t_test_verbose(a, sample2=None, mean=None, fn=None):
    '''Служебная функция с подробной информацией 
       результата t-теста Уэлша'''
    abar = a.mean()
    avar = a.var(ddof=1)  # несмещенное значение
    na = a.size
    adof = na - 1
    conf_int = stats.t.interval(0.95, len(a)-1, 
                                loc=sp.mean(a), scale=stats.sem(a))

    if type(a) == type(sample2):
        bbar = sample2.mean()
        bvar = sample2.var(ddof=1)
        nb = sample2.size
        bdof = nb - 1

        dof = (avar/na + bvar/nb)**2 /                 (avar**2/(na**2*adof) + bvar**2/(nb**2*bdof))
        return {'p-значение'           : 
                   fn(a, sample2, equal_var=False).pvalue,  #  выполняет t-тест Уэлша   
                'степени свободы   '   : dof,  #t_test(a, b),   
                'интервал уверенности' : conf_int,         
                'n1'           : a.count(),
                'n2'           : sample2.count(),
                'среднее x'    : a.mean(),
                'среднее y'    : sample2.mean(),
                'дисперсия x'  : a.var(),
                'дисперсия y'  : sample2.var(),
                't-статистика' : fn( a, sample2, equal_var=False ).statistic} 
    else:
        dof = (avar/na) / (avar/(na*adof))
        return {'p-значение'           : fn(a, mean).pvalue,    
                'степени свободы df'   : dof,    
                'интервал уверенности' : conf_int, 
                'n1'                   : a.count(),
                'среднее x'            : a.mean(),
                'дисперсия x'          : a.var(),
                't-статистика'         : fn(a, mean).statistic} 


# *Двухсторонние тесты*

# In[41]:


def ex_2_17():
    '''Двухсторонний t-тест'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    a = groups.get_group(0)
    b = groups.get_group(1)    
    return t_test_verbose(a, sample2=b, fn=stats.ttest_ind)  

ex_2_17()


# ### Одновыборочный t-тест

# In[42]:


def ex_2_18():
    '''Одновыборочный t-тест'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    b = groups.get_group(1) 
    return t_test_verbose(b, mean=90, fn=stats.ttest_1samp)  

ex_2_18()


# ### Многократные выборки

# In[43]:


def ex_2_19():
    '''Построение графика синтетических времен задержки 
       путем извлечения бутстраповских выборок'''
    groups = load_data('new-site.tsv').groupby('site')['dwell-time']
    b = groups.get_group(1) 
    xs = [b.sample(len(b), replace=True).mean() for _ in range(1000)] 
    pd.Series(xs).hist(bins=20)
    plt.xlabel('Бутстрапированные средние времени задержки, сек.')
    plt.ylabel('Частота') 
    #saveplot('ex_2_19.png') 
    plt.show() 
    
ex_2_19()


# ### Проверка многочисленных вариантов дизайна веб-сайта

# *Вычисление выборочных средних*

# In[44]:


def ex_2_20():
    '''Выборочные средние значения 
       20 разных вариантов дизайна веб-сайта'''
    df = load_data('multiple-sites.tsv')
    return df.groupby('site').aggregate(sp.mean)

ex_2_20()


# In[65]:


import itertools

def ex_2_21():
    '''Проверка вариантов дизайна веб-сайта на основе t-тест 
       по принципу "каждый с каждым"'''
    groups = load_data('multiple-sites.tsv').groupby('site')
    alpha = 0.05
    
    pairs = [list(x)   # найти сочетания из n по k
             for x in itertools.combinations(range(len(groups)), 2)]  

    for pair in pairs:
        gr, gr2 = groups.get_group( pair[0] ), groups.get_group( pair[1] )
        site_a, site_b = pair[0], pair[1]
        a, b = gr['dwell-time'], gr2['dwell-time']  
        p_val = stats.ttest_ind(a, b, equal_var = False).pvalue  
        if p_val < alpha: 
            print('Варианты %i и %i значимо различаются: %f' 
                  % (site_a, site_b, p_val))
    
ex_2_21()


# In[64]:


def ex_2_22():
    '''Проверка вариантов дизайна веб-сайта на основе t-теста 
       против изначального (0)'''
    groups = load_data('multiple-sites.tsv').groupby('site')
    alpha = 0.05 
    baseline = groups.get_group(0)['dwell-time']
    for site_a in range(1, len(groups)):
        a = groups.get_group( site_a )['dwell-time']
        p_val = stats.ttest_ind(a, baseline, equal_var = False).pvalue 
        if p_val < alpha: 
            print('Вариант %i значимо отличается от изначального: %f' 
                  % (site_a, p_val))
    
ex_2_22()


# ### Поправка Бонферрони

# In[63]:


def ex_2_23():
    '''Проверка вариантов дизайна веб-сайта на основе t-теста 
       против изначального (0) с поправкой Бонферрони'''
    groups = load_data('multiple-sites.tsv').groupby('site')
    alpha = 0.05 / len(groups)
    baseline = groups.get_group(0)['dwell-time']
    for site_a in range(1, len(groups)):
        a = groups.get_group(site_a)['dwell-time']
        p_val = stats.ttest_ind(a, baseline, equal_var = False).pvalue 
        if p_val < alpha: 
            print('Вариант %i веб-сайта отличается от изначального: %f' 
                  % (site_a, p_val))
    
ex_2_23()


# ### F-распределение

# In[48]:


def ex_2_Fisher():
    '''Визуализация разных F-распределений на графике'''
    mu = 0
    d1_values, d2_values = [4, 9, 49], [95, 90, 50]
    linestyles = ['-', '--', ':', '-.']
    x = sp.linspace(0, 5, 101)[1:] 
    ax = None
    for (d1, d2, ls) in zip(d1_values, d2_values, linestyles):
        dist = stats.f(d1, d2, mu)
        df  = pd.DataFrame( {0:x, 1:dist.pdf(x)} )   
        ax = df.plot(0, 1, ls=ls, 
                     label=r'$d_1=%i,\ d_2=%i$' % (d1,d2), ax=ax)

    plt.xlabel('$x$\nF-статистика')
    plt.ylabel('Плотность вероятности \n$p(x|d_1, d_2)$')
    plt.grid(True)
    #saveplot('ex_2_Fisher.png') 
    plt.show()

ex_2_Fisher()


# ### F-статистика

# In[49]:


def ssdev(xs):
    '''Сумма квадратов отклонений между 
       каждым элементом и средним по выборке'''
    x_hat = xs.mean() 
    square_deviation = lambda x : (x - x_hat) ** 2 
    return sum( map(square_deviation, xs) ) 


# ### F-тест

# In[54]:


def f_test(df, groups):
    '''Проверка на основе F-теста'''
    m, n = len(groups), sum(groups.count())
    df1, df2 = m - 1, n - m 
    ssw = sum( groups.apply(lambda g: ssdev(g)) )  # внутригрупповая сумма квадратов отклонений
    sst = ssdev( df['dwell-time'] )                # полная сумма квадратов по всему набору
    ssb = sst - ssw                                # межгрупповая сумма квадратов отклонений
    msb = ssb / df1                                # усредненная межгрупповая
    msw = ssw / df2                                # усредненная внутригрупповая
    f_stat = msb / msw
    return stats.f.sf(f_stat, df1, df2)            # функция выживания (иногда точнее, чем 1-cdf )


# In[55]:


def ex_2_24():
    '''Проверка вариантов дизайна веб-сайта 
       на основе F-теста'''
    df = load_data('multiple-sites.tsv')
    groups = df.groupby('site')['dwell-time']
    return f_test(df, groups)

ex_2_24()


# In[56]:


def ex_2_25():
    '''Визуализация распределений всех вариантов 
       дизайна веб-сайта на одной коробчатой диаграмме'''
    df = load_data('multiple-sites.tsv')
    df.boxplot(by='site', showmeans=True)
    plt.xlabel('Номер дизайна веб-сайта')
    plt.ylabel('Время задержки, сек.')
    plt.title('')
    plt.suptitle('')
    plt.gca().xaxis.grid(False)
    #saveplot('ex_2_25.png') 
    plt.show()

ex_2_25()


# In[57]:


def ex_2_26():
    '''Проверка вариантов 0 и 10 дизайна веб-сайта 
       на основе F-теста'''
    df = load_data('multiple-sites.tsv')
    groups   = df.groupby('site')['dwell-time']
    site_0   = groups.get_group(0) 
    site_10  = groups.get_group(10)
    _, p_val = stats.ttest_ind(site_0, site_10, equal_var=False)
    return p_val

ex_2_26()


# In[58]:


def ex_2_27():
    '''Проверка вариантов 0 и 6 дизайна веб-сайта 
       на основе F-теста'''
    df = load_data('multiple-sites.tsv')
    groups   = df.groupby('site')['dwell-time']
    site_0   = groups.get_group(0) 
    site_6   = groups.get_group(6)
    _, p_val = stats.ttest_ind(site_0, site_6, equal_var=False)
    return p_val
    
ex_2_27()


# ### Размер эффекта

# *Интервальный индекс d Коэна*

# In[59]:


def pooled_standard_deviation(a, b):
    '''Объединенное стандартное отклонение 
       (не объединенная стандартная ошибка)'''
    return sp.sqrt( standard_deviation(a) ** 2 +
                    standard_deviation(b) ** 2)


# In[60]:


def ex_2_28():
    '''Вычисление интервального индекса d Коэна 
       для варианта дизайна веб-сайта под номером 6'''
    df = load_data('multiple-sites.tsv')
    groups = df.groupby('site')['dwell-time']
    a      = groups.get_group(0)
    b      = groups.get_group(6)
    return (b.mean() - a.mean()) / pooled_standard_deviation(a, b)

ex_2_28()


# <a href="#home"><i class="fa fa-home fa-fw"></i></a><br><br>
