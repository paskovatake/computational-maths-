#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
from functools import partial 


# In[19]:


def f(x,y):
    return x + np.cos(y)


# # 1. Явный метод Эйлера

# $$\dfrac{y_{n+1}-y_n}{h} = f(x_n, y_n), \qquad x_n = 1 + h n$$
# 
# У нас $f(x, y) = x + \cos y$. Тогда:
# 
# $$y_{n+1} = y_n + (x_n + \cos y_n) h$$

# In[80]:


def y_next(x_prev, y_prev, h):
    return (y_prev + (x_prev + np.cos(y_prev))*h)

def explicit(N):
    y_arr1 = {}
    y_arr1[1] = 30
    h = 1/N
    for i in range(N):
        x_next = 1 + (i + 1)/N
        x_prev = 1 + i/N
        y_arr1[x_next] = y_next(x_prev, y_arr1[x_prev], h)
        
    lists = sorted(y_arr1.items()) 
    x1, y1 = zip(*lists)
    return tuple((x, y))


# In[75]:


x,y = explicit(100)

plt.figure(figsize = (15,10))
plt.plot(x, y, color = 'red')
plt.title('Явный метод Эйлера')
plt.show()


# # 2. Неявный метод Эйлера

# $$\dfrac{y_{n+1}-y_n}{h} = f(x_{n+1}, y_{n+1}), \qquad x_n = 1 + h n$$
# 
# Будем решать простыми итерациями:
# 
# $$y_{n+1} \approx y^{(k+1)} =  y_n + h f(x_{n+1}, y^{(k)}), \qquad y^{(0)} = y_n$$

# In[45]:


def s_iter(f, x_0, N_i):
    x = x_0
    for i in range(N_i):
        x = f(x)
    return x


def imp_y(y, f, x_n_plus_1, y_n, h):
    return (y_n + h * f(x_n_plus_1, y))


# In[76]:


def implicit(N, N_i):
    y_arr2 = {}
    y_arr2[1] = 30
    h = 1/N
    for i in range(N):
        x_next = 1 + (i + 1)/N
        x_prev = 1 + i/N
        g = partial(imp_y, f = f, x_n_plus_1 = x_next, y_n = y_arr2[x_prev], h = h)
        y_arr2[x_next] = s_iter(g, y_arr2[x_prev], N_i)


    lists = sorted(y_arr2.items()) 
    x2, y2 = zip(*lists) 
    return tuple((x2, y2))

x2, y2 = implicit(100, 100)

plt.figure(figsize = (15,10))
plt.plot(x2, y2, color = 'green')
plt.title('Неявный метод Эйлера')
plt.show()


# # 3. Метод Эйлера с пересчетом

# $$\tilde{y} = y_n + hf(x_{n}, y_{n})$$
# $$y_{n+1} = y_{n} + h\dfrac{f(x_{n}, y_{n}) + f(x_{n+1}, \tilde{y})}{2}$$

# In[77]:


def y_next3(x_prev, x_next, y_prev, h, f):
    y_tilde = y_prev + h*f(x_prev, y_prev)
    y_next = y_prev + h*(f(x_prev, y_prev) + f(x_next, y_tilde))/2
    return y_next

def recount(N):
    y_arr3 = {}
    y_arr3[1] = 30
    h = 1/N
    for i in range(N):
        x_next = 1 + (i + 1)/N
        x_prev = 1 + i/N
        y_arr3[x_next] = y_next3(x_prev, x_next, y_arr3[x_prev], h, f)

    lists = sorted(y_arr3.items())
    x3, y3 = zip(*lists)
    return tuple((x3,y3))

x3, y3 = recount(100)

plt.figure(figsize=(15,10))
plt.plot(x3, y3, color = 'blue')
plt.title('Метод Эйлера с пересчётом')
plt.show()


# ## Сравним методы при разных разбиениях сетки

# In[81]:


N = 4
N_i = 2
x1, y1 = explicit(N)
x2, y2 = implicit(N, N_i)
x3, y3 = recount(N)


plt.figure(figsize = (5,5))
plt.plot(x1, y1, color = 'red', label = 'Явный')
plt.plot(x2, y2, color = 'green', label = 'Неявный')
plt.plot(x3, y3, color = 'blue', label = 'С пересчетом')
plt.legend(fontsize = 12)
plt.show()

N = 25
N_i = 5
x1, y1 = explicit(N)
x2, y2 = implicit(N, N_i)
x3, y3 = recount(N)


plt.figure(figsize = (5, 5))
plt.plot(x1, y1, color = 'red', label = 'Явный')
plt.plot(x2, y2, color = 'green', label = 'Неявный')
plt.plot(x3, y3, color = 'blue', label = 'С пересчетом')
plt.legend(fontsize = 12)
plt.show()

N = 100
N_i = 20
x1, y1 = explicit(N)
x2, y2 = implicit(N, N_i)
x3, y3 = recount(N)


plt.figure(figsize = (5,5))
plt.plot(x1, y1, color = 'red', label = 'Явный')
plt.plot(x2, y2, color = 'green', label = 'Неявный')
plt.plot(x3, y3, color = 'blue', label = 'С пересчетом')
plt.legend(fontsize = 12)
plt.show()


# In[ ]:




