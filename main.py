import pandas
import numpy as np

# In[2]:


d = pandas.read_excel('2018.xlsx', engine='openpyxl')


# In[3]:


temp = []
buf = []
c = 0
for i in d['T']:
  buf.append(i)
  c += 1
  if c == 8:
    temp.append(buf)
    buf = []
    c = 0
	
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

sr_temp = []


for i in temp:
    sr_temp.append(np.mean(i))
	
arr_sr_t = []
start = 6
for i in range(len(sr_temp) - start):
    arr_sr_t.append(sr_temp[i:i + start])
list_st_t = sr_temp[start:]


tg_a_p = []
for i in arr_sr_t:
    buf = []
    for c in range(1,len(i)):
        buf.append(np.arctan((i[c] - i[0])/c))
    tg_a_p.append(buf)
	
	
model_atang = KMeans(n_clusters = 8)
model_atang.fit(tg_a_p)


# In[60]:


ppred = model_atang.predict(tg_a_p)

colors = ['#455667', '#677889', '#454332', '#655443', '#122334','#233445','#344556','#455667']

for i in range(len(tg_a_p)):
  val = sr_temp[i]
  color = ppred[i]
  
  plt.scatter(i, val, c = colors[color])
plt.plot(sr_temp)
plt.show()