#create dataframe with cases and index on 4th May

q8 =q3_inner.copy()

q8['Date']= pd.to_datetime(q8['Date'])

match_date = pd.to_datetime("04MAY2020:00:00:00", format='%d%b%Y:%H:%M:%S')

q8_single = q8[(q8['Date'] == match_date)]

#drop Death column

q8_single=q8_single.drop(['Deaths'], axis=1)


#drop Death column

q8_single=q8_single.drop(['Deaths'], axis=1)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#######################################
sns.regplot('Cases', 'Index', data=q8_single, scatter=True, fit_reg=True)
plt.xlim(1e0, 1e8)
plt.xscale('log')
plt.show()





q8_single.plot.scatter(x='Cases',y='Index')
plt.xscale('log')
plt.xlim(1e0, 1e8)
plt.show()
