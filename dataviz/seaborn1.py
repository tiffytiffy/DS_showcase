
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#create new dataframe with cases, index and the 6 selected countries

q3_inner = pd.merge(q2_cases3, q2_index3, on=['CountryName','CountryCode','Date'])
q3_countries=q3_inner.query('CountryName in ["China","South Korea","United States","France","United Kingdom","Italy"]' )

f, ax = plt.subplots(figsize=(7, 7))
ax.set(xscale="log", yscale="linear")
ax.set_title('Comparison of stringency of COVID-19 response in six countries')


sns.lineplot(x='Cases', y='Index', hue='CountryName', 
             data=q3_countries, 
             hue_order=['China','South Korea', 'United States', 'France','United Kingdom','Italy'])
plt.ylabel("Stringency Index") 
plt.xlabel('Reported number of COVID-19 cases')



