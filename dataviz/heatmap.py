
import pandas as pd
import seaborn as sns


#top 10 countries by number of cases
q4_top10=q4_cases.groupby('CountryName')['Cases'].sum().reset_index().nlargest(10,'Cases').rename(columns={'Cases':'total_cases'})


#create dataset with top 10 countries by number of cases
q4b = q4_cases2.query('CountryName in ["United States","Spain","Italy","Germany","China","United Kingdom","France", "Iran", "Turkey", "Russia"]' )

#check countries in dataframe
print(q4b['CountryName'].unique())



# reshape data to look like a matrix with week_num (rows), CountryName (columns), and Cases(cell values)
heat=q4b.pivot("CountryName", "week_num","Cases")


#add total num of cases into dataframe
heat=heat.join(q4_top10.set_index('CountryName'), on='CountryName')


#sort data by total num of cases
heat.sort_values(by=['total_cases'], ascending=False, inplace=True)

#remove total_cases from dataset
heat2=heat.drop('total_cases', axis=1)



# create heatmap


ax = plt.axes()
ax.set_title('Weekly confirmed cases')
sns.heatmap(heat2, ax=ax, annot=False, linewidths=2,  square=True, cmap="YlOrBr", cbar_kws = dict(use_gridspec=False,location="top"))

# Add title and axis names
plt.title('Weekly confirmed cases')
plt.xlabel('week_num')
plt.ylabel('Top 10 countries')

plt.show()

