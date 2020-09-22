

#Q6 -

#create dataframe contains death between 7th March to 10th May

#make a copy of dataframe
q6 = q2_deaths3.copy()

#convert date into 'date' format
q6['Date']= pd.to_datetime(q6['Date'])

#create dataframe contains death between 7thMarch to 10th May

start, end = pd.to_datetime(["07MAR2020:00:00:00", "10MAY2020:00:00:00"], 
                            format='%d%b%Y:%H:%M:%S')


q6_deaths = q6[
    (q6['Date'] >= start) & 
    (q6['Date'] <= end)]


q6_deaths.head()

# filter on UK records only
q6_deaths=q6_deaths.query('CountryName =="United Kingdom"')

#convert datetime back to string
from datetime import datetime

q6_deaths['Date2']=q6_deaths['Date'].apply(lambda x: x.strftime('%Y%m%d'))



import matplotlib.pyplot as plt

fig, ax = plt.subplots() 
x=q6_deaths['Date2']
y=q6_deaths['Deaths']
plt.plot(x,y)

ax.set_xticks(x[::7])
ax.set_xticklabels(x[::2], rotation=45)

plt.show()

plt.title('Number of confirmed deaths in UK between 7thMarch to 10stMay')

plt.ylabel("Number of confirmed deaths") 
plt.xlabel('date')
