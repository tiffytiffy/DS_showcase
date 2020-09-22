
#add no of deaths into dataframe
q3_inner['Deaths']= q2_deaths3['Deaths']

q3_inner.dtypes

#convert date into 'date' format
q3_inner['Date']= pd.to_datetime(q3_inner['Date'])

#get confrimed cases and deaths on 10th May 2020
q5_inner=q3_inner[(q3_inner['Date'] == '2020-05-10') ].copy()

#drop index from dataframe
q5_inner=q5_inner.drop("Index", axis=1)


# Add a new column named 'flag' to indicate countries
 
q5_inner['flag'] = [1 if x =='USA' else 0 for x in q5_inner['CountryCode']] 

#import counter method to see if 'flag' variable is working
from collections import Counter
print(Counter(q5_inner['flag']))

#sort dataframe by flag
q5_inner.sort_values(by=['flag'], ascending=False, inplace=True)


#USA
q5_USA = q5_inner.iloc[[0],[0,3,4]].copy()


#Rest of world cases
q5_row = pd.DataFrame(data = {
    'CountryName' : ['Rest of World'],
    'Cases' : [q5_inner['Cases'][1:].sum()],
     'Deaths' : [q5_inner['Deaths'][1:].sum()]
})


#combining dataframe together
q5_all = pd.concat([q5_USA, q5_row])

#add population figures into dataframe

pop=[328000000,7800000000]

q5_all['pop']=pop



fig, (ax1,ax2,ax3)  = plt.subplots(1, 3)

#overall title for the charts
fig.suptitle('US COVID-19 vs Rest of World',  fontsize=18)

#title for plot1
ax1.set_title('Population')
values = q5_all['pop']
labels = q5_all['CountryName'] 

#calculate percentage and display absolute value on chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

ax1.pie(values, labels=labels, autopct=make_autopct(values))

#title for plot2
ax2.set_title('Cases')
values = q5_all['Cases']
labels = q5_all['CountryName'] 

#calculate percentage and display absolute value on chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

ax2.pie(values, labels=labels, autopct=make_autopct(values))

#title for plot3
ax3.set_title('Deaths')
values = q5_all['Deaths']
labels = q5_all['CountryName'] 

#calculate percentage and display absolute value on chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

ax3.pie(values, labels=labels, autopct=make_autopct(values))

plt.show()

