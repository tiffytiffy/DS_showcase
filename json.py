
#extract all countries  from 1st Jan till 10th May

import pandas as pd
from pandas import DataFrame
import json, urllib.request



# Accessing data of all countries between required period
data = urllib.request.urlopen("https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/2020-01-01/2020-05-10").read()
output = json.loads(data)

 

pip install nested_lookup

from nested_lookup import nested_lookup


q1_dates=DataFrame(nested_lookup('date_value',output))
q1_dates.columns=['Date']

q1_cases=DataFrame(nested_lookup('confirmed',output))
q1_cases.columns=['Cases']

q1_deaths=DataFrame(nested_lookup('deaths',output))
q1_deaths.columns=['Deaths']

q1_index=DataFrame(nested_lookup('stringency_legacy',output))
q1_index.columns=['Stringency_Index']

q1_country=DataFrame(nested_lookup('country_code',output))
q1_country.columns=['CountryCode']


#join relevant together
q1_cases2=pd.concat([q1_country, q1_dates, q1_cases],axis=1, sort=False )
q1_deaths2=pd.concat([q1_country, q1_dates, q1_deaths],axis=1, sort=False )
q1_index2=pd.concat([q1_country, q1_dates, q1_index],axis=1, sort=False )

#reshape dataframe
q1_cases3 = q1_cases2.pivot_table(index='CountryCode', columns='Date', values='Cases')
q1_deaths3 = q1_deaths2.pivot_table(index='CountryCode', columns='Date', values='Deaths')
q1_index3 = q1_index2.pivot_table(index='CountryCode', columns='Date', values='Stringency_Index')


#check anything missing in dataframe
#q1_cases3.isnull().sum().sum()  #8385 missing
#q1_deaths3.isnull().sum().sum() #8385 missing 
#q1_index3.isnull().sum().sum()  #62 missing


#replace NA with zeros in dataframe
#q1_cases3.fillna(0, inplace=True)
#q1_deaths3.fillna(0, inplace=True)


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('My_OxCGRT_summary.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
q1_cases3.to_excel(writer, sheet_name='confirmedcases')
q1_deaths3.to_excel(writer, sheet_name='confirmeddeaths')
q1_index3.to_excel(writer, sheet_name='stringencyindex_legacy')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
