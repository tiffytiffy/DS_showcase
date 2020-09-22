#Import excel file
import pandas as pd

pd.set_option('display.max_columns',None)
workbook_url = 'C:/Users/tstacks/OxCGRT_summary.xlsx'
q2_cases = pd.read_excel(workbook_url, sheet_name='confirmedcases', nrows=166,ignore_index=True)
q2_deaths = pd.read_excel(workbook_url, sheet_name='confirmeddeaths', nrows=166,ignore_index=True)
q2_index = pd.read_excel(workbook_url, sheet_name='stringencyindex_legacy', nrows=166,ignore_index=True)


q2_cases.head()
q2_deaths.head()
q2_index.head()


#check all variable types 
q2_cases.dtypes
q2_deaths.dtypes
q2_index.dtypes

#descriptive stats to check all columns
q2_cases.describe(include='all')
q2_deaths.describe(include='all')
q2_index.describe(include='all')




##modify structure of dataframe for plotting
#select the columns to use for a multi-level index.
q2_cases = q2_cases.set_index(['CountryName','CountryCode'])
q2_deaths = q2_deaths.set_index(['CountryName','CountryCode'])
q2_index = q2_index.set_index(['CountryName','CountryCode'])

#reshape the data
q2_cases2 = q2_cases.stack(dropna=False).reset_index()
q2_deaths2 = q2_deaths.stack(dropna=False).reset_index()
q2_index2 = q2_index.stack(dropna=False).reset_index()

#rename column names
q2_cases3 = q2_cases2.set_axis(['CountryName', 'CountryCode', 'Date', 'Cases'], axis=1, inplace=False)
q2_deaths3 = q2_deaths2.set_axis(['CountryName', 'CountryCode', 'Date', 'Deaths'], axis=1, inplace=False)
q2_index3 = q2_index2.set_axis(['CountryName', 'CountryCode', 'Date', 'Index'], axis=1, inplace=False)


#check if there are any missing in dataset
q2_cases3.isnull().sum().sum() #no missing
q2_deaths3.isnull().sum().sum() #no missing 
q2_index3.isnull().sum().sum()  #index got 17 missing in the dataset


#find out which columns have missing data in index dataframe - there\are missing data from 1stMay2020 till 10thMay2020
#miss=q2_index3.isna().any()[lambda x: x]


#which counrty is missing?
row_miss = q2_index3[q2_index3.isnull().any(axis=1)]
#Taiwan, Algeria and Mali have missing data


#replace missing with forward filling i.e. copying from previous Index 
q2_index3.fillna(method='ffill', inplace=True)

#double check if there's anymore missing data
q2_index3.isnull().sum().sum() 

#now there's no missing in any of these countries
taiwan = q2_index3[q2_index3.CountryName =='Taiwan'] 
algeria =q2_index3[q2_index3.CountryName =='Algeria'] 
mali = q2_index3[q2_index3.CountryName =='Mali'] 

