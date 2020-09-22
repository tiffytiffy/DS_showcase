
#load in latitude & longitude for countries
# read in csv file
ctry_loc= pd.read_csv('lat_long3.csv', delimiter=",", encoding="ISO-8859-1")

#join latitude and longitude to dataframe

df_all3=pd.merge(df_all2, ctry_loc[['latitude','longitude','name']], left_on='CountryName', right_on='name', how='left')

df_all3['CountryCode'].nunique() #173

#check to ensure no missing value
df_all3.isna().sum() 

df_all3.isnull().sum()


import plotly.express as px

# default radius of the biggest bubble is 20px, you can change it by size_max
fig = px.scatter_geo(df_all3, lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                     size="ConfirmedCases", size_max=40, animation_frame="Date", projection="natural earth", opacity=0.6) 
fig.write_html('q3_cases_map.html', auto_open=True)


df_all3['CountryCode'].nunique() #173
