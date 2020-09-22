
#make a copy of dataframe
q7 = q2_cases3.copy()

#convert date into 'date' format
q7['Date']= pd.to_datetime(q7['Date'])

#create dataframe contains death between 1st March to 1st May

start, end = pd.to_datetime(["01MAR2020:00:00:00", "01MAY2020:00:00:00"], 
                            format='%d%b%Y:%H:%M:%S')

q7_cases = q7[
    (q7['Date'] >= start) & 
    (q7['Date'] <= end)]


q7_cases.head()
q7_cases.dtypes

#convert datetime back to string
from datetime import datetime

q7_cases['Date2']= q7_cases['Date'].apply(lambda x: x.strftime('%Y%m%d'))

q7_cases.head()

#load in latitude & longitude for countries
# read in csv file


ctry_loc= pd.read_csv('lat_long2.csv', delimiter=",", encoding="ISO-8859-1")

q7_cases2=pd.merge(q7_cases, ctry_loc, left_on='CountryName', right_on='name')

#limit data to UK, Spain, Italy, France, and USA

q7_cases2=q7_cases2.drop("name", axis=1)
q7_countries=q7_cases2.query('CountryName in ["Spain","United States","France","United Kingdom","Italy"]' )

q7_countries.dtypes

import plotly.express as px

# default radius of the biggest bubble is 20px, you can change it by size_max
fig = px.scatter_geo(q7_countries, lat='latitude', lon='longitude' , color="CountryName", hover_name="CountryName", 
                     size="Cases", size_max=40, animation_frame="Date2", projection="natural earth", opacity=0.6) 
fig.write_html('q7_cases.html', auto_open=True)

#pip install -c plotly plotly-orca==1.2.1 psutil requests

#save as static jpeg or pdf
fig.write_image("q7_cases.jpeg", width=600, height=350, scale=2)
fig.write_image("q7_cases.pdf", width=600, height=350, scale=10)
