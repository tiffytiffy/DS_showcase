
#add no of deaths into dataframe
q3_inner['Deaths']= q2_deaths3['Deaths']

#convert date into 'date' format
q3_inner['Date']= pd.to_datetime(q3_inner['Date'])

#create new dataframe with date=4thMay
match_date = pd.to_datetime("04MAY2020:00:00:00", format='%d%b%Y:%H:%M:%S')

q9 = q3_inner[(q3_inner['Date'] == match_date)]


q9.query('Cases>1000')


import plotly.express as px

q9_fig = px.scatter(q9.query("Cases>1000"), x="Cases", y="Index", size="Deaths", color="CountryName", hover_name="CountryName", log_x=True)
q9_fig.write_html('q9_fig.html', auto_open=True)
