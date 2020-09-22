
#save the token in "mapboxToken" file 
token='pk.eyJ1IjoidGlmZnl0aWZmeSIsImEiOiJja2NmMHExbWUwZGpjMnpubnMyYTF2cTl4In0.0sjxEN8QIURcxEo2H9L3mA'
f = open("mapbox.token", "w")
f.write(token)
f.close()

import plotly.graph_objects as go
token= open("mapbox.token").read()

fig = go.Figure(go.Scattermapbox(mode = "markers+text+lines",
lat = [51.5074, 51.1279, 50.9513,49.0097,41.2599], lon = [0.1278, 1.3134, 1.8587,2.5479,28.7427],
marker = {'size': 15, 'symbol': ["car", "ferry", "car","airport","airport"]},
text = ["London", "Dover", "Calais","Charles de Gaulle Airport","Istanbul Airport"],
textposition = "bottom right"))

fig.update_layout(mapbox = {'accesstoken': token,
'center': {'lat': 48, 'lon': 12},
'style': "outdoors", 'zoom': 4})
fig.write_html('q6_mapbox.html', auto_open=True)
