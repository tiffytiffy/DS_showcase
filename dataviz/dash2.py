
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Creating a dash web app
app = dash.Dash()

#continent=df_all3['Continent_Name'].unique()
measure=df_all3.iloc[:,5:8].columns.values.tolist()
#policies=df_all3.iloc[:,3:5].columns.values.tolist()

fig= px.scatter_geo(df_all3, lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                    size='ConfirmedCases', size_max=40, title="Confirmed Cases for All Continents",
                    animation_frame="Date", projection="natural earth", opacity=0.4)


app.layout = html.Div([ 
    html.Label("Scope:"),

dcc.Dropdown(id="Scope", 
             options=[{'label': 'World', 'value': 'All'},
                      {'label': 'Asia', 'value': 'Asia'},
                      {'label': 'Africa', 'value': 'Africa'},
                      {'label': 'Europe', 'value': 'Europe'},
                      {'label': 'North America', 'value': 'North America'},
                      {'label': 'South America', 'value': 'South America'},
                      {'label': 'Oceania', 'value': 'Oceania'}], value='All'),
html.Br(),html.Br(),html.Label("Input Data"),

dcc.RadioItems(id="Data", 
               options=[{'label': j, 'value':j} for j in measure],
               value='ConfirmedCases'),

html.Br(),html.Br(),html.Label("Policy"),

dcc.RadioItems(id="pol", options=[{'label':'Not Selected', 'value': 'Not Selected'}, {'label': 'School Closing', 'value': 'School closing'}, 
                                  {'label': 'Stay at home requirements', 'value': 'Stay at home requirements'}], value='Not Selected'),

dcc.Graph(id = 'chart', figure={
    "layout": {"height": 700 } # px

})

])


@app.callback(
Output('chart', 'figure'),
[Input('Scope', 'value'),
Input('Data', 'value'),
Input('pol', 'value')
])


def updatechart(g,h,pol):
    if pol=="Not Selected":
        if g=="All" and h=="ConfirmedCases": 
            return fig
        elif g=="All" and h=="ConfirmedDeaths":
            return px.scatter_geo(df_all3, lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, title=""+h+" for All Continents", animation_frame="Date", projection="natural earth", opacity=0.4)
        elif g=="All" and h=="StringencyIndex":
            return px.scatter_geo(df_all3, lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, title=""+h+" for All Continents", animation_frame="Date", projection="natural earth", opacity=0.4)
        
        
        elif g=="Asia" and h=="ConfirmedCases": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='asia', title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        elif g=="Asia" and h=="ConfirmedDeaths": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h,size_max=40, scope='asia', title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        elif g=="Asia" and h=="StringencyIndex": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40,  scope='asia', title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        
        
        elif g=="Africa" and h=="ConfirmedCases" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='africa',title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        elif g=="Africa" and h=="ConfirmedDeaths" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='africa',title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        elif g=="Africa" and h=="StringencyIndex" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='africa',title=""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4)
        
        
        elif g=="Europe" and h=="ConfirmedCases": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name",hover_name="CountryName", 
                                  size=h, size_max=40, scope='europe',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="Europe" and h=="ConfirmedDeaths": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40,scope='europe', title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="Europe" and h=="StringencyIndex": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name",hover_name="CountryName", 
                                  size=h, size_max=40, scope='europe',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        
        
        elif g=="North America" and h=="ConfirmedCases" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name",hover_name="CountryName", 
                                  size="ConfirmedCases", size_max=40, scope='north america',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="North America" and h=="ConfirmedDeaths" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size="ConfirmedDeaths", size_max=40, scope='north america',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="North America" and h=="StringencyIndex": 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40,scope='north america', title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        
        
        elif g=="South America" and h=="ConfirmedCases" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='south america',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="South America" and h=="ConfirmedDeaths" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='south america',title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="South America" and h=="StringencyIndex" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, scope='south america', title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        
        # plotly hasn't got scope for oceania
        elif g=="Oceania" and h=="ConfirmedCases" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name",hover_name="CountryName", 
                                  size=h, size_max=40,title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="Oceania" and h=="ConfirmedDeaths" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name",hover_name="CountryName", 
                                  size=h, size_max=40, title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 
        elif g=="Oceania" and h=="StringencyIndex" : 
            return px.scatter_geo(df_all3.query("Continent_Name=='"+g+"'"), lat='latitude', lon='longitude' , color="Continent_Name", hover_name="CountryName", 
                                  size=h, size_max=40, title =""+h+" in "+g+"", animation_frame="Date", projection="natural earth", opacity=0.4) 



    if pol=="School closing":
        if g=="All": 
            return px.choropleth(df_all3, locations='CountryCode', color=pol, hover_name="CountryName", animation_frame="Date", 
                                 title=""+pol+" in World", color_continuous_scale=px.colors.sequential.Plasma) 

        elif g=="Asia" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='asia',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 

 
        elif g=="Africa" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='africa',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 

        elif g=="Europe" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='europe',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 
    
        elif g=="North America" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='north america',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 
    
        elif g=="South America" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='south america',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 
        elif g=="Oceania" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, hover_name="CountryName", 
                                 animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 
    
    
    if pol=="Stay at home requirements":
        if g=="All" : 
            return px.choropleth(df_all3, locations='CountryCode', color=pol, hover_name="CountryName", animation_frame="Date", 
                                 title=""+pol+" in World", color_continuous_scale=px.colors.sequential.Plasma) 


        elif g=="Asia" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='asia',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 


        elif g=="Africa" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='africa',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
  

        elif g=="Europe" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='europe',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
   
    
        elif g=="North America"  : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='north america',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
  
    
        elif g=="South America"  : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, scope='south america',
                                 hover_name="CountryName", animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 

        elif g=="Oceania" : 
            return px.choropleth(df_all3.query("Continent_Name=='"+g+"'"), locations='CountryCode', color=pol, hover_name="CountryName",
                                 animation_frame="Date", title=""+pol+" in "+g+"", color_continuous_scale=px.colors.sequential.Plasma) 
 

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
