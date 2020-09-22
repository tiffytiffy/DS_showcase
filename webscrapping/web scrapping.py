# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:10:09 2020

@author: Tstacks
"""
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import requests
import bs4
import pandas as pd
import csv


def soup(nextpage):
    url ="https://www.autotrader.co.uk/car-search?advertClassification=standard&postcode=SL6%201EH&onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&is-quick-search=TRUE&page={}".format(nextpage)
    page = requests.get(url)
    soup = bs4.BeautifulSoup(page.text, "html.parser")
    info = soup.find_all('div', {'class':'information-container'})
    titles = [car.find('h2', {'class':'listing-title'}).find('a',{'class':'js-click-handler listing-fpa-link tracking-standard-link'}).text for car in info]

#extract car price
    price = soup.find_all('div',attrs={"class" : "vehicle-price"})
    car_price = [p.text for p in price]
    car_price2 = [elem.replace("£","") for elem in car_price]
    
## your other elements in here e.g. attention
    listings = [car.find('p').text for car in info]
#car reg
    reg= [car.find('ul').findChildren('li',recursive=True) [0] for car in info]
#registration - find first row 'li' within the unordered line 'ul'
    print(reg) # unforunatfe all reg number scrapped are within a HTML tag, so need to remove them
#first convert 'reg' into string then use beautifulsoup get text method to strip out all the HTML tag
    str_cells = str(reg)
    reg2 = BeautifulSoup(str_cells, "lxml").get_text()
    print(reg2) #data looks clearner but they are all in one string so need to split them into separate list
#using split by comma method to separate the strong into separate element
    reg3 = reg2.split(",")
#extract just year from car reg info
    reg4 = [v[1:5] for v in reg3]
#bodytype
    bodytype= [car.find('ul').findChildren('li',recursive=True) [1] for car in info]
    str_cells = str(bodytype)
    bodytype2 = BeautifulSoup(str_cells, "lxml").get_text().strip("[]")
    bodytype3=bodytype2.split(", ")

#mileage
    mileage= [car.find('ul').findChildren('li',recursive=True) [2] for car in info]
    str_cells = str(mileage)
    mileage2 = BeautifulSoup(str_cells, "lxml").get_text().strip("[]")
    mileage3=mileage2.split(" miles") 
#get rid of last element
    mileage4=mileage3[:-1]
    mileage5 = [elem.replace(", ","") for elem in mileage4]

#cylinders
    cylinder= [car.find('ul').findChildren('li',recursive=True) [3] for car in info]
    str_cells = str(cylinder)
    cylinder2 = BeautifulSoup(str_cells, "lxml").get_text()
    cylinder3=cylinder2.split(",")   
#replacing the square brackets after splitting
    cylinder4 = [elem.replace("[","").replace("]","") for elem in cylinder3]
#get rid of wrong elements 

#transmission
    trans= [car.find('ul').findChildren('li',recursive=True) [5] for car in info]
    str_cells = str(trans)
    trans2 = BeautifulSoup(str_cells, "lxml").get_text()
    trans3=trans2.split(",")   
#replacing the square brackets after splitting
    trans4 = [elem.replace("[","").replace("]","") for elem in trans3]

#fuel
    fuel= [car.find('ul').findChildren('li',recursive=True) [6] for car in info]
    str_cells = str(fuel)
    fuel2 = BeautifulSoup(str_cells, "lxml").get_text()
    fuel3=fuel2.split(",")   
#replacing the square brackets after splitting
    fuel4 = [elem.replace("[","").replace("]","") for elem in fuel3]


#owners
    temp_own=[car.find('ul').findChildren('li',recursive=True) for car in info]
#since not all ads have owner info and they are allocated in line 8 if available so need to use len function to check if data exist, else need to place a missing value
    owners = [e[7] if len(e) == 8 else '' for e in temp_own]
    str_cells = str(owners)
    owners2 = BeautifulSoup(str_cells, "lxml").get_text()
    owners3=owners2.split(",")   
#replacing the square brackets after splitting
    owners4 = [elem.replace("[","").replace("]","") for elem in owners3]

#location
    loc =  [car.find('span',{'class':'seller-town'}) for car in info]
    str_cells = str(loc)
    loc2 = BeautifulSoup(str_cells, "lxml").get_text()
    loc3=loc2.split(",")   
#replacing the square brackets after splitting
    loc4 = [elem.replace("[","").replace("]","") for elem in loc3]


#no of seats
    seats=soup.find_all('ul',attrs={"class" : "action-links"})
    path = [car.find('a',{'class':'tracking-motoring-products-link action-anchor'})['href'] for car in seats]
    df = pd.DataFrame({'href':path})
    df['blurb'] = df['href'].str.split('numSeats=').str[1]
    df['seats'] = df['blurb'].str[:1]
    seats_no = df['seats'].tolist()



    return titles, listings, car_price2, reg4, mileage5, fuel4, bodytype3, loc4, trans4, seats_no, cylinder4,owners4;

# Run the function to extract the elements for selected pages
soup(1)
soup(2)
soup(3)
soup(4)
soup(5)
soup(6)
soup(7)
soup(8)
soup(9)
soup(10)

# Convert tuple into lists for the elements extracted from each page
p1 = list(soup(1))
p2 = list(soup(2))
p3 = list(soup(3))
p4 = list(soup(4))
p5 = list(soup(5))
p6 = list(soup(6))
p7 = list(soup(7))
p8 = list(soup(8))
p9 = list(soup(9))
p10 =list(soup(10))


#transpose list
p1_trans=list(map(list,zip(*p1)))
p2_trans=list(map(list,zip(*p2)))
p3_trans=list(map(list,zip(*p3)))
p4_trans=list(map(list,zip(*p4)))
p5_trans=list(map(list,zip(*p5)))
p6_trans=list(map(list,zip(*p6)))
p7_trans=list(map(list,zip(*p7)))
p8_trans=list(map(list,zip(*p8)))
p9_trans=list(map(list,zip(*p9)))
p10_trans=list(map(list,zip(*p10)))

#convert list into dataframe
df_p1 = pd.DataFrame(p1_trans) 
df_p2 = pd.DataFrame(p2_trans) 
df_p3 = pd.DataFrame(p3_trans) 
df_p4 = pd.DataFrame(p4_trans) 
df_p5 = pd.DataFrame(p5_trans) 
df_p6 = pd.DataFrame(p6_trans) 
df_p7 = pd.DataFrame(p7_trans) 
df_p8 = pd.DataFrame(p8_trans) 
df_p9 = pd.DataFrame(p9_trans) 
df_p10 = pd.DataFrame(p10_trans) 

#combine 10 pages together
df_all = pd.concat([df_p1, df_p2, df_p3, df_p4, df_p5, df_p6, df_p7, df_p8, df_p9, df_p10], ignore_index=True)


#give column names 
df_all.columns =['Title', 'Listing', 'Price', 'Reg_year', 'Mileage' , 'Fuel', 'Body_type', 'Location', 'Transmission','No_of_seats',
                               'Cylinders', 'No_of_owners']



df_all.to_csv('car_info.csv')





with open('car_info.csv','wt') as out:
    csv_out=csv.writer(out)
    #csv_out.writerow(['Title', 'Listing', 'Price','Reg_year', 'Mileage', 'Fuel','body_type', 'Location', 'Transmission','No_of_seats', 'Cylinders', 'No_of_owners'])
    for j in df_car:
        csv_out.writerow(j)
        
        
