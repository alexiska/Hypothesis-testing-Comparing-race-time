#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def site(url,pages):
     """
     An argument is passed into the function which consist of the desired url. 
     A for loop will loop through each page and extract following 
     data for each column and store them in respectively list:
     participant name, posision, born, club, start position and time.
     The lists are then returned.
     
     INPUT: the url page and number of pages to webscrape
     OUTPUT: Five list of values position, name, born, club and time
     """
    
    name = []
    pos = []
    born = []
    club = []
    start_pos = []
    time = []
    
    for page in range(1,pages):
        req = Request(url.format(page))
        webpage = urlopen(req).read()

        page_soup = soup(webpage, 'html.parser')
        page_soup.findAll("tr", {"class": "list-highlight"})
        page_soup.findAll("td", {"valign": "top"})
        containers = page_soup.find_all('td')

        for column in range(0,7):

            #column for position
            if column == 0:
                for i in range(0,len(containers),6):
                    pos.append(containers[i].text)
            #column for participant 
            if column == 1:  
                for i in range(1,len(containers),6):
                    name.append(containers[i].text.strip().replace('» ',''))
            #column for birth year 
            if column == 2:  
                for i in range(2,len(containers),6):
                    born.append(containers[i].text.strip())
            #column for birth year 
            if column == 3:  
                for i in range(3,len(containers),6):
                     club.append(containers[i].text.strip())
            #column for time
            if column == 4:  
                for i in range(5,len(containers),6):
                     time.append(containers[i].text.strip())

    return(pos,name,born,club,time)

# Scrape men 10 km 2012 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/815?layout=racetimer&page={}&rc_id=4935#top', 5)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\men_10km_2012.csv', index=False)

# Scrape women 10 km 2012 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/815?commit=Visa+resultat+%3E%3E&layout=racetimer&page={}&per_page=250&race_id=815&rc_id=5061#top', 3)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\women_10km_2012.csv', index=False)

# Scrape men 5 km 2012 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/815?commit=Visa+resultat+%3E%3E&layout=racetimer&page={}&per_page=250&race_id=815&rc_id=5062#top', 4)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\men_5km_2012.csv', index=False)

# Scrape women 5 km 2012 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/815?commit=Visa+resultat+%3E%3E&layout=racetimer&page={}&per_page=250&race_id=815&rc_id=5063#top', 6)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\women_5km_2012.csv', index=False)

# Scrape men 10 km 2011 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/634?layout=racetimer&page={}&rc_id=3507#top', 4)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\men_10km_2011.csv', index=False)

# Scrape women 10 km 2011 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/634?layout=racetimer&page={}&rc_id=3553#top', 3)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\women_10km_2011.csv', index=False)

# Scrape men 5 km 2011 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/634?layout=racetimer&page={}&rc_id=3554#top', 4)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\men_5km_2011.csv', index=False)

# Scrape women 5 km 2011 race
pos, name, born, club, time = site('https://www.racetimer.se/sv/race/resultlist/634?layout=racetimer&page={}&rc_id=3555#top', 6)
df=pd.DataFrame({'Pos':pos,'Name':name,'Born':born,'Club':club,'Time':time})
df.to_csv(r'C:\Users\orkab\Desktop\python övning\Blodomploppet\women_5km_2011.csv', index=False)


# In[ ]:




