#Python program to scrape website 
#and save quotes from website
import requests
from bs4 import BeautifulSoup
import csv
 
URL = "https://www.daraz.pk/catalog/?spm=a2a0e.tm80331704.cate_5.5.77cc5aa7fPImi7&q=Smart%20Phones&from=hp_categories&src=all_channelhttps://www.amazon.com/gp/browse.html?node=6563140011"
r = requests.get(URL)

 
soup = BeautifulSoup(r.content, 'html5lib')
 
quotes=[]  # a list to store quotes
 

table = soup.find('ol', attrs = {'class':'a-carousel'})

for row in table.find_all('li',
                         attrs = {'class':'a-carousel-card ucw-widget-carousel-element'}):
    quote = {}
    quote['theme'] = row.h5.text
    quote['url'] = row.a['href']
    quote['img'] = row.img['src']
    quote['lines'] = row.img['alt']
    quote['price'] = row.img['alt']
    quotes.append(quote)
 
filename = 'amazon_smart_home.csv'
with open(filename, 'w', newline='') as f:
    w = csv.DictWriter(f,['theme','url','img','lines','price'])
    w.writeheader()
    for quote in quotes:
        w.writerow(quote)

