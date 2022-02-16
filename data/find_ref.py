from bs4 import BeautifulSoup

soup = BeautifulSoup(open('W15-1509.tei.xml'))

l =  soup.find('text').find('div').find('p').contents
for i in range(len(l)):
    print(l[i])