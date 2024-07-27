from bs4 import BeautifulSoup as soup 
from urllib.request import urlopen as uReq 
# Find the correct url to scrape the data in Reddit 
redditUrl = ""

Client = uReq(redditUrl)
pageHTML = Client.read()
Client.close()
redditSoup = soup(pageHTML,"html.parser")

containers = redditSoup.find_all("div", {"class":"_3O0U0u"})
print(len(containers))