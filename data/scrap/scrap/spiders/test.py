from pathlib import Path

import scrapy

def text(xd):
    naw=0
    text=""
    for i in xd:
        if i=='<':
            naw+=1
        if naw==0:
            text+=i
        if i=='>':
            naw-=1
    return text

def wywalspacjeiendl(xd):
    text=""
    f=False
    for i in xd:
        if i!='\n' and i!='\t' and i!='\r' and i!=' ':
            if f:
                text+=" "
                f=False
            text+=i
        else:
            f=True
        
    return text

class QuotesSpider(scrapy.Spider):
    name = "test"

    start_urls = [
        #"https://en.uj.edu.pl/en_GB",
        "https://www.usosweb.uj.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&prz_kod=WFz.KPSC-8574",
        #"https://bip.uj.edu.pl/"
    ]
    # def start_requests(self):
    #     urls = [
    #         "https://www.usosweb.uj.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&prz_kod=WFz.KPSC-8574",
    #     ]
    #     for url in urls:
    #         xd=scrapy.Request(url=url,meta = {'dont_redirect': True}, callback=self.parse)
    #         yield xd
    vis = set()  # Użyjemy zestawu dla szybszego sprawdzania

    def parse(self, response):
        ans=""
        for xd in response.css("p").getall():
            txt=text(xd)
            if len(txt)>2:
                ans+=wywalspacjeiendl(txt)+" "
        yield {
            "link": response.url,
            "text": ans
            }

        # for href in response.css("a::attr(href)"):
        #     url = response.urljoin(href.get())
        #     if "uj" in url and url not in self.vis:
        #         self.vis.add(url)
        #         yield response.follow(url, callback=self.parse)
            
#nuh uh