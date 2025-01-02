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
    name = "uj"

    start_urls = [
        #"https://en.uj.edu.pl/en_GB",
        "https://www.uj.edu.pl/pl",
        "https://bip.uj.edu.pl/"
    ]
    vis = set()  # Użyjemy zestawu dla szybszego sprawdzania

    def parse(self, response):
        for xd in response.css("p").getall():
            txt=text(xd)
            if len(txt)>10:
                yield {
                 "text": wywalspacjeiendl(txt)
                }

        for href in response.css("a::attr(href)"):
            url = response.urljoin(href.get())
            if "uj" in url and url not in self.vis:
                self.vis.add(url)
                yield response.follow(url, callback=self.parse)
            
#nuh uh