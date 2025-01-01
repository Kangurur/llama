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
        if i!='\n' and i!=' ':
            if f:
                text+=" "
                f=False
            text+=i
        else:
            f=True
        
    return text

class QuotesSpider(scrapy.Spider):
    name = "ksi2"

    start_urls = [
        "https://ksi.ii.uj.edu.pl/pl/",
        "https://ksi.ii.uj.edu.pl/pl/about/",
        "https://ksi.ii.uj.edu.pl/pl/events/",
        
    ]
    
    def parse(self, response):
        for xd in response.css("div.content").getall():
            txt=text(xd)
            if len(txt)>2:
                yield {
                "text": wywalspacjeiendl(txt),
                }
