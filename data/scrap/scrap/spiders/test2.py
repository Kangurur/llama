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
    name = "test2"

    start_urls = [
        "https://quotes.toscrape.com/tag/humor/"
    ]
    it=0
    def parse(self, response):
        for xd in response.css("a::attr(href)").getall():
                yield {
                    "text": xd
                }
        
                
        for href in response.css("a::attr(href)"):
                yield response.follow(href, callback=self.parse)

            
#nuh uh