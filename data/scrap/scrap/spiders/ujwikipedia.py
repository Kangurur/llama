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

class QuotesSpider(scrapy.Spider):
    name = "wiki"

    start_urls = [
        "https://pl.wikipedia.org/wiki/Uniwersytet_Jagiello%C5%84ski",
    ]
    
    def parse(self, response):
        for xd in response.css("p").getall():
            naw=0
            text=""
            for i in xd:
                if i=='<':
                    naw+=1
                if naw==0:
                    text+=i
                if i=='>':
                    naw-=1
            if len(text)>2:
                yield {
                       "text": text,
                }
