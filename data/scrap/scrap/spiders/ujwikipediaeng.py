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
    name = "wiki2"

    start_urls = [
        "https://en.wikipedia.org/wiki/Jagiellonian_University",
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
