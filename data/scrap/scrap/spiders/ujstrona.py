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
        "https://en.uj.edu.pl/en_GB",
        #"https://www.uj.edu.pl/pl",
        #"https://bip.uj.edu.pl/"
    ]
    vis = set()  # Użyjemy zestawu dla szybszego sprawdzania
    tunie = set()
    tunie.add("ruj") #stąd nie ma odwrotu :skull:
    tunie.add("usosweb") #to się osobno ogarnie

    def parse(self, response):
        ans=""
        for xd in response.css("p").getall():
            txt=text(xd)
            if len(txt)>2:
                ans+=wywalspacjeiendl(txt)+" "
        if len(ans)>3 and "sts" not in response.url:
            yield {
                "link": response.url,
                "text": ans
                }

        for href in response.css("a::attr(href)"):
            url = response.urljoin(href.get())
            #id=url[8:url.find(".")]
            if "en.uj.edu.pl" in url and url not in self.vis and "wiadomos" not in url and "journal" not in url:
                self.vis.add(url)
                yield response.follow(url, callback=self.parse)
            
#nuh uh