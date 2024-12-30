from pathlib import Path

import scrapy


class QuotesSpider(scrapy.Spider):
    name = "ksi"

    start_urls = [
        "https://ksi.ii.uj.edu.pl/pl/",
        "https://ksi.ii.uj.edu.pl/pl/about/",
        "https://ksi.ii.uj.edu.pl/pl/events/",
        
    ]
    
    def parse(self, response):
        for xd in response.css("div.content"):
            yield {
                "title": xd.css("h2::text").get(),
                "text": xd.css("p::text").getall(),
                "list": xd.css("li::text").getall(),
            }
