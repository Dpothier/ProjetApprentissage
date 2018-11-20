# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from scrapy import FormRequest
import re

class MaSpiderSpider(scrapy.Spider):
    visited_recall_numbers = set()
    name = 'carcomplaints'
    domain = 'https://www.carcomplaints.com/'


    def __init__(self):

        self.start_urls = ['https://www.carcomplaints.com/Chrysler/Town_Country/2011/electrical/TIPM_failure_would_not_start.shtml']

    def parse(self, response):
        self.parseComplaints(response)
        #details_selector = "table > tr > td > a"
        #recalls = response.css(details_selector)
        #recalls_url = recalls.css('::attr(href)').extract()
        #for index, item in enumerate(recalls.css('::text').extract()):
        #    if item not in self.visited_recall_numbers:
        #        self.visited_recall_numbers.add(item)
        #        yield Request('{}{}'.format(self.domain, recalls_url[index]), self.parseDetails)

        #if recalls_url:
        #    yield FormRequest.from_response(response, formnumber=1, clickdata={'value': 'Next'}, callback=self.parse)



    def parseComplaints(self, response):
        pattern = re.compile('www.carcomplaints.com/(\w*)/(\w*)/(\w*)/(\w*)')
        page_url = response.url
        re_results = re.search(pattern, page_url)
        make = re_results.group(1)
        model = re_results.group(2)
        year = re_results.group(3)
        system = re_results.group(4)

        complaints_path = '#pcomments > .complain'

        for complaint in response.css(complaints_path):

            item = CarComplaintsItem()
            item['system'] = system
            item['details'] = complaint.css('div > p::text').extract_first()
            item['make'] = make
            item['model'] = model
            item['year'] = year
            yield item



class CarComplaintsItem(scrapy.Item):
    system = scrapy.Field()
    details = scrapy.Field()
    make = scrapy.Field()
    model = scrapy.Field()
    year = scrapy.Field()

