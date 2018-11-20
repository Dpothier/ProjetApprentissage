# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from scrapy import FormRequest
import re

class MaSpiderSpider(scrapy.Spider):
    visited_recall_numbers = set()
    name = 'carcomplaints'
    domain = 'https://www.carcomplaints.com'

    rejectedComplaint = 0
    acceptedComplaint = 0


    def __init__(self):

        self.start_urls = ['https://www.carcomplaints.com']

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, callback=self.parse_main)

    def parse_main(self, response):
        model_urls = response.css('#mainmakes > ul > li > a::attr(href)').extract()

        for model in model_urls:
            yield Request(self.domain + model, callback=self.parse_make)

    def parse_make(self, response):
        model_urls = response.css('.browseby-content > ul > li > a::attr(href)').extract()

        for model in model_urls:
            yield Request(self.domain + model, callback=self.parse_model)


    def parse_model(self, response):
        years_url  = response.css('ul.timeline > li > a::attr(href)').extract()

        for year in years_url:
            yield Request(self.domain + year, callback=self.parse_year)


    def parse_year(self, response):
        page_url = response.url

        systems_path = '#graph > ul > li > a'

        for system in response.css(systems_path):
            system_url = system.css('::attr(href)').extract_first()
            yield Request(page_url + system_url, callback=self.parse_system)


    def parse_system(self, response):
        page_url = response.url

        problem_path = '#graph > ul > li > a'

        for system in response.css(problem_path):
            problem_url = system.css('::attr(href)').extract_first()
            problem_name = system.css('::text').extract_first()
            if not re.search('NHTSA', problem_name):
                yield Request(page_url + problem_url, callback=self.parse_problem)


    def parse_problem(self, response):
        pattern = re.compile('www.carcomplaints.com/([\w-]*)/([\w-]*)/([\w-]*)/([\w-]*)/([\w-]*).')
        page_url = response.url
        re_results = re.search(pattern, page_url)
        make = re_results.group(1)
        model = re_results.group(2)
        year = re_results.group(3)
        system = re_results.group(4)
        problem = re_results.group(5)

        complaints_path = '#pcomments > .complaint'

        for complaint in response.css(complaints_path):
            detail = self.get_details(complaint)
            if detail:
                item = CarComplaintsItem()
                item['system'] = system
                item['details'] = detail
                item['make'] = make
                item['model'] = model
                item['year'] = year
                item['problem'] = problem
                self.acceptedComplaint += 1
                yield item
            else:
                self.rejectedComplaint += 1

        next_complaints = response.css('#pcomments > h3.heading.primary.switchpage.bottom')
        if next_complaints:
            next_complaints_url = next_complaints.css('a::attr(href)').extract_first()
            yield Request(self.domain + next_complaints_url, callback=self.parse_problem)

    def get_details(self, complaint):
        details = complaint.css('div.comments > div:not(.ad) > p::text').extract_first()
        if not details:
            details = complaint.css('div.comments::text').extract_first()

        if details:
            details = details.strip()

        return details

    def close(self, reason):
        print("{}/{} complaints were rejected because the spider couldn't get their details".format(
            self.rejectedComplaint, self.acceptedComplaint + self.rejectedComplaint))

class CarComplaintsItem(scrapy.Item):
    system = scrapy.Field()
    details = scrapy.Field()
    make = scrapy.Field()
    model = scrapy.Field()
    year = scrapy.Field()
    problem = scrapy.Field()

