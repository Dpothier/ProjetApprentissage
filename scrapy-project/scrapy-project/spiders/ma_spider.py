# -*- coding: utf-8 -*-
import scrapy


class MaSpiderSpider(scrapy.Spider):
    name = 'ma_spider'

    def __init__(self, arg):
        start_urls = [
            'https://www.google.ca/search?source=hp&ei=-YAEWtq6Osz4jwT045GACw&q=python&oq=python&gs_l={}psy-ab.3..35i39k1l2j0i67k1l5j0l3.927.5352.0.5893.23.15.5.0.0.0.186.1634.1j11.13.0....0...1.1.64.psy-ab..5.17.1679.6..0i131i46k1j46i131k1j0i10k1j0i131k1.90.CrfB5oVlM_Q']
        return callback
    
    def parse(self, response):
        selector = "#ires > ol > div > h3> a"
        return_value = {}
        for index, item in enumerate(response.css(selector).extract()):
            return_value[index] = item
        print(return_value)
        yield None

    def parse_page_1(self,response):

        yield(self.parse(response))