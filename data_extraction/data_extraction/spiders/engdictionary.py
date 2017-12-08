# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from scrapy import FormRequest
import re

class MaSpiderSpider(scrapy.Spider):
    name = 'engdictionary'
    domain = 'http://www.engineering-dictionary.org/Dictionary-of-Automotive-Terms/'

    def __init__(self):
        self.start_urls = ['http://www.engineering-dictionary.org/Dictionary-of-Automotive-Terms/A/1']

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, callback=self.parse_index)

    def parse_index(self, response):
        print(response.url)
        letters_urls = response.css('.bible_dictionary_body > div > a::attr(href)').extract()

        print(letters_urls)
        for url in letters_urls:
            yield Request(url, callback=self.parse_letter)


    def parse_letter(self, response):
        pages_urls = response.css('.bible_dictionary_body > a::attr(href)').extract()

        for url in pages_urls:
            yield Request(url, callback=self.parse_page)

    def parse_page(self, response):
        terms_urls = response.css('.bible_dictionary_body > table > tr > td > a::attr(href)').extract()
        term_extractor = re.compile('http:\/\/www.engineering-dictionary.org\/Dictionary-of-Automotive-Terms\/[\w,-]*\/([\w,-]*)')
        terms_name = [re.search(term_extractor, url).group(1) for url in terms_urls if re.search(term_extractor, url)]

        for term in terms_name:
            item = DictionaryItem()
            item["term"] = term.lower().replace('_', ' ')
            yield item

class DictionaryItem(scrapy.Item):
    term = scrapy.Field()
