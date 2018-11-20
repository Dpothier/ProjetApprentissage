# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from scrapy import FormRequest
import re

class Motorera(scrapy.Spider):
    name = 'motorera'
    domain = 'https://www.motorera.com/dictionary/'


    def __init__(self):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z']
        two_letters = []
        for first_letter in letters:
            for second_letter in letters:
                two_letters.append(first_letter + second_letter)

        self.start_urls = []
        for two_letter in two_letters:
            self.start_urls.append('https://www.motorera.com/dictionary/{}.htm'.format(two_letter))


    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, callback=self.parse_subletter)

    def parse_index(self, response):
        letters_urls = response.css('.halfmoon > ul > li> a::attr(href)').extract()

        for letter in letters_urls:
            yield Request(self.domain + letter, callback=self.parse_letter)

    def parse_letter(self, response):
        subletters = response.css('header > .halfmoon')[1]
        subletters_url = subletters.css('ul > li > a::attr(href)').extrat()

        for subletters in subletters_url:
            yield Request(self.domain + subletters, callback=self.parse_subletter)

    def parse_subletter(self, response):
        words = response.css('div.main > dl > dt > a::text').extract()

        for word in words:
            item = DictionaryItem()
            item["word"] = word
            yield item


class DictionaryItem(scrapy.Item):
    word = scrapy.Field()


