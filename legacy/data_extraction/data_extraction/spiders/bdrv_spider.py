# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request
from scrapy import FormRequest
import re

class MaSpiderSpider(scrapy.Spider):
    visited_recall_numbers = set()
    name = 'bdrv_spider'
    domain = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/'


    def __init__(self):

        #acura = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=4146&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #audi = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=1878&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #bmw = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2341&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #buick = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=1893&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #cadilac = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=1870!6869&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #cherokee = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=5525!27679&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #chevrolet = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=5359!4804!3229!1896!6128&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        chrysler = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=6181!27999!1872!5986!34917!6402&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        dodge = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=33038!24277!28000!6400!1849!2444!3714!4864!5115&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        ford = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=4273!3712!3592!1867!8659!6126!5711!34816!17580!5164!4862!4800!4745&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        honda = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=4785!1954&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #hyundai = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=3759&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #mercedes = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2471!34716!2118&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #michelin = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=7502&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #mitsubishi = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=7355&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #pontiac = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=1890&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #subaru = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2510&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        toyota = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2474!2734&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #toyo = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=13058&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #volkwagen = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2154!24279&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #volvo = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=4876!1838&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'
        #yamaha = 'http://wwwapps.tc.gc.ca/Saf-Sec-Sur/7/VRDB-BDRV/search-recherche/results-resultats.aspx?lang=eng&mk=2032!4783&md=0&fy=0&ty=9999&ft=&ls=0&sy=0'

        self.start_urls = [chrysler, dodge, ford, honda, toyota]

    def parse(self, response):
        details_selector = "table > tr > td > a"
        recalls = response.css(details_selector)
        recalls_url = recalls.css('::attr(href)').extract()
        for index, item in enumerate(recalls.css('::text').extract()):
            if item not in self.visited_recall_numbers:
                self.visited_recall_numbers.add(item)
                yield Request('{}{}'.format(self.domain, recalls_url[index]), self.parseDetails)

        if recalls_url:
            yield FormRequest.from_response(response, formnumber=1, clickdata={'value': 'Next'}, callback=self.parse)



    def parseDetails(self, response):
        id_pattern = re.compile('# (\d*)')
        id_line = response.css('#BodyContent_LB_Recall_Number::text').extract_first()
        recall_id = re.search(id_pattern, id_line).group(1)
        system = response.css('#BodyContent_LB_System_d::text').extract_first()
        details = response.css('#BodyContent_LB_RecallDetail_d::text').extract_first()

        #First row contains column names so is ignored
        model_table_rows = response.css('#BodyContent_DG_RecallDetail > tr')[1:]
        models = []
        for index, item in enumerate(model_table_rows):
            texts = item.css('td > span::text').extract()
            make = texts[0]
            model = texts[1]
            years = texts[2].split()
            for year in years:
                item = RecallItem()
                item['recall_id'] = recall_id
                item['system'] = system
                item['details'] = details
                item['make'] = make
                item['model'] = model
                item['year'] = year

                yield item


class RecallItem(scrapy.Item):
    recall_id = scrapy.Field()
    system = scrapy.Field()
    details = scrapy.Field()
    make = scrapy.Field()
    model = scrapy.Field()
    year = scrapy.Field()

