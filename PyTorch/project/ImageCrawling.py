"""
bookcase, chair
cat face, dog face
--> total 4 categories
"""

from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'C:/Users/young/Pictures/Prograpy_image/dog'},
                             feeder_threads=1,
                             parser_threads=2,
                             downloader_threads=4,
                             )
crawler.crawl(keyword='dog', max_num=500)