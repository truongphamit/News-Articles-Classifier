# -*- coding: utf8 -*-

import urllib2

import os
from BeautifulSoup import BeautifulSoup

base_url = "http://dantri.com.vn/"
categories = ["xa-hoi", "the-gioi", "the-thao", "giao-duc-khuyen-hoc", "kinh-doanh", "van-hoa", "giai-tri", "phap-luat", "suc-khoe", "suc-manh-so", "o-to-xe-may", "tinh-yeu-gioi-tinh"]

def crawNews(url, path):
    print url
    page = urllib2.urlopen(url, timeout=10).read()
    soup = BeautifulSoup(page.decode('utf8', 'ignore'), convertEntities=BeautifulSoup.HTML_ENTITIES)
    title = soup.find("h1", "fon31 mgb15").getText()
    description = soup.find("h2", "fon33 mt1 sapo").getText().replace("Dân trí".decode("utf-8"), "")
    content = soup.find("div", "fon34 mt3 mr2 fon43 detail-content").findAll("p")
    body = ""
    for i in content:
        body += "\n" + i.getText()
    file = open(path, "w")
    file.write(title.encode("utf-8") + "\n" + description.encode("utf-8") + "\n" + body.encode("utf-8"))
    file.close()
    print "Get News: " + path

def getCate(cate):
    directory = "Data/Test/" + cate
    if not os.path.exists(directory):
        os.makedirs(directory)
    count = 1
    for i in range(600, 700):
        try:
            print "----------------------------" + base_url + cate + "/trang-%d.htm" % i
            page = urllib2.urlopen(base_url + cate + "/trang-%d.htm" % i, timeout=15).read()
            soup = BeautifulSoup(page.decode('utf8', 'ignore'), convertEntities=BeautifulSoup.HTML_ENTITIES)
            links = soup.find('div', id="listcheckepl").findAll("div")
            for l in links:
                try:
                    crawNews(base_url + l.find("a").get('href'), "%s/%s_%d.txt" % (directory, cate, count))
                    count = count + 1
                except:
                    print "error"
        except:
            print "error"

for ct in categories:
    getCate(ct)