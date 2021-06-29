import csv
import json
import jieba.analyse

with open('./news.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    json_hotspot = json_data['热点']
    hotspot_list = []
    for row in json_hotspot:
        if row['content']:
            text = row['content']
            keys = jieba.analyse.extract_tags(sentence=text, topK=10, withWeight=True, allowPOS=('ns', 'n'))
            keyWords = []
            for item in keys:
                keyWords.append(item[0])
            strKeyWords = ','.join(keyWords)
            hotspot_list.append([row['title'], row['date'], row['content'], row['url'], strKeyWords])


    title = [['标题', '日期', '内容', 'url', '文章中10个关键词']]
    with open(r'./news.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(title)
        writer.writerows(hotspot_list)
