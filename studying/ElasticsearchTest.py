from elasticsearch import Elasticsearch
"""
TODO 没有成功
"""
# es = Elasticsearch()    # 默认连接本地elasticsearch
# es = Elasticsearch(['127.0.0.1:9200'])  # 连接本地9200端口

es = Elasticsearch(hosts="http://101.42.137.122:9200")

print(es.index(index='py2',  id=1, body={'name': "张开", "age": 18}))
print(es.get(index='py2',  id=1))
