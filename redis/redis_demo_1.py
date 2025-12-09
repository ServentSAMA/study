# 导入redis库
import redis

# 创建Redis连接，参数说明：
# host: Redis服务器地址
# port: Redis服务器端口
# db: 选择数据库编号（0-15）
# decode_responses: 将返回结果自动解码为字符串
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# 设置键值对
# 参数说明：
# name: 键名
# value: 值
r.set(name="key", value="value")

# 获取键对应的值并打印
print(r.get("key"))

# 其他常用操作示例
# 1. 设置带过期时间的键值对
r.setex("temp_key", 60, "temp_value")  # 60秒后自动删除

# 2. 获取多个键的值
values = r.mget(["key1", "key2", "key3"])
print(values)

# 3. 删除键
r.delete("key")

# 4. 检查键是否存在
if r.exists("key"):
    print("Key exists")
else:
    print("Key does not exist")

# 5. 自增操作
r.incr("counter")  # 值加1
r.incrby("counter", 5)  # 值加5
# 以下是python操作redis zset（有序集合）类型的基本操作教程

# 向有序集合中添加元素，参数说明：
# name: 有序集合的名称
# mapping: 字典，键为元素，值为分数
r.zadd("my_zset", {"member1": 1, "member2": 2, "member3": 3})

# 获取有序集合中指定元素的分数
score = r.zscore("my_zset", "member2")
print(f"member2的分数是: {score}")

# 获取有序集合中指定范围内的元素，按分数从小到大排序
# start: 起始索引，从0开始
# end: 结束索引，-1表示最后一个元素
members = r.zrange("my_zset", 0, -1)
print("按分数从小到大排序的元素:", members)

# 获取有序集合中指定范围内的元素及其分数，按分数从小到大排序
members_with_scores = r.zrange("my_zset", 0, -1, withscores=True)
print("按分数从小到大排序的元素及其分数:", members_with_scores)

# 获取有序集合中指定分数范围内的元素，按分数从小到大排序
# min: 最小分数
# max: 最大分数
members_in_score_range = r.zrangebyscore("my_zset", min=1, max=2)
print("分数在1到2之间的元素:", members_in_score_range)

# 获取有序集合中元素的数量
zset_count = r.zcard("my_zset")
print("有序集合中的元素数量:", zset_count)

# 移除有序集合中的指定元素
r.zrem("my_zset", "member3")
print("移除member3后，有序集合剩余元素:", r.zrange("my_zset", 0, -1))
# 以下是python操作redis消息队列的基本操作教程

# 生产者：向消息队列中添加消息
# 使用lpush方法将消息添加到列表的左侧（模拟队列的入队操作）
r.lpush("message_queue", "message1")
r.lpush("message_queue", "message2")
r.lpush("message_queue", "message3")

# 消费者：从消息队列中获取消息
# 使用rpop方法从列表的右侧获取消息（模拟队列的出队操作）
message = r.rpop("message_queue")
print("从消息队列中获取的消息:", message)

# 阻塞式获取消息，当队列中没有消息时，会等待指定的时间（这里设置为10秒）
# 如果在指定时间内有消息到达，会立即返回消息；否则返回None
blocking_message = r.brpop("message_queue", 10)
if blocking_message:
    print("阻塞式获取的消息:", blocking_message[1])
else:
    print("在指定时间内没有获取到消息")

