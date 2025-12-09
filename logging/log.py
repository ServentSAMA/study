import logging
import logging.config
import os
import os

# Python日志框架教程
# 1. 基本配置
# logging.basicConfig() 是最简单的日志配置方式
# format参数定义日志输出格式
# filename参数指定日志文件路径
# level参数设置日志级别
logging.basicConfig(format='[%(asctime)s] [%(levelname)s]:%(message)s', filename='log/test.log', level=logging.DEBUG)

# 2. 日志级别
# DEBUG < INFO < WARNING < ERROR < CRITICAL
# 只有大于等于设置级别的日志才会被记录

# 3. 日志输出示例
logging.info('这是一条info级别的日志')
logging.error('这是一条error级别的日志')
logging.debug('这是一条debug级别的日志')
logging.warning('这是一条warning级别的日志')
logging.critical('这是一条critical级别的日志')

# 4. 高级配置（注释掉的示例）
# 可以使用logging.config模块通过配置文件进行更复杂的日志配置
# 5. 使用logging.config模块的配置教程
# 5.1 配置文件介绍
# logging.config模块允许我们通过配置文件来进行更复杂和灵活的日志配置。
# 常见的配置文件格式有两种：字典配置和INI格式配置。这里以INI格式为例。

# 5.2 配置文件准备
# 首先，我们需要创建一个INI格式的配置文件，例如 'logging.conf'。
# 以下是一个简单的 'logging.conf' 示例：
'''
[loggers]
keys=root

[handlers]
keys=fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=fileHandler

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('log/test.log', 'a')

[formatter_simpleFormatter]
format=[%(asctime)s] [%(levelname)s]:%(message)s
'''

# 5.3 在代码中使用配置文件
# 接下来，我们需要在代码中加载这个配置文件。
pro_path = os.path.abspath('.')
conf_path = os.path.abspath(os.path.join(pro_path, 'study', 'logging', 'logging.conf'))
try:
    logging.config.fileConfig(conf_path)
    print("成功加载日志配置文件")
except Exception as e:
    print(f"加载日志配置文件时出错: {e}")

# pro_path = os.path.abspath('.')
# conf_path = os.path.abspath(os.path.join(pro_path + '/study/logging/logging.conf'))
# logging.config.fileConfig(conf_path)