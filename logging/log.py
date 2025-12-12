import logging
import logging.config
import os

# pro_path = os.path.abspath('.')
# conf_path = os.path.abspath(os.path.join(pro_path + '/study/logging/logging.conf'))
# print(conf_path)
logging.basicConfig(format='[%(asctime)s] [%(levelname)s]:%(message)s', filename='study/log/test.log', level=logging.INFO)
# logging.config.fileConfig(conf_path)
logging.info('this is log')