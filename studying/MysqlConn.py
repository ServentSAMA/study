import pymysql


class TestMysql:

    def __init__(self, username, host, password, database):
        self.username = username
        self.password = password
        self.host = host
        self.database = database

    def conn_mysql(self):
        conn = pymysql.connect(user=self.username, host=self.host, password=self.password, database=self.database,
                               charset='utf8')
        return conn

    def close_mysql(self):
        print("MySQL is Closed")

    # 查询数据
    def get_data(self):
        conn = self.conn_mysql()
        cur = conn.cursor()
        #
        while True:
            sql = input('输入SQL语句:')
            cur.execute(sql)
            results = cur.fetchall()
            for i in results:
                print(str(i))
            yn = input('按N断开连接，任意键继续：').strip()
            if yn == 'N':
                break
        #
        cur.close()
        self.close_mysql()


if __name__ == "__main__":
    # 定义变量
    username = input('用户名:').strip()
    host = input('主机名:').strip()
    passwd = input('密码:').strip()
    database = input('库名:').strip()
    # 使用try--except
    try:
        # 创建TestMysql的实例
        mysql = TestMysql(username, host, passwd, database)
        mysql.conn_mysql()
        mysql.get_data()
    except pymysql.err.ProgrammingError as e:
        print("Exception Error is %s" % e)
    except pymysql.err.OperationalError as e:
        print("Exception Error is %s" % e)
