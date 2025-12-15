import pdfplumber
import pandas as pd
import numpy as np

pdf_path = 'E:\迅雷下载\hqmx_20240408213526_00001\hqmx_20240408213526_0000{0}.pdf'
result_df = pd.DataFrame()
for i in range(5):
    hqmx = pdfplumber.open(pdf_path.format(i + 1))
    for page in hqmx.pages:
        table = page.extract_table()
        df_detail = pd.DataFrame(table[1:], columns=table[0])
        # 合并每页的数据集
        result_df = pd.concat([df_detail, result_df], ignore_index=True)






# writer = pd.ExcelWriter('E:\迅雷下载\hqmx_20240408213526_00001\\test.xlsx')
# print(result_df)
# result_df.to_excel(writer, sheet_name='明细')

array = []

# print(result_df.iloc[0]['交易金额'].split('\n'))

for index, row in result_df.iterrows():

    no = row['序号'].split('\n')
    note = row['摘要'].split('\n')
    date = row['交易日期'].split('\n')
    yue = row['交易金额'].split('\n')
    zhanghu = row['账户余额'].split('\n')
    
    for i in range(len(note)):
        map = [no[i],note[i], date[i], yue[i], zhanghu[i]]
        array.append(map)
print(array)

A = np.array(array)

data = pd.DataFrame(A)

writer = pd.ExcelWriter('E:\迅雷下载\hqmx_20240408213526_00001\\A.xlsx')
data.to_excel(writer, 'page_1', float_format='%.5f')

writer.close()


