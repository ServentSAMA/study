import win32com.client

def ConvertDocToPdf(src, dst):
    wps = win32com.client.Dispatch("Ket.Application")
    excel = wps.Workbooks.Open(src)
    excel.ExportAsFixedFormat(0, dst)
    excel.Close()
    wps.Quit()


if __name__ == '__main__':
    ConvertDocToPdf(r"C:\\Users\\Shen\\Desktop\\项目\\科技成果\\北海局高层次科技创新人才申请表（20250328求确认版）.xlsx", r"C:\\Users\\Shen\\Desktop\\项目\\test.pdf")