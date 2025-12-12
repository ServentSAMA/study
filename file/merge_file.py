'''
合并文件夹的所有文件到一个文件
'''
import os

def get_all_files_in_folder(folder_path):
    files = []
    for root, directories, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files

def merge_files(files, output_file):
    with open(output_file, "wb") as out:
        for file in files:
            with open(file, "rb") as f:
                out.write(f.read())
    print("All files merged successfully!")

folder_path = "D:\\SQL_BACK\\数字孪生\\20251114"
files = get_all_files_in_folder(folder_path)
print(files)


output_file = "D:\\SQL_BACK\\数字孪生\\all.sql"
merge_files(files, output_file)


