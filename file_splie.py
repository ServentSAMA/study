import os


def split_txt_file(file_path, max_size_kb=500, output_dir='output'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取基础文件名和扩展名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extension = os.path.splitext(file_path)[1]

    # 计算最大字节数（500KB）
    max_size = max_size_kb * 1024
    file_count = 1

    with open(file_path, 'rb') as f:
        chunk = f.read(max_size)
        remaining = b''

        while chunk:
            # 查找最后一个换行符位置
            last_newline = chunk.rfind(b'\n')

            # 如果找到换行符且不是最后一行
            if last_newline != -1:
                remaining = chunk[last_newline + 1:]
                chunk = chunk[:last_newline + 1]
            else:
                remaining = b''

            # 生成输出文件名
            output_name = f"{base_name}_{file_count:03d}{extension}"
            output_path = os.path.join(output_dir, output_name)

            # 写入分割文件
            with open(output_path, 'wb') as out_file:
                out_file.write(chunk)

            # 准备下一个块（当前剩余部分 + 新读取内容）
            chunk = remaining + f.read(max_size - len(remaining))
            file_count += 1


if __name__ == "__main__":
    # 使用示例
    input_file = "D:\电子书\\temp\\唐寅在异界 (1).txt"  # 输入文件路径
    split_txt_file(input_file)
