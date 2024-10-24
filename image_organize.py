import os
import shutil

def move_images_from_list(image_folder, destination_folder, file_list_path):
    """
    テキストファイルに書かれた画像ファイル名を基に、画像を指定したフォルダに移動する。

    Parameters:
        image_folder (str): 元の画像フォルダのパス
        destination_folder (str): 移動先のフォルダのパス
        file_list_path (str): 画像ファイル名が書かれているテキストファイルのパス
    """
    # 移動先フォルダが存在しない場合は作成する
    os.makedirs(destination_folder, exist_ok=True)

    # テキストファイルからファイル名を読み込む
    with open(file_list_path, 'r') as file:
        image_names = [line.strip() for line in file]

    # 各画像を移動
    for image_name in image_names:
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)

        # ファイルが存在するか確認して移動する
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved: {image_name}")
        else:
            print(f"Not found: {image_name}")

# 使用例
image_folder = 'Image_Cat/'          # 元の画像フォルダ
destination_folder = 'class5'  # 移動先フォルダ
file_list_path = 'class5.txt'       # 画像ファイル名が書かれたテキストファイル

move_images_from_list(image_folder, destination_folder, file_list_path)
