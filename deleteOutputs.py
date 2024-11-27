import os
import shutil
import argparse

def delete_unwanted_files_and_folders(root_path):
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            if item not in ['color', 'depth']:
                shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            if item != 'intrinsics.txt':
                os.remove(item_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete unwanted files and folders.')
    parser.add_argument('--root_path', type=str, help='The root directory path')
    args = parser.parse_args()

    delete_unwanted_files_and_folders(args.root_path)