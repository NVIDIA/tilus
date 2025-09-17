import os
from setuptools import setup


def get_data_files():
    data_directories = ['include']

    data_files = []

    for data_dir in data_directories:
        for root, dirs, files in os.walk(data_dir):
            relative_root = os.path.relpath(root, data_dir)

            # The destination directory relative to the site-packages directory
            # For example, 'titus/include/titus/cuda'
            destination_dir = os.path.join('titus', data_dir, relative_root)

            # List the full paths of the files to be included
            file_paths = [os.path.join(root, filename) for filename in files]

            # Append the destination and file paths to the list
            if file_paths:
                data_files.append((destination_dir, file_paths))
    return data_files

setup(
    data_files=get_data_files(),
)
