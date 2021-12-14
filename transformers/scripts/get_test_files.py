from shutil import copy
import os

FILEPATHS_TXT = 'C:\\Users\\anke\\PycharmProjects\\pythonProject\\fpaths_test (1).txt'
FILES = 'E:\\paintings\\test_quest'


def mkdirs(newdir, mode=0o777):
    try:
        os.makedirs(newdir, mode)
    except OSError as err:
        return err

def get_folder_with_files():
    # mkdirs('E:\\paintings\\test_quest1')
    with open(FILEPATHS_TXT, 'r', encoding='utf8') as fpaths:
        for line in fpaths:
            line = line.split()
            line = line[1].replace('F', "E")
            copy(line, FILES)


if __name__ == '__main__':
    get_folder_with_files()
