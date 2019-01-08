# -*- coding: utf-8 -*-.
import logging
import os
import numpy as np  # conda install numpy
import importlib
import pickle
from multiprocessing import Pool
from time import time
from image_iterator import extract_images_from_files
#if importlib.util.find_spec('PyQt5'):
from plus_highlighting import feature_qt
from ImageProcessor import ImageProcessor

np.set_printoptions(linewidth=200)

DEBUG = True
LOGGING_FORMAT = '%(levelname)-8s [%(asctime)s] %(message)s'
if DEBUG:
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
else:
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


def prc_one_image(np_image):
    """Полностью обработать одну страницу (изображение в формате numpy)"""
    image_cls = ImageProcessor(np_image,
                               black_threshold=BLACK_THRESHOLD,
                               unmark_useless_cells_func=unmark_useless_cells)
    filled_cells = feature_qt(image_cls)
    # Теперь удаляем кусок ячеек, которые вообще никому не интересны
    if remove_useless_cells:
        filled_cells = remove_useless_cells(filled_cells)
    return filled_cells


def prc_list_of_files(files, *, black_threshold=230,
                      unmark_useless_cells_func=None,
                      remove_useless_cells_func=None):
    global unmark_useless_cells, remove_useless_cells, BLACK_THRESHOLD
    unmark_useless_cells = unmark_useless_cells_func
    remove_useless_cells = remove_useless_cells_func
    BLACK_THRESHOLD = black_threshold
    images = extract_images_from_files(files)
    recognized_pages = [prc_one_image(image) for image in images]
    return recognized_pages



if __name__ == '__main__':
    pass
    # Исключительно для отладки:
    # recognized_pages = prc_list_of_files(r'tests\test_imgs&pdfs\tst_01.pdf', njobs=2)
    # recognized_pages = prc_list_of_files(r'tests\test_imgs&pdfs\bad_scan.png', njobs=1)
    # print(recognized_pages)

    os.chdir(r'tests\test_imgs&pdfs')
    # images = extract_images_from_files('tst_01.pdf', pages_to_process=[0, 1])
    images = extract_images_from_files(r'C:\Dropbox\ВМШ 5-7 класс 2017-18\Py_VMSH_2017\Scan12.pdf')
    for image in images:
        print(feature_qt(ImageProcessor(image)))

