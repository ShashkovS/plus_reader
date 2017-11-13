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

    # TODO: пока безусловное сохранение в save_marked_name — это треш
    # Вычисляем координаты узлов сетки

    image_cls = ImageProcessor(np_image)
    # if unmark_useless_cells:
    #     filled_cells = unmark_useless_cells(filled_cells)
    # If PyQt5 installed, than show
    if importlib.util.find_spec('PyQt5'):
        feature_qt(image_cls)
    # Теперь удаляем кусок ячеек, которые вообще никому не интересны
    # if remove_useless_cells:
    #     filled_cells = remove_useless_cells(filled_cells)
    return image_cls.filled_cells




def prc_all_images(iterable_of_np_images, njobs=1):
    stt = time()
    if njobs == 1:
        recognized_pages = [prc_one_image(image, pg_num) for pg_num, image in enumerate(iterable_of_np_images)]
    else:
        prc_pool = Pool(njobs)
        recognized_pages = prc_pool.map(prc_one_image, iterable_of_np_images)
    ent = time()
    if DEBUG:
        logging.info('Done in ' + str(ent - stt))
    return recognized_pages


def prc_list_of_files(files, *, njobs=1, black_threshold=210, unmark_useless_cells_func=None, remove_useless_cells_func=None):
    global unmark_useless_cells, remove_useless_cells, BLACK_THRESHOLD
    unmark_useless_cells = unmark_useless_cells_func
    remove_useless_cells = remove_useless_cells_func
    BLACK_THRESHOLD = black_threshold
    images = extract_images_from_files(files)
    np_images = (np.array(img.convert("L")) for img in images)
    recognized_pages = prc_all_images(np_images, njobs=njobs)
    return recognized_pages



if __name__ == '__main__':
    pass
    # Исключительно для отладки:
    # recognized_pages = prc_list_of_files(r'tests\test_imgs&pdfs\tst_01.pdf', njobs=2)
    # recognized_pages = prc_list_of_files(r'tests\test_imgs&pdfs\bad_scan.png', njobs=1)
    # print(recognized_pages)


    os.chdir(r'tests\test_imgs&pdfs')
    images = extract_images_from_files('tst_01.pdf', pages_to_process=[0, 1])
    # np_images = (np.array(img.convert("L")) for img in images)
    image_cls = ImageProcessor(next(images))
    # prc_one_image(next(images))
    feature_qt(image_cls)

    # recognized_pages = prc_all_images(np_images, njobs=2)
    # gray_np_image = cv2.cvtColor(cv2.imread('test_prepated_image_01.png'), cv2.COLOR_BGR2GRAY)
    # gray_np_image = cv2.cvtColor(cv2.imread('bad_scan.png'), cv2.COLOR_BGR2GRAY)
    # filled_cells, coords_of_horiz_lns, coords_of_vert_lns = prc_one_prepared_image(gray_np_image)
    # # Запишем в дамп, чтобы запускалось быстрее
    # with open(r'test_dump_2.pickle', 'wb') as f:
    #     pickle.dump((gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns), f)
    # exit()
    # with open(r'test_dump_2.pickle', 'rb') as f:
    #     (gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns) = pickle.load(f)
    # coords_of_horiz_lns = coords_of_horiz_lns
    # filled_cells = feature_qt(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns)
    # print(filled_cells)
