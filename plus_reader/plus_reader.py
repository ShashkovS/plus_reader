# -*- coding: utf-8 -*-.
import logging
import os
import cv2  # pip install --upgrade opencv-python
import numpy as np  # conda install numpy
import importlib
import pickle
from multiprocessing import Pool
from time import time
from image_iterator import extract_images_from_files
#if importlib.util.find_spec('PyQt5'):
from plus_highlighting import feature_qt

np.set_printoptions(linewidth=200)

DEBUG = True
LOGGING_FORMAT = '%(levelname)-8s [%(asctime)s] %(message)s'
if DEBUG:
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
else:
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


def align_image(img):
    """Выровнять изображение.
    Идея подхода в следующем: рассмотрим несколько разных поворотов и выберем из них "лучший".
    Лучший — это тот, в котором "горизонтальные" линии занимают меньше всего места по вертикали.
    """
    # TODO: Этот кусок работает дико медленно и занимает большую часть времени
    # TODO: Скорее всего можно определять угол поворота как-нибудь быстрее
    logging.info('Aligning image...')
    rows, cols = img.shape

    def try_angles(angles, sv, ev):
        yy = np.zeros_like(angles)
        yy[0], yy[-1] = sv, ev
        for i in range(1, len(angles) - 1):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angles[i], 1)
            dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, borderValue=255)
            vr = (255 - dst).astype(float).sum(axis=1)
            yy[i] = (vr**4).sum()
        return yy

    xx = [-5, 0, 5]
    yy = [0, 0, 0]
    mx = 1
    for _ in range(4):
        xx = np.linspace(xx[mx-1], xx[mx+1], 7)
        yy = try_angles(xx, yy[mx-1], yy[mx+1])
        mx = np.argmax(yy)
        best_angle = xx[mx]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),best_angle,1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, borderValue=255)
    logging.info(str('Best angle = ') + str(best_angle) + str(' Penalty= ') + str(yy[mx]))
    return dst


def blur_image(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return blur


def img_to_bitmap_np(img):
    """
    в формате numpy ndarrray в черно-белом формате.
    Например, в виде (здесь 255 --- это белый, 0 --- чёрный)
    array([[ 255.,  255.,  255.,  255.,  255.],
           [ 255.,    0.,    0.,    0.,  255.],
           [ 255.,  255.,  255.,  255.,  255.]])
    """
    # TODO: С поворотом здесь какой-то треш. Это должно быть вынесено в extract_images_from_files
    # if ar.shape[1] > ar.shape[0]:  # Почему-то изображение повёрнуто
    #     ar = ar.T[::-1, :]

    # BITMAP_BORDER = 233
    # ar_bit = np.zeros_like(ar)
    # ar_bit[ar > BITMAP_BORDER] = 255
    # ret, ar_bit = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret = (max(ret, 190) + 230) // 2
    ret = BLACK_THRESHOLD
    ret, ar_bit = cv2.threshold(img, ret, 255, cv2.THRESH_BINARY)
    return ar_bit


def remove_dots_from_image(gray_np_image):
    """Удаляет мелкий сор из изображения
    Удаляем все точки, у которых нет достаточно убедительных соседей
    """
    clean1 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    clean2 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    clean3 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    clean = clean1 & clean2 & clean3
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    if DEBUG: cv2.imwrite("_clean.png", clean)
    return clean


def remove_background(gray_np_image):
    """Удаляет мелкий сор из изображения"""
    med = cv2.medianBlur(gray_np_image, 75)
    dif = cv2.add(gray_np_image, 255-med)
    if VERBOSE:
        cv2.imwrite("_med.png", med)
    return dif


def find_lines_on_image(gray_np_image, direction):
    """Найти длинные линии в ч/б изобажении в формате numpy ndarray.
    Возвращает ч/б изображения с сильно растянутыми и жирными линиями.
    Предполагается, что размер изображения - А4, и таблица занимает
    большую часть страницы. Если это не так, то нужно
    "играть параметрами"
    """
    if direction not in ('horizontal', 'vertical'):
        logging.error('horizontal or vertical, please')
        raise ValueError('horizontal or vertical, please')
    im_height, im_width = gray_np_image.shape
    # Каждый пиксель будет размыт до квадрата с такой стороной
    # TODO: Здесь мутные константы, которые я подбирал руками для наших кондуитов. Это — треш
    ERODE_SIZE = round((im_width + im_height) / 600)
    MIN_LINE_LEN = round((im_width + im_height) / 15)
    EXPAND_LINE_LEN = round((im_width + im_height) / 6)
    EXPAND_LINE_WID = round((im_width + im_height) / 850)
    if direction == 'horizontal':
        dilate_parm = (MIN_LINE_LEN, 1)
        erode_parm = (EXPAND_LINE_LEN, EXPAND_LINE_WID)
    elif direction == 'vertical':
        dilate_parm = (1, MIN_LINE_LEN)
        erode_parm = (EXPAND_LINE_WID, EXPAND_LINE_LEN)
    else:
        logging.error('direction must be "horizontal" or "vertical"')
        raise ValueError('direction must be "horizontal" or "vertical"')
    # Немного "размажем"
    img_copy = cv2.erode(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (ERODE_SIZE, ERODE_SIZE)))
    if DEBUG: cv2.imwrite('_' + direction + "1.png", img_copy)
    # Теперь оставим только то, что представляет собой "длинную" линию
    img_copy = cv2.dilate(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_parm))
    if DEBUG: cv2.imwrite('_' + direction + "2.png", img_copy)
    # И назад растянем
    img_copy = cv2.erode(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, erode_parm))
    if DEBUG: cv2.imwrite('_' + direction + "3.png", img_copy)
    img_copy = cv2.dilate(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_parm))
    if DEBUG: cv2.imwrite('_' + direction + "4.png", img_copy)
    return img_copy


def mark_plus(gray_np_image, horizontal_lines, vertical_lines):
    """Принимает на вход картинку и маску таблицы.
    Размалёвывает плюсы
    """
    BLACK_LINE_THRESHOLD = 175
    table_mask = cv2.min(horizontal_lines, vertical_lines)  # Горизонтальные и вертикальные линии вместе
    table_mask[table_mask>BLACK_LINE_THRESHOLD] = 255
    table_mask[table_mask<=BLACK_LINE_THRESHOLD] = 0
    plus = (gray_np_image | ~table_mask)  # Убрали из изображения сами линии
    # plus = img_to_bitmap_np(plus)
    # plus = remove_dots_from_image(plus)
    # Замажем чёрным всё, в окрестности чего много точек.
    # Почти все плюсы превратятся в "жирные" кляксы
    # TODO: Здесь мутные константы, которые я подбирал руками для наших кондуитов. Это — треш
    plus = ~cv2.adaptiveThreshold(plus, 210, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, -9)
    # Вернём наместро сами плюсы
    plus = cv2.min(plus, gray_np_image)
    # Очистим точки в границах таблицы, чтобы не мешались
    plus |= ~table_mask
    if DEBUG: cv2.imwrite("_plus.png", plus); cv2.imwrite("_table_mask.png", table_mask);
    return plus


def calcutale_lines_coords(horizontal_lines, vertical_lines):
    """Берёт на вход картинку с горизонтальными линиями и вертикальными линиями,
    вычисляет их цетры и возвращает список центров горизонтальных и вертикальных линий
    """
    MIN_LINE_HOR = .6
    MIN_LINE_VER = .6
    im_height, im_width = horizontal_lines.shape
    hor_l = np.zeros(im_height, np.bool)
    vert_l = np.zeros(im_width, np.bool)
    for i in range(im_height):
        hor_l[i] = (im_width - np.sum(horizontal_lines[i, :] > 128)) / im_width > MIN_LINE_HOR
    for i in range(im_width):
        vert_l[i] = (im_height - np.sum(vertical_lines[:, i] > 128)) / im_height> MIN_LINE_VER
    def prc(ar):
        ans = []
        bl_st, bl_en = -1, -1
        for i, bl in enumerate(ar):
            if bl and bl_st < 0:
                bl_st = i
                bl_en = i
            elif bl:
                bl_en = i
            elif not bl and bl_st >= 0:
                ans.append((bl_en + bl_st) // 2)
                bl_st, bl_en = -1, -1
        return ans
    logging.info('Finding centers')
    hor = prc(hor_l)
    vert = prc(vert_l)
    if DEBUG:  # Рисуем получившуюся табличку
        lines = horizontal_lines.copy()
        lines.fill(255)
        for i in hor:
            for j in range(lines.shape[1]):
                lines[i][j] = 0
                lines[i+1][j] = 0
                lines[i-1][j] = 0
        for j in vert:
            for i in range(lines.shape[0]):
                lines[i][j] = 0
                lines[i][j+1] = 0
                lines[i][j-1] = 0
        cv2.imwrite("_lines.png", lines)
    return hor, vert


def find_filled_cells(gray_np_image, hor, vert):
    """Самая главная функция — определяет, заполнена ли ячейка
    """
    # TODO: убрать магические константы
    MIN_FILLED_PART = 0.030
    MIN_FILLED_DOTS = gray_np_image.size / 9000
    BLACK_DOT_THRESHOLD = 225
    rows, colums = len(hor) - 1, len(vert) - 1
    filled = np.zeros((rows, colums), np.bool)
    for i in range(rows):
        for j in range(colums):
            block = gray_np_image[hor[i]:hor[i+1], vert[j]:vert[j+1]]
            # logging.info('block shape = ' + str(block.shape))
            part = 1 - (np.sum(block) / block.size / 255)
            filled[i][j] = (part >= MIN_FILLED_PART or
                            block[block < BLACK_DOT_THRESHOLD].size >= MIN_FILLED_DOTS)
    return filled


def prc_one_image(np_image, pgnum=[0]):
    """Полностью обработать одну страницу (изображение в формате numpy)"""
    if isinstance(pgnum, list):
        # Используется хук для того, чтобы использовать уникальные номера
        use_pgnum = pgnum[0]
        pgnum[0] += 1
    else:
        use_pgnum = pgnum
    gray_np_image = align_image(np_image)
    gray_np_image = blur_image(gray_np_image)
    # Удаляем мусор
    gray_np_image = remove_background(gray_np_image)
    gray_np_image[gray_np_image<100] = 0
    gray_np_image = img_to_bitmap_np(gray_np_image)
    gray_np_image = remove_dots_from_image(gray_np_image)

    # TODO: пока безусловное сохранение в save_marked_name — это треш
    horizontal_lines = find_lines_on_image(gray_np_image, 'horizontal')
    vertical_lines = find_lines_on_image(gray_np_image, 'vertical')
    # Вычисляем координаты узлов сетки
    coords_of_horiz_lns, coords_of_vert_lns = calcutale_lines_coords(horizontal_lines, vertical_lines)
    bitmap_np_image = gray_np_image
    plus = mark_plus(bitmap_np_image, horizontal_lines, vertical_lines)
    filled_cells = find_filled_cells(plus, coords_of_horiz_lns, coords_of_vert_lns)

    if unmark_useless_cells:
        filled_cells = unmark_useless_cells(filled_cells)
    # If PyQt5 installed, than show
    if importlib.util.find_spec('PyQt5'):
        filled_cells = feature_qt(np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns)
    # Теперь удаляем кусок ячеек, которые вообще никому не интересны
    if remove_useless_cells:
        filled_cells = remove_useless_cells(filled_cells)
    return filled_cells


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
    # images = extract_images_from_files('tst_01.pdf', pages_to_process=[0, 1])
    # np_images = (np.array(img.convert("L")) for img in images)
    # recognized_pages = prc_all_images(np_images, njobs=2)
    #gray_np_image = cv2.cvtColor(cv2.imread('test_prepated_image_01.png'), cv2.COLOR_BGR2GRAY)
    # gray_np_image = cv2.cvtColor(cv2.imread('bad_scan.png'), cv2.COLOR_BGR2GRAY)
    #filled_cells, coords_of_horiz_lns, coords_of_vert_lns = prc_one_prepared_image(gray_np_image)
    # # Запишем в дамп, чтобы запускалось быстрее
    # with open(r'test_dump_2.pickle', 'wb') as f:
    #     pickle.dump((gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns), f)
    # exit()
    with open(r'test_dump_2.pickle', 'rb') as f:
        (gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns) = pickle.load(f)
    coords_of_horiz_lns = coords_of_horiz_lns
    filled_cells = feature_qt(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns)
    print(filled_cells)
