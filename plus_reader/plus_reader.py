# -*- coding: utf-8 -*-.
import logging
import os
import cv2  # pip install --upgrade opencv-python
import numpy as np  # conda install numpy
from multiprocessing import Pool
from time import time
from image_iterator import extract_images_from_files
from plus_highlighting import feature_qt
import pickle


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


def scan_image_generator(pdf_filename):
    """Генератор, возвращающий последовательность изображений для распознавания.
    в формате numpy ndarrray в черно-белом формате.
    Например, в виде (здесь 255 --- это белый, 0 --- чёрный)
    array([[ 255.,  255.,  255.,  255.,  255.],
           [ 255.,    0.,    0.,    0.,  255.],
           [ 255.,  255.,  255.,  255.,  255.]])
    Может брать данные из pdf или напрямую из сканированных картинок
    """
    BITMAP_BORDER = 150
    for img in extract_images_from_files(pdf_filename):
        ar = np.array(img.convert("L"))  # Делаем ч/б
        # TODO: С поворотом здесь какой-то треш. Это должно быть вынесено в extract_images_from_files
        if ar.shape[1] > ar.shape[0]:  # Почему-то изображение повёрнуто
            ar = ar.T[::-1, :]
        ar = align_image(ar)
        ar_bit = np.zeros_like(ar)
        ar_bit[ar > BITMAP_BORDER] = 255
        yield ar_bit


def img_to_bitmap_np(img):
    """
    в формате numpy ndarrray в черно-белом формате.
    Например, в виде (здесь 255 --- это белый, 0 --- чёрный)
    array([[ 255.,  255.,  255.,  255.,  255.],
           [ 255.,    0.,    0.,    0.,  255.],
           [ 255.,  255.,  255.,  255.,  255.]])
    """
    ar = np.array(img.convert("L"))  # Делаем ч/б
    # TODO: С поворотом здесь какой-то треш. Это должно быть вынесено в extract_images_from_files
    if ar.shape[1] > ar.shape[0]:  # Почему-то изображение повёрнуто
        ar = ar.T[::-1, :]
    ar = align_image(ar)

    # BITMAP_BORDER = 233
    # ar_bit = np.zeros_like(ar)
    # ar_bit[ar > BITMAP_BORDER] = 255

    blur = cv2.GaussianBlur(ar, (5, 5), 0)
    ret, ar_bit = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret = (max(ret, 210) + 250) // 2
    ret, ar_bit = cv2.threshold(blur, ret, 255, cv2.THRESH_BINARY)

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
    BLACK_LINE_THRESHOLD = 50
    table_mask = horizontal_lines & vertical_lines  # Горизонтальные и вертикальные линии вместе
    table_mask[table_mask>BLACK_LINE_THRESHOLD] = 255
    table_mask[table_mask<=BLACK_LINE_THRESHOLD] = 0
    plus = (gray_np_image | ~table_mask)  # Убрали из изображения сами линии
    # Замажем чёрным всё, в окрестности чего много точек.
    # Почти все плюсы превратятся в "жирные" кляксы
    # TODO: Здесь мутные константы, которые я подбирал руками для наших кондуитов. Это — треш
    plus = ~cv2.adaptiveThreshold(plus, 210, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, -9)
    # Вернём наместро сами плюсы
    plus = np.minimum(plus, gray_np_image)
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
    MIN_FILLED_PART = 0.10
    MIN_FILLED_DOTS = gray_np_image.size / 9000
    BLACK_DOT_THRESHOLD = 80
    rows, colums = len(hor) - 1, len(vert) - 1
    filled = np.zeros((rows, colums), np.bool)
    for i in range(rows):
        for j in range(colums):
            block = gray_np_image[hor[i]:hor[i+1], vert[j]:vert[j+1]]
            part = 1 - (np.sum(block) / block.size / 255)
            filled[i][j] = (part >= MIN_FILLED_PART or
                            block[block < BLACK_DOT_THRESHOLD].size >= MIN_FILLED_DOTS)
    return filled


def unmark_useless_cells(filled_cells):
    """Может оказаться, что в таблице некоторые строки всегда заполнены. И нам не итересно, заполнены ли они.
    Эта фукнция снимает с них отметки, например, для того, чтобы эти ячейки не подсвечивались как заполненные"""
    # Лично у нас первая и последняя строка, а также нулевой и второй столбец отмечать не нужно
    # TODO: здесь мутный хардкод под наши кондуиты. Это должно быть как-то переделано
    # filled_cells[filled_cells[:, 2] == False, :] = False
    # filled_cells[:, 0] = False
    # filled_cells[0, :] = False
    # filled_cells[-1, :] = False
    return filled_cells


def remove_useless_cells(filled_cells):
    """Часть ячеек нас совершенно не интересует. Удалим из итоговой выдачи"""
    # Лично у нас первая и последняя строка, а также первый столбец не нужны
    # Кроме того, вовсе удалим строки, в которых не заполнена фамилия.
    # TODO: здесь мутный хардкод под наши кондуиты. Это должно быть как-то переделано
    # filled_cells = filled_cells[1:-1, 1:]
    # filled_cells = filled_cells[filled_cells[:, 1] == True, :]
    # filled_cells = np.delete(filled_cells, 1, axis=1)  # Здесь столбец с фамилией
    logging.info('*'*100)
    logging.info(filled_cells.astype(int))
    logging.info('*'*100)
    return filled_cells


def prc_one_prepared_image(gray_np_image, save_marked_name=None):
    """
    Распознаёт кондуит в ЧБ-изображении, переданном в виде numpy ndarray'я.
    Возвращает список списков [номер листа, номер строки, None, список результатов], например,
    [2, 19, None, [1, 0, 0, 1, 1, 1, 0, 0, 0]].
    Кроме того, возвращает цветное изображение, на котором жёлтым выделены ячейки,
    распознанные как плюс. Изображение в формате трёхцветного numpy ndarray'я
    (первая координата - номер строки, вторая - столбца, третья - цвет, blue=0, green=1, red=2
    """
    # Удаляем мусор
    clean = gray_np_image
    # clean = remove_dots_from_image(gray_np_image)
    # Находим горизонтальные и вертикальные линии
    horizontal_lines = find_lines_on_image(gray_np_image, 'horizontal')
    vertical_lines = find_lines_on_image(gray_np_image, 'vertical')
    # Вычисляем координаты узлов сетки
    horizontal_coords, vertical_coords = calcutale_lines_coords(horizontal_lines, vertical_lines)
    plus = mark_plus(clean, horizontal_lines, vertical_lines)
    filled_cells = find_filled_cells(plus, horizontal_coords, vertical_coords)
    filled_cells = unmark_useless_cells(filled_cells)
    # TODO Благодаря GUI кусок с выводом отмеченных точек переезжает «на потом»
    # if save_marked_name:
    #     colored = mark_filled_cells(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    #     cv2.imwrite(save_marked_name, colored)
    return filled_cells, horizontal_coords, vertical_coords


def prc_one_image(pil_image, pgnum=[0]):
    """Полностью обработать одну страницу (изображение в формате PIL)"""
    if isinstance(pgnum, list):
        # Используется хук для того, чтобы использовать уникальные номера
        use_pgnum = pgnum[0]
        pgnum[0] += 1
    else:
        use_pgnum = pgnum
    gray_np_image = img_to_bitmap_np(pil_image)
    # TODO: пока безусловное сохранение в save_marked_name — это треш
    filled_cells, horizontal_coords, vertical_coords = prc_one_prepared_image(gray_np_image, save_marked_name="sum_page_{}.png".format(use_pgnum))
    filled_cells = feature_qt(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    # Теперь удаляем кусок ячеек, которые вообще никому не интересны
    filled_cells = remove_useless_cells(filled_cells)
    return filled_cells


def prc_all_images(iterable_of_pil_images, njobs=1):
    stt = time()
    if njobs == 1:
        recognized_pages = [prc_one_image(image, pg_num) for pg_num, image in enumerate(iterable_of_pil_images)]
    else:
        prc_pool = Pool(njobs)
        recognized_pages = prc_pool.map(prc_one_image, iterable_of_pil_images)
    ent = time()
    if DEBUG:
        logging.info('Done in ' + str(ent - stt))
    return recognized_pages


if __name__ == '__main__':
    pass
    # Исключительно для отладки:
    os.chdir(r'tests\test_imgs&pdfs')
    # images = extract_images_from_files('tst_01.pdf', pages_to_process=[0])
    # recognized_pages = prc_all_images(images)
    # gray_np_image = cv2.cvtColor(cv2.imread('test_prepated_image_01.png'), cv2.COLOR_BGR2GRAY)
    # filled_cells, horizontal_coords, vertical_coords = prc_one_prepared_image(gray_np_image)
    # Запишем в дамп, чтобы запускалось быстрее
    # with open(r'test_dump.pickle', 'wb') as f:
    #     pickle.dump((gray_np_image, filled_cells, horizontal_coords, vertical_coords), f)
    with open(r'test_dump.pickle', 'rb') as f:
        (gray_np_image, filled_cells, horizontal_coords, vertical_coords) = pickle.load(f)
    filled_cells = feature_qt(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    print(filled_cells)

