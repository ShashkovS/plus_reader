# -*- coding: utf-8 -*-.
import struct
import io
import os
import zlib
import cv2  # pip install --upgrade opencv-python
import numpy as np  # conda install numpy
import PyPDF2  # pip install --upgrade pypdf2
import xlwt  # pip install --upgrade xlwt
from PIL import Image  # pip install --upgrade pillow
from multiprocessing import Pool
from time import time
np.set_printoptions(linewidth=200)

VERBOSE = True
# TODO: Переделать разные принты на logging


def _tiff_header_for_CCITT(width, height, img_size, CCITT_group=4):
    """Функция формирует заголовок TIFF файла, данные изображения которого
    закодированы в CCITT group 4.
    Используется для восстановления изображения из данных pdf-файла,
    в котором в соответствии с форматом вырезаются заголовки
    """
    tiff_header_struct = '<' + '2s' + 'h' + 'l' + 'h' + 'hhll' * 8 + 'h'
    return struct.pack(tiff_header_struct,
                       b'II',  # Byte order indication: Little indian
                       42,  # Version number (always 42)
                       8,  # Offset to first IFD
                       8,  # Number of tags in IFD
                       256, 4, 1, width,  # ImageWidth, LONG, 1, width
                       257, 4, 1, height,  # ImageLength, LONG, 1, lenght
                       258, 3, 1, 1,  # BitsPerSample, SHORT, 1, 1
                       259, 3, 1, CCITT_group,  # Compression, SHORT, 1, 4 = CCITT Group 4 fax encoding
                       262, 3, 1, 0,  # Threshholding, SHORT, 1, 0 = WhiteIsZero
                       273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                       278, 4, 1, height,  # RowsPerStrip, LONG, 1, lenght
                       279, 4, 1, img_size,  # StripByteCounts, LONG, 1, size of image
                       0  # last IFD
                       )


def extract_images_from_pdf(pdf_filename, pages_to_process=None):
    """Генератор, извлекающий все изображения из pdf-файла
    и возвращающий их в формате Pillow Image
    """
    with open(pdf_filename, 'rb') as pdf_file:
        # TODO: Прикрутить обработку всех стандартов:
        # TODO: ASCIIHexDecode ASCII85Decode LZWDecode FlateDecode RunLengthDecode CCITTFaxDecode JBIG2Decode DCTDecode JPXDecode
        # TODO: Вот дока: http://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/pdf_reference_1-7.pdf, стр. 67
        # TODO: Самое сложное — JBIG2Decode, это достаточно новый формат с непростым кодированием
        # TODO: Реализация на js: https://github.com/mozilla/pdf.js/blob/ca936ee0c7ac5baeca76a45dfc5485b3607de290/src/core/jbig2.js
        # TODO: Хорошая реализация на C: http://www.artifex.com/jbig2dec/download/jbig2dec-0.13.tar.gz
        # TODO: Реализация на .NET: https://github.com/devteamexpress/JBig2Decoder.NET
        # TODO: После того, как будут реализованы все стандарты, из этого нужно будет сделать отдельную либу.
        # TODO: В данный момент аналогов нет, будет новьё. И статью на habr о мучениях в процессе
        cond_scan_reader = PyPDF2.PdfFileReader(pdf_file)
        if pages_to_process is None:
            pages_to_process = range(0, cond_scan_reader.getNumPages())
        for i in pages_to_process:  # цикл по всем страницам
            page = cond_scan_reader.getPage(i)  # Получаем текущую страницу
            xObject = page['/Resources']['/XObject'].getObject()  # Извлекаем неё ресурсы
            for obj in xObject:  # Перебираем все объеты, нам нужна картинка
                if xObject[obj]['/Subtype'] == '/Image':
                    # Получаем размер изображения
                    width = xObject[obj]['/Width']
                    height = xObject[obj]['/Height']
                    size = (width, height)
                    # Получаем данные изображения
                    try:
                        data = xObject[obj].getData()
                    except NotImplementedError:
                        data = xObject[obj]._data  # sorry, getData() does not work for CCITTFaxDecode
                    img_size = len(data)
                    # Определяем цветность
                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        mode = "RGB"
                    else:
                        mode = "P"
                    # В зависимости от способа кодирования получаем изображение:
                    image_codec = xObject[obj]['/Filter']
                    if image_codec == '/FlateDecode':  # png
                        img = Image.frombytes(mode, size, data)
                    elif image_codec == '/DCTDecode':  # jpg
                        img = Image.open(io.BytesIO(data))
                    elif image_codec == '/JPXDecode':  # jp2
                        img = Image.open(io.BytesIO(data))
                    elif image_codec == '/CCITTFaxDecode':  # CCITT4
                        tiff_header = _tiff_header_for_CCITT(width, height, img_size)
                        img = Image.open(io.BytesIO(tiff_header + data))
                    elif image_codec == ['/FlateDecode', '/DCTDecode']:  # JPEG compression was applied first, and then it was Deflated
                        img = Image.open(io.BytesIO(zlib.decompress(data)))
                    yield img


def align_image(img):
    """Выровнять изображение.
    Идея подхода в следующем: рассмотрим несколько разных поворотов и выберем из них "лучший".
    Лучший — это тот, в котором "горизонтальные" линии занимают меньше всего места по вертикали.
    """
    print('Aligning image...')
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
    if VERBOSE:
        print('Best angle =', best_angle, ' Penalty=', yy[mx])
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
    for img in extract_images_from_pdf(pdf_filename):
        ar = np.array(img.convert("L"))  # Делаем ч/б
        # TODO: С поворотом здесь какой-то треш. Это должно быть вынесено в extract_images_from_pdf
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
    # TODO: С поворотом здесь какой-то треш. Это должно быть вынесено в extract_images_from_pdf
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
    """Удаляет мелкий сор из изображения"""
    clean1 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    clean2 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    clean3 = cv2.dilate(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    clean = clean1 & clean2 & clean3
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    if VERBOSE: cv2.imwrite("_clean.png", clean)
    return clean


def find_lines_on_image(gray_np_image, direction):
    """Найти длинные линии в ч/б изобажении в формате numpy ndarray.
    Возвращает ч/б изображения с сильно растянутыми и жирными линиями.
    Предполагается, что размер изображения - А4, и таблица занимает
    большую часть страницы. Если это не так, то нужно
    "играть параметрами"
    """
    if direction not in ('horizontal', 'vertical'):
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
        raise ValueError('direction must be "horizontal" or "vertical"')
    # Немного "размажем"
    img_copy = cv2.erode(gray_np_image, cv2.getStructuringElement(cv2.MORPH_RECT, (ERODE_SIZE, ERODE_SIZE)))
    if VERBOSE: cv2.imwrite('_' + direction + "1.png", img_copy)
    # Теперь оставим только то, что представляет собой "длинную" линию
    img_copy = cv2.dilate(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_parm))
    if VERBOSE: cv2.imwrite('_' + direction + "2.png", img_copy)
    # И назад растянем
    img_copy = cv2.erode(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, erode_parm))
    if VERBOSE: cv2.imwrite('_' + direction + "3.png", img_copy)
    img_copy = cv2.dilate(img_copy, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_parm))
    if VERBOSE: cv2.imwrite('_' + direction + "4.png", img_copy)
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
    if VERBOSE: cv2.imwrite("_plus.png", plus); cv2.imwrite("_table_mask.png", table_mask);
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
    print('Finding centers')
    hor = prc(hor_l)
    vert = prc(vert_l)
    if VERBOSE:  # Рисуем получившуюся табличку
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
    # Лично у нас первая и последняя строка, а также нулевой и второй столбец отмечать не нужно
    # TODO: здесь мутный хардкод под наши кондуиты. Это должно быть как-то переделано
    # filled_cells[filled_cells[:, 2] == False, :] = False
    # filled_cells[:, 0] = False
    # filled_cells[0, :] = False
    # filled_cells[-1, :] = False
    return filled_cells


def remove_useless_cells(filled_cells):
    # Лично у нас первая и последняя строка, а также первый столбец не нужны
    # Кроме того, вовсе удалим строки, в которых не заполнена фамилия.
    # TODO: здесь мутный хардкод под наши кондуиты. Это должно быть как-то переделано
    # filled_cells = filled_cells[1:-1, 1:]
    # filled_cells = filled_cells[filled_cells[:, 1] == True, :]
    # filled_cells = np.delete(filled_cells, 1, axis=1)  # Здесь столбец с фамилией
    # print('*'*100)
    # print(filled_cells.astype(int))
    # print('*'*100)
    return filled_cells


def mark_filled_cells(gray_np_image, filled_cells, hor, vert):
    clrd = cv2.cvtColor(gray_np_image, cv2.COLOR_GRAY2BGR)
    clrd_r = clrd.copy()
    for i in range(len(hor) - 1):
        for j in range(len(vert) - 1):
            if filled_cells[i][j]:
                clrd_r[hor[i]:hor[i+1], vert[j]:vert[j+1], :] = [0, 255, 255]  # 0=Blue
                # clrd_r[hor[i]:hor[i+1], vert[j]:vert[j+1], 1] = 255  # 1=Green
                # clrd_r[hor[i]:hor[i+1], vert[j]:vert[j+1], 2] = 255  # 2=Red
    return cv2.addWeighted(clrd, .7, clrd_r, .3, 0)


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
    plus = mark_plus(clean, horizontal_lines, vertical_lines)  #
    filled_cells = find_filled_cells(plus, horizontal_coords, vertical_coords)
    if save_marked_name:
        colored = mark_filled_cells(gray_np_image, unmark_useless_cells(filled_cells), horizontal_coords, vertical_coords)
        cv2.imwrite(save_marked_name, colored)
    return filled_cells


def prc_one_image(pil_image, pgnum=[0], res_list=None):
    """Полностью обработать одну страницу (изображение в формате PIL)"""
    if isinstance(pgnum, list):
        # Используется хук для того, чтобы использовать уникальные номера
        use_pgnum = pgnum[0]
        pgnum[0] += 1
    else:
        use_pgnum = pgnum
    gray_np_image = img_to_bitmap_np(pil_image)
    # TODO: пока безусловное сохранение в save_marked_name — это треш
    filled_cells = prc_one_prepared_image(gray_np_image, save_marked_name="sum_page_{}.png".format(pgnum))
    filled_cells = remove_useless_cells(filled_cells)
    # res_list[pgnum] = filled_cells
    return filled_cells


def prc_all_images(iterable_of_pil_images, njobs=1):
    stt = time()
    if njobs == 1:
        recognized_pages = [prc_one_image(image, pg_num) for pg_num, image in enumerate(iterable_of_pil_images)]
    else:
        prc_pool = Pool(njobs)
        recognized_pages = prc_pool.map(prc_one_image, iterable_of_pil_images)
    ent = time()
    if VERBOSE:
        print('Done in ', ent - stt)
    return recognized_pages


if __name__ == '__main__':
    pass
    # Исключительно для отладки:
    os.chdir(r'tests\test_imgs&pdfs')
    images = extract_images_from_pdf('tst_01.pdf', pages_to_process=[0, 1])
    recognized_pages = prc_all_images(images)


