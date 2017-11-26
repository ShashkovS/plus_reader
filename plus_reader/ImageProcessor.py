import cv2
import logging
import numpy as np
from bisect import bisect_left
from cell_recognizer import ext_find_filled_cells

FILL_COLOR = np.array([[[0, 255, 255]]], dtype=np.uint8)
BORDER_COLOR = np.array([[[0, 0, 255]]], dtype=np.uint8)
BORDER_WIDTH = 2
MAX_SIZE = 800
DEBUG = False
RESIZE_TO1 = 2400
RESIZE_TO2 = 1600
DEFAULT_BW_THRESHOLD = 230

def _dft_unmark_useless_cells(filled_cells):
    return filled_cells


def _dft_remove_useless_cells(filled_cells):
    return filled_cells


class ImageProcessor():
    def __init__(self, image, *,
                 show_borders=True,
                 black_threshold=DEFAULT_BW_THRESHOLD,
                 unmark_useless_cells_func=None):
        self.black_threshold = black_threshold
        self.unmark_useless_cells_func = unmark_useless_cells_func
        self.BW = BORDER_WIDTH if show_borders else 0
        if isinstance(image, np.ndarray):
            self.np_orig_image = image
        # elif isinstance(image, PIL_Image):
        else:
            self.np_orig_image = np.array(image.convert("L"))
        # else:
        #     raise ValueError('Image should be instance of PIL.Image or numpy.array')
        # Конвертим в ЧБ
        for resize_to in (RESIZE_TO1, RESIZE_TO2):
            ch, cw, *_ = self.np_orig_image.shape
            if max(ch, cw) > resize_to:
                nch, ncw = ch * resize_to//max(ch, cw), cw * resize_to//max(ch, cw)
                self.np_orig_image = cv2.resize(self.np_orig_image, (ncw, nch), interpolation=cv2.INTER_AREA)
        if self.np_orig_image.ndim == 3:
            self.gray_np_image = cv2.cvtColor(self.np_orig_image / 255, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_np_image = self.np_orig_image.copy()
            # Сохраняем размеры
        self.H, self.W, *_ = self.gray_np_image.shape
        # Выравниваем
        self.gray_np_image = _align_image(self.gray_np_image)
        # Немного размываем, чтобы смыть тонкие шумы
        self.gray_np_image = _blur_image(self.gray_np_image)
        # Удаляем мусор
        self.gray_np_image = remove_background(self.gray_np_image)
        # Делаем мастшабную обработку
        self.bitmap_lines_filled_cells_and_marking()

    def find_filled_cells(self):
        # Делаем первичную разметку
        self.filled_cells = ext_find_filled_cells(self.image_without_lines, self.coords_of_horiz_lns, self.coords_of_vert_lns)
        # Возможно, отметки с некоторых ячеек нужно убрать, так как они никому не нужны
        if self.unmark_useless_cells_func:
            try:
                self.filled_cells = self.unmark_useless_cells_func(self.filled_cells)
            except Exception as e:
                logging.error(e)

    def bitmap_lines_filled_cells_and_marking(self):
        # Создаём версию в ЧБ
        self.create_bitmap()
        # Делаем весь процессинг
        self.find_lines_coords()
        # Распознаём ячейки
        self.find_filled_cells()
        # Делаем первичную маркировку распознанных ячеек
        self.initial_mark_filled_cells()

    def create_bitmap(self):
        # Конвертим в ЧБ
        self.bitmap_np_image = _img_to_bitmap_np(self.gray_np_image, self.black_threshold)
        self.bitmap_np_image = _remove_dots_from_image(self.bitmap_np_image)

    def find_lines_coords(self):
        # Определяем линии таблицы
        self.horizontal_lines = _find_lines_on_image(self.bitmap_np_image, 'horizontal')
        self.vertical_lines = _find_lines_on_image(self.bitmap_np_image, 'vertical')
        # Удалаяем из изображения линии таблицы
        self.image_without_lines = _remove_table_lines_on_image(self.bitmap_np_image, self.horizontal_lines, self.vertical_lines)
        # Определяем координаты этих линий
        self.coords_of_horiz_lns, self.coords_of_vert_lns = _calcutale_lines_coords(self.horizontal_lines, self.vertical_lines)

    def toggle_highlight_cell(self, x_vert_ind, y_horiz_ind, *, initial_mode=False):
        """Маркировка и снятие маркировки ячейки"""
        FILL_COLOR = np.array([0, 255, 255], dtype=np.uint8)
        ALPHA = .3
        bw = self.BW
        if x_vert_ind < 0 or x_vert_ind >= len(self.coords_of_vert_lns) - 1:
            logging.error('Cell x-index out of range: {}, cor.range: {}-{}'.format(x_vert_ind, 0, len(self.coords_of_vert_lns) - 1))
            return
        if y_horiz_ind < 0 or y_horiz_ind >= len(self.coords_of_horiz_lns) - 1:
            logging.error('Cell y-index out of range: {}, cor.range: {}-{}'.format(y_horiz_ind, 0, len(self.coords_of_horiz_lns) - 1))
            return
        h1, h2, v1, v2 = self.coords_of_horiz_lns[y_horiz_ind] + bw, self.coords_of_horiz_lns[y_horiz_ind + 1] - bw, \
                         self.coords_of_vert_lns[x_vert_ind] + bw, self.coords_of_vert_lns[x_vert_ind + 1] - bw
        if not initial_mode ^ self.filled_cells[y_horiz_ind][x_vert_ind]:  # initial_mode инвертирует логику
            self.bgr_img_with_highlights[h1:h2, v1:v2, :] = (1 - ALPHA) * self.bgr_img_with_highlights[h1:h2, v1:v2, :] + ALPHA * FILL_COLOR  # GBR, so it's yellow
        elif not initial_mode and self.filled_cells[y_horiz_ind][x_vert_ind]:
            self.bgr_img_with_highlights[h1:h2, v1:v2, :] = (self.bgr_img_with_highlights[h1:h2, v1:v2, :] - ALPHA * FILL_COLOR) / (1 - ALPHA)
        if not initial_mode:
            self.filled_cells[y_horiz_ind][x_vert_ind] ^= True

    def initial_mark_filled_cells(self):
        """Выполнить первичную маркировку всех заполненных ячеек"""
        # Делаем цветную копию для пометок
        self.bgr_img_with_highlights = cv2.cvtColor(self.gray_np_image, cv2.COLOR_GRAY2BGR)
        # Помечаем ячейки
        for y_horiz_ind in range(len(self.coords_of_horiz_lns) - 1):
            for x_vert_ind in range(len(self.coords_of_vert_lns) - 1):
                self.toggle_highlight_cell(x_vert_ind, y_horiz_ind, initial_mode=True)
        # Добавляем распознанные границы красным:
        if self.BW:
            bw = self.BW
            for v_b in self.coords_of_vert_lns:
                v1, v2 = v_b - bw, v_b + bw
                self.bgr_img_with_highlights[:, v1:v2, :] = .7 * self.bgr_img_with_highlights[:, v1:v2, :] + .3 * BORDER_COLOR
            for h_b in self.coords_of_horiz_lns:
                h1, h2 = h_b - bw, h_b + bw
                self.bgr_img_with_highlights[h1:h2, :, :] = .7 * self.bgr_img_with_highlights[h1:h2, :, :] + .3 * BORDER_COLOR

    def coord_to_cell(self, x, y, w, h):
        real_x_coord, real_y_coord = self.window_coords_to_image_coords(x, y, w, h)
        x_ind = bisect_left(self.coords_of_vert_lns, real_x_coord) - 1
        y_ind = bisect_left(self.coords_of_horiz_lns, real_y_coord) - 1
        if x_ind < 0 or y_ind < 0 or x_ind >= len(self.coords_of_vert_lns) - 1 or y_ind >= len(self.coords_of_horiz_lns) - 1:
            res = None
        else:
            res = [x_ind, y_ind]
        logging.info(str(res))
        return res

    def window_coords_to_image_coords(self, x, y, w, h):
        real_x_coord = x * self.W / w
        real_y_coord = y * self.H / h
        return real_x_coord, real_y_coord


    def to_bin(self):
        """Конвертнуть текущее состояние кортинки в бинарную строку для передачи в QT"""
        retval, bin_image = cv2.imencode('.png', self.bgr_img_with_highlights)
        return bin_image


def _align_image(img):
    """Выровнять изображение.
    Идея подхода в следующем: рассмотрим несколько разных поворотов и выберем из них "лучший".
    Лучший — это тот, в котором "горизонтальные" линии занимают меньше всего места по вертикали.
    """
    # TODO: Этот кусок работает дико медленно и занимает большую часть времени
    # TODO: Скорее всего можно определять угол поворота как-нибудь быстрее
    logging.info('Aligning bgr_img_with_highlights...')
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
    M = cv2.getRotationMatrix2D((cols/2,rows/2), best_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, borderValue=255)
    logging.info(str('Best angle = ') + str(best_angle) + str(' Penalty= ') + str(yy[mx]))
    return dst


def _blur_image(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return blur


def _img_to_bitmap_np(img, threshold):
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
    ret = threshold
    ret, ar_bit = cv2.threshold(img, ret, 255, cv2.THRESH_BINARY)
    return ar_bit


def _remove_dots_from_image(gray_np_image):
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
    # TODO Ну треш же...
    CONST_FOR_REALLY_BLACK = 100
    med = cv2.medianBlur(gray_np_image, 75)
    dif = cv2.add(gray_np_image, 255-med)
    dif[dif < CONST_FOR_REALLY_BLACK] = 0
    if DEBUG:
        cv2.imwrite("_med.png", med)
    return dif


def _find_lines_on_image(gray_np_image, direction):
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


def _remove_table_lines_on_image(gray_np_image, horizontal_lines, vertical_lines):
    """Принимает на вход картинку и маску таблицы.
    Размалёвывает плюсы
    """
    BLACK_LINE_THRESHOLD = 175
    table_mask = cv2.min(horizontal_lines, vertical_lines)  # Горизонтальные и вертикальные линии вместе
    table_mask[table_mask>BLACK_LINE_THRESHOLD] = 255
    table_mask[table_mask<=BLACK_LINE_THRESHOLD] = 0
    img_no_lines = (gray_np_image | ~table_mask)  # Убрали из изображения сами линии
    # img_no_lines = _img_to_bitmap_np(img_no_lines)
    # img_no_lines = _remove_dots_from_image(img_no_lines)
    # Замажем чёрным всё, в окрестности чего много точек.
    # Почти все плюсы превратятся в "жирные" кляксы
    # TODO: Здесь мутные константы, которые я подбирал руками для наших кондуитов. Это — треш
    img_no_lines = ~cv2.adaptiveThreshold(img_no_lines, DEFAULT_BW_THRESHOLD, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, -9)
    # Вернём наместро сами плюсы
    img_no_lines = cv2.min(img_no_lines, gray_np_image)
    # Очистим точки в границах таблицы, чтобы не мешались
    img_no_lines |= ~table_mask
    if DEBUG: cv2.imwrite("_plus.png", img_no_lines); cv2.imwrite("_table_mask.png", table_mask);
    return img_no_lines


def _calcutale_lines_coords(horizontal_lines, vertical_lines):
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
