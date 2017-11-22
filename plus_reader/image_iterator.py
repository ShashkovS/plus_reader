# -*- coding: utf-8 -*-.
import struct
import io
import zlib
import os
import PyPDF2  # pip install --upgrade pypdf2
from PIL import Image  # pip install --upgrade pillow


def _tiff_header_for_CCITT(width: int, height: int, img_size: int, CCITT_group=4) -> bytes:
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
                       279, 4, 1, img_size,  # StripByteCounts, LONG, 1, size of bgr_img_with_highlights
                       0  # last IFD
                       )


def yield_images_from_pdf(pdf_filename, pages_to_process=None):
    """Изображение в pdf согласно стандарту может быть закодировано одним из следующих способов:
    ASCIIHexDecode ASCII85Decode LZWDecode FlateDecode RunLengthDecode CCITTFaxDecode JBIG2Decode JPXDecode
    Для каждого их них свой способ декодирования. Также после кодирования результат может быть сжат.
    Тогда указывается ещё и DCTDecode
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
                if not xObject[obj]['/Subtype'] == '/Image':
                    continue
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
                # Данные могут быть зазипованы:
                if image_codec == '/FlateDecode':  # png
                    img = Image.frombytes(mode, size, data)
                elif image_codec == '/DCTDecode':  # jpg
                    img = Image.open(io.BytesIO(data))
                elif image_codec == '/JPXDecode':  # jp2
                    img = Image.open(io.BytesIO(data))
                elif image_codec == '/CCITTFaxDecode':  # CCITT4
                    tiff_header = _tiff_header_for_CCITT(width, height, img_size)
                    img = Image.open(io.BytesIO(tiff_header + data))
                elif image_codec == ['/FlateDecode',
                                     '/DCTDecode']:  # JPEG compression was applied first, and then it was Deflated
                    img = Image.open(io.BytesIO(zlib.decompress(data)))
                else:
                    # TODO: тогда можно попробовать сконвертировать изображение в png при помощи imagemagic, если он есть в системе
                    raise ValueError('Unknown codec in pdf: ' + image_codec)
                yield img


def extract_images_from_files(filenames, pages_to_process=None):
    """Генератор, извлекающий все изображения из pdf-файла или картинки.
    Либо из списка pdf-файлов и картинок.
    Возвращающий итератор на извлечённые изображения в формате numpy.array
    """
    if isinstance(filenames, str):
        files_to_process = [filenames]
    else:
        files_to_process = list(filenames)
    # Проверяем, что все перечисленные файлы существуют
    for filename in files_to_process:
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
    # Ок, если переданы номера страниц, то это должен быть в точности один pdf-файл
    if pages_to_process is not None:
        if len(files_to_process) != 1 or not files_to_process[0].lower().endswith('.pdf'):
            raise ValueError('There must be exectly one pdf file for using pages_to_process parameter')
    # Окучиваем все файлы
    for filename in files_to_process:
        if filename.lower().endswith('.pdf'):
            yield from yield_images_from_pdf(filename, pages_to_process)
        else:
            # If it is not a pdf, than it must be an bgr_img_with_highlights
            img = Image.open(filename)
            yield img
