"""Read one uploaded chat document without granting filesystem access."""
from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

from charset_normalizer import from_bytes
from pypdf import PdfReader


_WORD = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def _read_pdf(content):
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise ValueError('PDF 文件损坏或格式无效，无法读取') from error
    if reader.is_encrypted:
        raise ValueError('PDF 已加密，请先解密后再添加')
    try:
        text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
    except Exception as error:
        raise ValueError('PDF 文字提取失败：' + str(error)) from error
    if not text.strip():
        raise ValueError('PDF 未包含可提取的文字；扫描件需先进行 OCR')
    return text


def _docx_blocks(element):
    if element.tag == _WORD + 'p':
        yield ''.join(node.text or '' if node.tag == _WORD + 't' else
                      '\t' if node.tag == _WORD + 'tab' else
                      '\n' if node.tag in (_WORD + 'br', _WORD + 'cr') else ''
                      for node in element.iter())
    elif element.tag == _WORD + 'tr':
        yield '\t'.join('\n'.join(_docx_blocks(cell)) for cell in element if cell.tag == _WORD + 'tc')
    else:
        for child in element:
            yield from _docx_blocks(child)


def _read_docx(content):
    try:
        with ZipFile(BytesIO(content)) as archive:
            if '[Content_Types].xml' not in archive.namelist():
                raise ValueError('DOCX 文件缺少内容类型信息')
            document = ElementTree.fromstring(archive.read('word/document.xml'))
    except Exception as error:
        raise ValueError('DOCX 文件损坏或格式无效，无法读取') from error
    body = document.find(_WORD + 'body')
    if document.tag != _WORD + 'document' or body is None:
        raise ValueError('DOCX 缺少有效的文档正文')
    text = '\n'.join(_docx_blocks(body))
    if not text.strip():
        raise ValueError('DOCX 未包含可提取的文字')
    return text


def _is_text(text):
    return not any((ord(char) < 32 and char not in '\t\n\r\f') or 0x7f <= ord(char) < 0xa0 for char in text)


def _read_text(content):
    for bom, encoding in ((BOM_UTF32_LE, 'utf-32'), (BOM_UTF32_BE, 'utf-32'),
                          (BOM_UTF16_LE, 'utf-16'), (BOM_UTF16_BE, 'utf-16'), (BOM_UTF8, 'utf-8-sig')):
        if content.startswith(bom):
            try:
                text = content.decode(encoding)
            except UnicodeError as error:
                raise ValueError('文本编码数据不完整，无法读取') from error
            break
    else:
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            if any(byte < 32 and byte not in (9, 10, 12, 13) for byte in content):
                raise ValueError('文件包含二进制数据，不是支持的文本文档') from None
            # Restrict the detector to the Chinese encodings supported here;
            # short Chinese text otherwise often gets classified as Korean.
            match = from_bytes(content, cp_isolation=['gb18030', 'big5']).best()
            if match is None:
                raise ValueError('无法识别文本编码，请转换为 UTF-8 后再添加') from None
            text = str(match)
    if not _is_text(text):
        raise ValueError('文件包含二进制数据，不是支持的文本文档')
    return text


def read_chat_document(name, content):
    if not isinstance(name, str) or not name.strip() or '\x00' in name:
        raise ValueError('请提供有效的文档名称')
    extension = name.rsplit('.', 1)[-1].casefold() if '.' in name else ''
    if extension == 'pdf' or content.startswith(b'%PDF-'):
        text = _read_pdf(content)
    elif extension == 'docx':
        text = _read_docx(content)
    elif extension == 'doc':
        raise ValueError('不支持旧版 DOC，请另存为 DOCX 或文本后再添加')
    else:
        text = _read_text(content)
    return {'name': name, 'size': len(content), 'text': text}
