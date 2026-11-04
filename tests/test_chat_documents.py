"""Chat uploads read only the supplied bytes, independently of local-file tools."""
import asyncio
from codecs import BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from io import BytesIO
import threading
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from modloader.app import Controller, build_app
from modloader.chat_documents import read_chat_document


def pdf_bytes(*texts, encrypted=False, scanned=False):
    writer = PdfWriter()
    for text in texts or ('',):
        page = writer.add_blank_page(width=400, height=300)
        stream = DecodedStreamObject()
        if scanned:
            picture = DecodedStreamObject()
            picture.set_data(b'\xff\xff\xff')
            picture.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'),
                            NameObject('/Width'): NumberObject(1), NameObject('/Height'): NumberObject(1),
                            NameObject('/ColorSpace'): NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'): NumberObject(8)})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/XObject'): DictionaryObject({NameObject('/Im0'): picture})})
            stream.set_data(b'q 100 0 0 100 20 20 cm /Im0 Do Q')
        else:
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                                     NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            stream.set_data(f'BT /F1 12 Tf 20 250 Td ({escaped}) Tj ET'.encode('ascii'))
        page.replace_contents(stream)
    if encrypted:
        writer.encrypt('fixture-password')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def docx_bytes(document=None):
    def paragraph(text):
        return '<w:p><w:r><w:t>' + escape(text) + '</w:t></w:r></w:p>'
    body = (paragraph('文档标题 & 模型参数') + '<w:tbl><w:tr><w:tc>' + paragraph('上下文') + '</w:tc><w:tc>' +
            paragraph('4096') + '</w:tc></w:tr><w:tr><w:tc>' + paragraph('量化') + '</w:tc><w:tc>' +
            paragraph('Q4_K_M') + '</w:tc></w:tr></w:tbl>' +
            '<w:p><w:r><w:t>末尾</w:t><w:tab/><w:t>完整</w:t><w:br/><w:t>保留</w:t></w:r></w:p>')
    document = document if document is not None else ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + '</w:body></w:document>').encode('utf-8')
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr('word/document.xml', document)
    return output.getvalue()


@pytest.mark.parametrize('name,content,text', [
    ('notes.md', '# 模型参数\n上下文：4096\n'.encode('utf-8'), '# 模型参数\n上下文：4096\n'),
    ('main.py', b'print("complete")\n', 'print("complete")\n'),
    ('data.json', '{"参数": 42}'.encode('utf-8-sig'), '{"参数": 42}'),
    ('table.csv', '项目,值\n上下文,4096\n'.encode('utf-8'), '项目,值\n上下文,4096\n'),
    ('empty.txt', b'', ''),
    *[('unicode.txt', bom + 'Unicode 文档 🌟\n'.encode(encoding), 'Unicode 文档 🌟\n')
      for bom, encoding in ((BOM_UTF16_LE, 'utf-16-le'), (BOM_UTF16_BE, 'utf-16-be'),
                            (BOM_UTF32_LE, 'utf-32-le'), (BOM_UTF32_BE, 'utf-32-be'))],
    ('中文.txt', '这是一份中文文档，包含模型参数和文件路径。上下文为 4096，量化使用 Q4_K_M，测试内容保持完整。'.encode('gb18030'),
     '这是一份中文文档，包含模型参数和文件路径。上下文为 4096，量化使用 Q4_K_M，测试内容保持完整。'),
    ('繁體.txt', '繁體中文文件，模型參數與檔案路徑。上下文為 4096，測試內容保持完整。'.encode('big5'),
     '繁體中文文件，模型參數與檔案路徑。上下文為 4096，測試內容保持完整。'),
])
def test_reads_text_and_source_encodings_without_changing_content(name, content, text):
    assert read_chat_document(name, content) == {'name': name, 'size': len(content), 'text': text}


def test_text_is_not_truncated_and_filename_is_not_a_local_file(tmp_path):
    local = tmp_path / 'local.txt'
    local.write_text('不得读取本机内容', encoding='utf-8')
    text = '完整文档行\n' * 40_000 + '最后一行'
    content = text.encode('utf-8')
    assert read_chat_document(str(local), content)['text'] == text
    assert local.read_text(encoding='utf-8') == '不得读取本机内容'


def test_reads_real_pdf_pages_and_docx_paragraph_table_order():
    pdf = pdf_bytes('First page: context 4096.', 'Final page: keep 42 items.')
    result = read_chat_document('parameters.PDF', pdf)
    assert result == {'name': 'parameters.PDF', 'size': len(pdf),
                      'text': 'First page: context 4096.\n\nFinal page: keep 42 items.'}
    # Signature detection still reads a PDF when its original suffix is absent.
    assert read_chat_document('document', pdf)['text'] == result['text']
    docx = docx_bytes()
    assert read_chat_document('参数.DOCX', docx) == {'name': '参数.DOCX', 'size': len(docx),
                    'text': '文档标题 & 模型参数\n上下文\t4096\n量化\tQ4_K_M\n末尾\t完整\n保留'}


@pytest.mark.parametrize('name,content,error', [
    ('broken.pdf', b'%PDF-1.7\nbroken', 'PDF 文件损坏'),
    ('secret.pdf', pdf_bytes('Protected', encrypted=True), 'PDF 已加密'),
    ('scan.pdf', pdf_bytes(scanned=True), '扫描件需先进行 OCR'),
    ('broken.docx', b'not a zip file', 'DOCX 文件损坏'),
    ('broken.docx', docx_bytes(b'<invalid'), 'DOCX 文件损坏'),
    ('body.docx', docx_bytes(b'<document/>'), 'DOCX 缺少有效的文档正文'),
    ('legacy.doc', b'legacy', '不支持旧版 DOC'),
    ('program.bin', b'MZ\x00\x01\x00\x00\xffbinary', '二进制'),
    ('broken.txt', BOM_UTF16_LE + b'a', '文本编码数据不完整'),
    ('', b'text', '文档名称'),
    ('invalid\x00.txt', b'text', '文档名称'),
])
def test_rejects_damaged_encrypted_scanned_and_binary_uploads(name, content, error):
    with pytest.raises(ValueError, match=error):
        read_chat_document(name, content)


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'chat-upload-management-secret')
    try:
        yield value
    finally:
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                base_url='http://testserver',
                                headers={'Authorization': 'Bearer chat-upload-management-secret'}) as value:
        yield value


async def test_attachment_api_accepts_raw_binary_and_uses_management_auth(client, controller):
    route = '/api/chat/attachments/text'
    content = docx_bytes()
    response = await client.post(route, params={'name': '参数 & 样例.docx'}, content=content,
                                 headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 200, response.text
    assert response.json() == read_chat_document('参数 & 样例.docx', content)
    assert controller.conversations.list() == {'items': [], 'active_id': None}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver') as unauthenticated:
        for headers in ({}, {'Authorization': 'Bearer wrong-token'}, {'Authorization': 'Bearer ' + controller.api_key}):
            response = await unauthenticated.post(route, params={'name': 'a.txt'}, content=b'text', headers=headers)
            assert response.status_code == 401
            assert response.json()['error']['type'] == 'authentication_error'
    for params, content, error in (({}, b'text', '文档名称'), ({'name': ''}, b'text', '文档名称'),
                                    ({'name': 'bad.pdf'}, b'not a PDF', 'PDF 文件损坏')):
        response = await client.post(route, params=params, content=content)
        assert response.status_code == 400
        assert error in response.json()['detail']
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)),
                                base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
        assert (await public.post(route, params={'name': 'a.txt'}, content=b'text')).status_code == 404


async def test_document_parsing_does_not_block_management_requests(client, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocked_reader(name, content):
        started.set()
        assert release.wait(5), 'test did not release the document parser'
        return read_chat_document(name, content)

    monkeypatch.setattr('modloader.app.read_chat_document', blocked_reader)
    task = asyncio.create_task(client.post('/api/chat/attachments/text', params={'name': 'a.txt'}, content=b'text'))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        response = await asyncio.wait_for(client.get('/api/conversations'), timeout=2)
        assert response.status_code == 200
    finally:
        release.set()
        result = await task
    assert result.status_code == 200 and result.json()['text'] == 'text'
