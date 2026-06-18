"""
utils/pdf_reader.py

Pulls raw text out of an uploaded PDF using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF


def extract_text(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file-like object
    (e.g. the object returned by st.file_uploader).

    Raises ValueError if the file is empty or has no extractable
    text (e.g. it's a scanned image with no OCR layer) so the caller
    can show a clean error instead of crashing later in the pipeline.
    """
    pdf_bytes = pdf_file.read()
    if not pdf_bytes:
        raise ValueError("Uploaded PDF appears to be empty.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = [page.get_text() for page in doc]
    doc.close()

    full_text = "\n".join(pages)

    if not full_text.strip():
        raise ValueError(
            "No extractable text found in this PDF. It may be a "
            "scanned image — OCR is out of scope for this MVP."
        )

    return full_text