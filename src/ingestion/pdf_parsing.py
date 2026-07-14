# src/ingestion/pdf_parsing.py
import re
import pdfplumber
from io import BytesIO
from loguru import logger


def clean_header(header: str) -> str:
    """
    Table headers in this document span multiple physical lines and mix in
    garbled non-English glyphs from a stacked bilingual layout. This keeps
    only clean, all-uppercase English label lines (e.g. "SUBJECT", "CODE",
    "THEORY") and discards the rest, collapsing them into one clean label.
    """
    if not header:
        return ""

    lines = header.split("\n")
    clean_lines = [
        line.strip() for line in lines
        if re.fullmatch(r"[A-Z][A-Z\s/]*", line.strip())
    ]
    return " ".join(clean_lines).strip()


def row_to_sentence(headers: list[str], row: list[str]) -> str:
    """
    Converts a single table row into an explicit, unambiguous, single-line
    sentence by pairing each cleaned column header with its cell value.
    Skips rows that don't have enough valid pairs to be real data (guards
    against merged/garbled rows like a stray "RESULT: PASS" footer row).
    """
    parts = []
    for header, value in zip(headers, row):
        clean = clean_header(header)
        value = (value or "").strip().replace("\n", " ")
        if not clean or not value:
            continue
        parts.append(f"{clean}: {value}")

    if len(parts) < 3:  # Too few valid fields — likely not a genuine data row
        return ""

    return ". ".join(parts) + "."


def extract_tables_as_sentences(page) -> tuple[list[str], list]:
    """
    Detects tables on a page, converts each data row into a standalone
    single-line sentence, and returns both the sentences and the table
    bounding boxes (so the caller can exclude that region from plain
    text extraction and avoid double-counting the same data).
    """
    sentences = []
    bboxes = []

    tables = page.find_tables()
    for table in tables:
        bboxes.append(table.bbox)
        extracted = table.extract()
        if not extracted or len(extracted) < 2:
            continue

        headers = extracted[0]
        for row in extracted[1:]:
            sentence = row_to_sentence(headers, row)
            if sentence:
                sentences.append(sentence)

    return sentences, bboxes


def extract_text_and_tables(file_bytes: bytes) -> str:
    """
    Extracts regular paragraph text and table-derived sentences from a PDF,
    excluding table regions from the plain-text pass so table data isn't
    represented twice (once as a flat row, once as a labeled sentence).
    """
    combined_text = ""

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            table_sentences, bboxes = extract_tables_as_sentences(page)

            # Crop out each detected table's region before extracting plain text
            working_page = page
            for bbox in bboxes:
                working_page = working_page.outside_bbox(bbox)

            page_text = working_page.extract_text() or ""
            if page_text.strip():
                combined_text += page_text.strip() + "\n"

            if table_sentences:
                logger.info(f"Page {page_num}: extracted {len(table_sentences)} clean table row sentence(s).")
                combined_text += "\n".join(table_sentences) + "\n"

    return combined_text