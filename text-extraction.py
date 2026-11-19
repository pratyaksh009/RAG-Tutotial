import os
import cv2
import fitz  # PyMuPDF
import openpyxl
import pytesseract
from docx import Document
from faster_whisper import WhisperModel
from PIL import Image


def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img).strip()


# 1. PDF: Extract native text; run OCR on pages with little/no text (scanned PDFs)
def extract_from_pdf(pdf_path: str, min_text_chars: int = 20) -> str:
    doc = fitz.open(pdf_path)
    output = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) < min_text_chars:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = ocr_image(img)
        output.append(f"--- Page {page_num} ---\n{text}")
    return "\n\n".join(output)


# 2. DOCX: Extract paragraphs, tables, and OCR any embedded images
def extract_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                parts.append(row_text)

    for idx, rel in enumerate(doc.part.rels.values(), start=1):
        if "image" in rel.target_ref:
            img_bytes = rel.target_part.blob
            temp_img = f"_temp_docx_img_{idx}"
            with open(temp_img, "wb") as f:
                f.write(img_bytes)
            try:
                ocr_text = ocr_image(Image.open(temp_img))
                if ocr_text:
                    parts.append(f"[DOCX Image {idx} OCR]: {ocr_text}")
            finally:
                if os.path.exists(temp_img):
                    os.remove(temp_img)

    return "\n".join(parts)


# 3. Excel (.xlsx): Extract cell values + OCR any embedded images
def extract_from_excel(xlsx_path: str) -> str:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"=== Sheet: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                parts.append(" | ".join("" if cell is None else str(cell) for cell in row))

        for idx, img in enumerate(getattr(ws, "_images", []), start=1):
            pil_img = Image.open(img._data())
            ocr_text = ocr_image(pil_img)
            if ocr_text:
                parts.append(f"[{sheet_name} Image {idx} OCR]: {ocr_text}")

    return "\n".join(parts)


# 4. Video: Sample frames every N seconds and run OCR (deduplicating identical consecutive text)
def extract_text_from_video_ocr(video_path: str, sample_every_sec: float = 2.0) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sample_every_sec))

    frame_idx = 0
    last_text = ""
    transcript_lines = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = ocr_image(Image.fromarray(rgb))
            if text and text != last_text:
                transcript_lines.append(f"[{timestamp:.1f}s] {text}")
                last_text = text
        frame_idx += 1

    cap.release()
    return "\n\n".join(transcript_lines)


# 5. Audio: Speech-to-text transcription (Whisper) rather than OCR
def extract_text_from_audio(audio_path: str, model_size: str = "base") -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments).strip()


if __name__ == "__main__":
    # Example calls:
    # print(extract_from_pdf("sample.pdf"))
    # print(extract_from_docx("sample.docx"))
    # print(extract_from_excel("sample.xlsx"))
    # print(extract_text_from_video_ocr("sample.mp4", sample_every_sec=2.0))
    # print(extract_text_from_audio("sample.mp3"))
    pass
