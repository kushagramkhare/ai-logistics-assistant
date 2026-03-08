import time
import requests
import io
import fitz
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def f_pdf_content(pdfUrl, max_seconds=3):
    start_time = time.monotonic()

    headers = {"User-Agent": "Mozilla/5.0"}

    # --- STEP 1: Download with timeout ---
    response = requests.get(pdfUrl, headers=headers, timeout=3)
    pdf_bytes = io.BytesIO(response.content)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""

    for page_num in range(len(doc)):

        # ⏱️ HARD TIME LIMIT
        if time.monotonic() - start_time > max_seconds:
            print("⏭️ PDF too slow, skipping:", pdfUrl)
            break

        page = doc[page_num]

        # --- Try text extraction ---
        text = page.get_text().strip()
        if text:
            full_text += text + "\n"
            continue

        # --- OCR fallback ---
        pix = page.get_pixmap(dpi=200)  # 200 is faster than 300
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        ocr_text = pytesseract.image_to_string(
            img, config="--oem 3 --psm 6"
        )

        full_text += ocr_text + "\n"

    doc.close()
    return full_text