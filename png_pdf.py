import pytesseract
from PIL import Image
import requests
from io import BytesIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Path to tesseract engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image URL
url = "https://i.pinimg.com/564x/75/79/51/7579517d81b9ad2dfee4210f84126284.jpg"

# Download image
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# OCR
text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")

#print("---- Extracted Text ----")
#print(text)

# 🔗 PDF URL
pdf_url = "https://www.iitj.ac.in/PageImages/Gallery/03-2025/AcademicCalendarPGCAIMLprogram-638780656840856504.pdf"

# Download PDF
headers = {"User-Agent": "Mozilla/5.0"}  # helps if site blocks bots
response = requests.get(pdf_url, headers=headers)
pdf_bytes = io.BytesIO(response.content)

# Open PDF from memory
doc = fitz.open(stream=pdf_bytes, filetype="pdf")

full_text = ""

for page_num in range(len(doc)):
    page = doc[page_num]

    # --- Step 1: Try direct text extraction ---
    text = page.get_text().strip()

    if text:
        print(f"Page {page_num+1}: Text layer found ✅")
        full_text += f"\n\n--- Page {page_num+1} (Text) ---\n{text}"

    else:
        print(f"Page {page_num+1}: No text layer → Using OCR 🔍")

        # --- Step 2: Convert page to image ---
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        # --- Step 3: OCR ---
        ocr_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        full_text += f"\n\n--- Page {page_num+1} (OCR) ---\n{ocr_text}"

doc.close()

print("\n\n========== FINAL EXTRACTED TEXT ==========\n")
print(full_text)