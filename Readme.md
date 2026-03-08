## Setup Instructions

### 1. Clone repo
git clone ...

### 2. Install Python dependencies
pip install -r backend/req.txt

### 3. Install Tesseract OCR
Download from:
https://github.com/tesseract-ocr/tesseract/releases

Add to PATH

### 4. Set Gemini API key in env 
$env:GOOGLE_API_KEY="your_key"

### 5. Generate vector store
cd backend
python vector_store.py

### 6. Start backend
uvicorn app:app --reload
The backend will start at:http://localhost:8000

### 7. Start frontend
cd frontend
npm install
npm run dev
The frontend will start at:http://localhost:3000
