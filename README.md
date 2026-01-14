## Generic PDF Text Extraction

This repo provides a lightweight, reusable Python package for extracting text
from PDFs. It uses direct text extraction with PyPDF2 and falls back to OCR
when needed (if optional OCR dependencies are installed).

### Features
- Direct text extraction with PyPDF2
- OCR fallback using `pdf2image` + `pytesseract` + `Pillow`
- Directory or single-file processing
- Configurable OCR settings and paths
- Simple CLI entrypoint

### Install (editable)
```
cd pdf_text_extraction
pip install -e .
```

### Install with OCR extras
```
pip install -e ".[ocr]"
```

### Install with LLM extras
```
pip install -e ".[llm]"
```

### CLI usage
```
pdf-text-extract /path/to/file_or_dir --output-dir /path/to/text
```

Optional flags:
- `--no-ocr` disables OCR fallback.
- `--tesseract-cmd` sets the tesseract binary path.
- `--tessdata-prefix` sets TESSDATA_PREFIX.
- `--poppler-path` sets poppler binary path (for pdf2image).
- `--llm` enables LLM JSON extraction.
- `--llm-output-dir` sets a directory for JSON outputs.
- `--llm-prompt-file` points to a template with `{document_text}`.

### Programmatic usage
```
from pathlib import Path
from pdf_text_extraction.extraction import ExtractionConfig, extract_text_from_pdf, extract_json_with_llm

config = ExtractionConfig()
text, method = extract_text_from_pdf(Path("document.pdf"), config)

config.llm_enabled = True
json_text = extract_json_with_llm(text, config)
```
