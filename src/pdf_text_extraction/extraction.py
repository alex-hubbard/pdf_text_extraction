from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import PyPDF2

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
except ImportError:
    convert_from_path = None
    pdfinfo_from_path = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    enable_ocr: bool = True
    ocr_chunk_size: int = 5
    ocr_dpi: int = 200
    min_chars_per_page: int = 30
    tesseract_cmd: Optional[str] = None
    tessdata_prefix: Optional[str] = None
    poppler_path: Optional[str] = None
    llm_enabled: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    llm_timeout: int = 60
    llm_system_prompt: str = (
        "You extract structured information from text. "
        "Return only valid JSON."
    )
    llm_prompt_template: str = "Extract key fields from the document:\n\n{document_text}"


def _ocr_dependencies_available() -> bool:
    return bool(pytesseract and convert_from_path and Image)


def configure_ocr_environment(config: ExtractionConfig) -> None:
    if config.tesseract_cmd and pytesseract:
        logger.info("Setting pytesseract.tesseract_cmd.")
        pytesseract.tesseract_cmd = config.tesseract_cmd

    if config.tessdata_prefix:
        tessdata_path = Path(config.tessdata_prefix)
        if tessdata_path.is_dir():
            os.environ["TESSDATA_PREFIX"] = str(tessdata_path.resolve())
            logger.info("Set TESSDATA_PREFIX.")
        else:
            logger.warning("TESSDATA_PREFIX directory does not exist: %s", tessdata_path)


def get_pdf_page_count_pdf2image(pdf_path: Path, poppler_path: Optional[str] = None) -> int:
    if not pdfinfo_from_path:
        logger.warning("pdfinfo_from_path not available. Page count fallback disabled.")
        return 0
    try:
        info = pdfinfo_from_path(str(pdf_path), poppler_path=poppler_path)
        return int(info.get("Pages", 0))
    except Exception as exc:
        logger.error("Error getting page count for %s: %s", pdf_path.name, exc)
        return 0


def ocr_pdf_pages(
    pdf_path: Path,
    total_pages: int,
    config: ExtractionConfig,
) -> Optional[str]:
    if not _ocr_dependencies_available():
        logger.error("OCR dependencies not available. Skipping OCR.")
        return None
    if total_pages == 0:
        logger.warning("Skipping OCR for %s because page count is 0.", pdf_path.name)
        return ""

    ocr_text_content = ""
    logger.info(
        "Attempting OCR for %s (%s pages) in chunks of %s at %s DPI.",
        pdf_path.name,
        total_pages,
        config.ocr_chunk_size,
        config.ocr_dpi,
    )

    for page_start_num in range(1, total_pages + 1, config.ocr_chunk_size):
        page_end_num = min(page_start_num + config.ocr_chunk_size - 1, total_pages)
        logger.info(
            "Processing pages %s to %s of %s for %s.",
            page_start_num,
            page_end_num,
            total_pages,
            pdf_path.name,
        )

        images = []
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=config.ocr_dpi,
                first_page=page_start_num,
                last_page=page_end_num,
                poppler_path=config.poppler_path,
            )

            for i, image in enumerate(images):
                current_page_for_log = page_start_num + i
                logger.info(
                    "OCRing page %s (image %s/%s).",
                    current_page_for_log,
                    i + 1,
                    len(images),
                )
                try:
                    page_text_content = pytesseract.image_to_string(image, lang="eng")
                    ocr_text_content += page_text_content + "\n\n--- Page Break ---\n\n"
                except pytesseract.TesseractNotFoundError:
                    logger.error("Tesseract executable not found.")
                    return None
                except pytesseract.TesseractError as exc:
                    logger.error("Tesseract error on page %s: %s", current_page_for_log, exc)
                except Exception as exc:
                    logger.warning("Error OCRing page %s: %s", current_page_for_log, exc)
                finally:
                    if hasattr(image, "close"):
                        image.close()

            del images
            gc.collect()
        except Exception as exc:
            logger.error(
                "Error during PDF to image conversion for %s pages %s-%s: %s",
                pdf_path.name,
                page_start_num,
                page_end_num,
                exc,
            )
            return None

    logger.info("OCR completed for %s. Extracted approx %s chars.", pdf_path.name, len(ocr_text_content))
    return ocr_text_content


def extract_text_from_pdf(pdf_path: Path, config: ExtractionConfig) -> Tuple[Optional[str], str]:
    direct_text_content = ""
    pypdf2_num_pages = 0
    extraction_method_used = "PyPDF2 (Direct)"

    pdf2image_page_count = 0
    if config.enable_ocr:
        pdf2image_page_count = get_pdf_page_count_pdf2image(pdf_path, poppler_path=config.poppler_path)

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            pypdf2_num_pages = len(reader.pages)

            if pypdf2_num_pages == 0 and pdf2image_page_count == 0:
                logger.warning("%s reported as 0 pages. Skipping.", pdf_path.name)
                return None, "No Pages Found"

            if config.enable_ocr and pypdf2_num_pages == 0 and pdf2image_page_count > 0:
                logger.warning(
                    "%s has 0 pages by PyPDF2 but %s by pdfinfo. Attempting OCR.",
                    pdf_path.name,
                    pdf2image_page_count,
                )
                ocr_text = ocr_pdf_pages(pdf_path, total_pages=pdf2image_page_count, config=config)
                if ocr_text is not None:
                    return ocr_text, "OCR (PyPDF2 reported 0 pages)"
                return None, "OCR Failed (PyPDF2 reported 0 pages)"

            logger.info("Reading %s pages from %s using PyPDF2.", pypdf2_num_pages, pdf_path.name)
            for page_num in range(pypdf2_num_pages):
                try:
                    page = reader.pages[page_num]
                    page_text_content = page.extract_text()
                    if page_text_content:
                        direct_text_content += page_text_content + "\n\n--- Page Break ---\n\n"
                except Exception as exc:
                    logger.warning(
                        "Could not extract text from page %s in %s: %s",
                        page_num + 1,
                        pdf_path.name,
                        exc,
                    )

        logger.info("PyPDF2 extracted approx %s chars from %s.", len(direct_text_content), pdf_path.name)

        if not config.enable_ocr:
            return direct_text_content, extraction_method_used

        min_chars_threshold = pypdf2_num_pages * config.min_chars_per_page
        effective_pages_for_ocr = pdf2image_page_count if pdf2image_page_count > 0 else pypdf2_num_pages

        if (
            effective_pages_for_ocr > 0
            and (not direct_text_content or len(direct_text_content.replace("--- Page Break ---", "").strip()) < min_chars_threshold)
        ):
            logger.info(
                "Direct text from %s is minimal. Attempting OCR fallback using %s pages.",
                pdf_path.name,
                effective_pages_for_ocr,
            )
            ocr_text_content = ocr_pdf_pages(pdf_path, total_pages=effective_pages_for_ocr, config=config)
            if ocr_text_content is not None:
                extraction_method_used = "OCR (Fallback)"
                return ocr_text_content, extraction_method_used
            logger.warning("OCR fallback failed for %s. Using PyPDF2 text (if any).", pdf_path.name)
            return (
                direct_text_content if direct_text_content else None,
                extraction_method_used if direct_text_content else "PyPDF2 (OCR Failed)",
            )

        return direct_text_content, extraction_method_used

    except FileNotFoundError:
        logger.error("File not found at %s", pdf_path)
        return None, "File Not Found"
    except PyPDF2.errors.PdfReadError as pdf_err:
        logger.error("Error reading PDF %s with PyPDF2: %s. Attempting OCR.", pdf_path.name, pdf_err)
        if config.enable_ocr:
            effective_pages_for_ocr = pdf2image_page_count if pdf2image_page_count > 0 else 0
            if effective_pages_for_ocr == 0 and pdfinfo_from_path:
                effective_pages_for_ocr = get_pdf_page_count_pdf2image(pdf_path, poppler_path=config.poppler_path)
            ocr_text_content = ocr_pdf_pages(pdf_path, total_pages=effective_pages_for_ocr, config=config)
            if ocr_text_content is not None:
                return ocr_text_content, "OCR (PyPDF2 ReadError)"
        return None, "PyPDF2 ReadError (OCR Failed)"
    except Exception as exc:
        logger.error("Unexpected error processing PDF %s: %s", pdf_path.name, exc)
        return None, "Unexpected Error"


def save_text_to_file(text_content: str, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text_content)
        logger.info("Saved extracted text to %s", output_path)
    except Exception as exc:
        logger.error("Error saving text to %s: %s", output_path, exc)


def move_processed_pdf(pdf_path: Path, destination_dir: Path) -> Optional[Path]:
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / pdf_path.name

        if destination_path.exists():
            timestamp = int(time.time())
            destination_path = destination_dir / f"{pdf_path.stem}_{timestamp}{pdf_path.suffix}"

        shutil.move(str(pdf_path), destination_path)
        logger.info("Moved processed PDF to %s", destination_path)
        return destination_path
    except Exception as exc:
        logger.error("Failed to move processed PDF %s: %s", pdf_path, exc)
        return None


def process_path(
    input_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    move_processed: bool = False,
    processed_dir: Optional[Path] = None,
    llm_output_dir: Optional[Path] = None,
) -> None:
    configure_ocr_environment(config)

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
    else:
        pdf_files = [input_path]

    if not pdf_files:
        logger.warning("No PDF files found at %s", input_path)
        return

    for pdf_file in pdf_files:
        if not pdf_file.is_file():
            continue
        logger.info("Processing %s", pdf_file.name)
        text, method = extract_text_from_pdf(pdf_file, config)
        if text:
            output_path = output_dir / f"{pdf_file.stem}.txt"
            save_text_to_file(text, output_path)
            logger.info("Extraction method used for %s: %s", pdf_file.name, method)
            if config.llm_enabled and llm_output_dir:
                llm_output_dir.mkdir(parents=True, exist_ok=True)
                json_output_path = llm_output_dir / f"{pdf_file.stem}.json"
                llm_json = extract_json_with_llm(text, config)
                if llm_json:
                    save_text_to_file(llm_json, json_output_path)
                    logger.info("Saved LLM JSON output to %s", json_output_path)
                else:
                    logger.warning("LLM extraction failed for %s", pdf_file.name)
            if move_processed:
                move_processed_pdf(pdf_file, processed_dir or (input_path / "processed"))
        else:
            logger.warning("No text extracted for %s (%s).", pdf_file.name, method)


def extract_json_with_llm(text_content: str, config: ExtractionConfig) -> Optional[str]:
    if not config.llm_enabled:
        logger.info("LLM extraction disabled. Skipping.")
        return None
    if not OpenAI:
        logger.error("OpenAI package not installed. Install with extras: pip install -e '.[llm]'.")
        return None

    api_key = config.llm_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set and llm_api_key not provided.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        prompt = config.llm_prompt_template.format(document_text=text_content)
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": config.llm_system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=config.llm_timeout,
        )
        if response and response.choices:
            return response.choices[0].message.content.strip()
        return None
    except Exception as exc:
        logger.error("LLM extraction failed: %s", exc)
        return None
