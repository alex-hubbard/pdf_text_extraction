from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pdf_text_extraction.extraction import ExtractionConfig, process_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files with optional OCR fallback."
    )
    parser.add_argument("input_path", type=Path, help="PDF file or directory of PDFs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_text"),
        help="Directory to write extracted text files.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback.")
    parser.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract binary.")
    parser.add_argument("--tessdata-prefix", type=str, default=None, help="TESSDATA_PREFIX path.")
    parser.add_argument("--poppler-path", type=str, default=None, help="Poppler binary path.")
    parser.add_argument("--ocr-chunk-size", type=int, default=5, help="OCR pages per chunk.")
    parser.add_argument("--ocr-dpi", type=int, default=200, help="DPI for OCR rasterization.")
    parser.add_argument(
        "--min-chars-per-page",
        type=int,
        default=30,
        help="Minimum chars per page before OCR fallback.",
    )
    parser.add_argument(
        "--move-processed",
        action="store_true",
        help="Move successfully processed PDFs into a processed directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Destination directory for processed PDFs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument("--llm", action="store_true", help="Enable LLM extraction.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model.")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key override.")
    parser.add_argument(
        "--llm-prompt-file",
        type=Path,
        default=None,
        help="Prompt template file (use {document_text} placeholder).",
    )
    parser.add_argument(
        "--llm-system-prompt",
        type=str,
        default=None,
        help="Override system prompt.",
    )
    parser.add_argument(
        "--llm-validate",
        action="store_true",
        help="Enable LLM validation of extracted JSON.",
    )
    parser.add_argument(
        "--llm-validation-prompt-file",
        type=Path,
        default=None,
        help="Validation prompt template file (use {document_text} and {extracted_json}).",
    )
    parser.add_argument(
        "--llm-validation-system-prompt",
        type=str,
        default=None,
        help="Override validation system prompt.",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=Path,
        default=None,
        help="Directory to write LLM JSON outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    llm_prompt_template = None
    if args.llm_prompt_file:
        llm_prompt_template = args.llm_prompt_file.read_text(encoding="utf-8")
    llm_validation_prompt_template = None
    if args.llm_validation_prompt_file:
        llm_validation_prompt_template = args.llm_validation_prompt_file.read_text(encoding="utf-8")

    config = ExtractionConfig(
        enable_ocr=not args.no_ocr,
        ocr_chunk_size=args.ocr_chunk_size,
        ocr_dpi=args.ocr_dpi,
        min_chars_per_page=args.min_chars_per_page,
        tesseract_cmd=args.tesseract_cmd,
        tessdata_prefix=args.tessdata_prefix,
        poppler_path=args.poppler_path,
        llm_enabled=args.llm,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_system_prompt=args.llm_system_prompt or ExtractionConfig.llm_system_prompt,
        llm_prompt_template=llm_prompt_template or ExtractionConfig.llm_prompt_template,
        llm_validation_enabled=args.llm_validate,
        llm_validation_system_prompt=(
            args.llm_validation_system_prompt or ExtractionConfig.llm_validation_system_prompt
        ),
        llm_validation_prompt_template=(
            llm_validation_prompt_template or ExtractionConfig.llm_validation_prompt_template
        ),
    )

    process_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config=config,
        move_processed=args.move_processed,
        processed_dir=args.processed_dir,
        llm_output_dir=args.llm_output_dir,
    )
