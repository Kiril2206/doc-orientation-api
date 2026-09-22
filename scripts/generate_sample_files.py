"""Generate synthetic sample documents for automated testing and UI previews."""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "app" / "static" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def draw_invoice(width=1200, height=1600):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try default or standard font
    try:
        font_title = ImageFont.truetype("arial.ttf", 48)
        font_header = ImageFont.truetype("arial.ttf", 28)
        font_bold = ImageFont.truetype("arialbd.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font_title = font_header = font_bold = font_body = font_small = ImageFont.load_default()

    # Decorative header bar
    draw.rectangle([(50, 40), (width - 50, 48)], fill=(37, 99, 235))

    # Company & Title
    draw.text((60, 70), "ACME CLOUD SOLUTIONS", fill=(15, 23, 42), font=font_title)
    draw.text((60, 130), "Enterprise Machine Learning & Document Intelligence", fill=(100, 116, 139), font=font_small)
    
    draw.text((width - 320, 75), "INVOICE", fill=(37, 99, 235), font=font_title)
    draw.text((width - 320, 135), "Invoice #: INV-2026-0881", fill=(71, 85, 105), font=font_body)
    draw.text((width - 320, 165), "Date: September 22, 2026", fill=(71, 85, 105), font=font_body)
    draw.text((width - 320, 195), "Due Date: October 22, 2026", fill=(71, 85, 105), font=font_body)

    # Divider
    draw.line([(50, 240), (width - 50, 240)], fill=(226, 232, 240), width=2)

    # Billing Details
    draw.text((60, 260), "BILLED TO:", fill=(100, 116, 139), font=font_bold)
    draw.text((60, 295), "Global Logistics Systems GmbH", fill=(15, 23, 42), font=font_body)
    draw.text((60, 325), "Tech Park 4, Suite 200", fill=(51, 65, 85), font=font_body)
    draw.text((60, 355), "10115 Berlin, Germany", fill=(51, 65, 85), font=font_body)
    draw.text((60, 385), "VAT ID: DE 394 812 001", fill=(51, 65, 85), font=font_body)

    draw.text((width // 2 + 50, 260), "PAYMENT METHOD:", fill=(100, 116, 139), font=font_bold)
    draw.text((width // 2 + 50, 295), "Bank Transfer (SEPA / Wire)", fill=(15, 23, 42), font=font_body)
    draw.text((width // 2 + 50, 325), "IBAN: DE89 3704 0044 0532 0130 00", fill=(51, 65, 85), font=font_body)
    draw.text((width // 2 + 50, 355), "BIC: COBADEFFXXX", fill=(51, 65, 85), font=font_body)

    # Items Table Header
    y = 440
    draw.rectangle([(50, y), (width - 50, y + 45)], fill=(241, 245, 249))
    draw.text((70, y + 10), "DESCRIPTION", fill=(51, 65, 85), font=font_bold)
    draw.text((width - 450, y + 10), "HOURS / QTY", fill=(51, 65, 85), font=font_bold)
    draw.text((width - 280, y + 10), "RATE", fill=(51, 65, 85), font=font_bold)
    draw.text((width - 150, y + 10), "AMOUNT", fill=(51, 65, 85), font=font_bold)

    # Line Items
    items = [
        ("AWS Production Architecture & ECS Migration", "40 hrs", "$125.00", "$5,000.00"),
        ("ONNX Runtime Orientation Model Optimization", "25 hrs", "$140.00", "$3,500.00"),
        ("RapidOCR Hybrid Verification Integration", "20 hrs", "$130.00", "$2,600.00"),
        ("Gemini Multimodal Live API Fallback Route", "15 hrs", "$130.00", "$1,950.00"),
        ("High-Concurrency Memory & Guardrail Hardening", "10 hrs", "$145.00", "$1,450.00"),
    ]

    y += 55
    for desc, qty, rate, amt in items:
        draw.text((70, y), desc, fill=(30, 41, 59), font=font_body)
        draw.text((width - 430, y), qty, fill=(71, 85, 105), font=font_body)
        draw.text((width - 270, y), rate, fill=(71, 85, 105), font=font_body)
        draw.text((width - 150, y), amt, fill=(15, 23, 42), font=font_bold)
        y += 40
        draw.line([(50, y), (width - 50, y)], fill=(241, 245, 249), width=1)
        y += 15

    # Totals Box
    y += 40
    draw.rectangle([(width - 420, y), (width - 50, y + 170)], outline=(226, 232, 240), width=1, fill=(248, 250, 252))
    draw.text((width - 400, y + 15), "Subtotal:", fill=(71, 85, 105), font=font_body)
    draw.text((width - 180, y + 15), "$14,500.00", fill=(15, 23, 42), font=font_body)
    
    draw.text((width - 400, y + 50), "Tax (19% VAT):", fill=(71, 85, 105), font=font_body)
    draw.text((width - 180, y + 50), "$2,755.00", fill=(15, 23, 42), font=font_body)

    draw.line([(width - 400, y + 90), (width - 70, y + 90)], fill=(203, 213, 225), width=1)
    draw.text((width - 400, y + 110), "Total Balance Due:", fill=(15, 23, 42), font=font_bold)
    draw.text((width - 180, y + 110), "$17,255.00", fill=(37, 99, 235), font=font_bold)

    # Footer Notes
    draw.text((60, height - 120), "Thank you for your partnership! Please remit payment within 30 days.", fill=(100, 116, 139), font=font_small)
    draw.text((60, height - 90), "For billing inquiries, contact accounting@acme-cloud.example.com", fill=(148, 163, 184), font=font_small)

    return img


def draw_page_content(page_num, rotation_desc, width=1200, height=1600):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 44)
        font_sub = ImageFont.truetype("arial.ttf", 26)
        font_body = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        font_title = font_sub = font_body = ImageFont.load_default()

    draw.rectangle([(50, 40), (width - 50, 48)], fill=(37, 99, 235))
    draw.text((60, 80), f"ANNUAL OPERATIONS REPORT — SECTION {page_num}", fill=(15, 23, 42), font=font_title)
    draw.text((60, 140), f"Page {page_num} Orientation State: {rotation_desc}", fill=(71, 85, 105), font=font_sub)
    draw.line([(50, 190), (width - 50, 190)], fill=(226, 232, 240), width=2)

    paragraphs = [
        f"This document page demonstrates real-world document orientation detection. In enterprise ingestion "
        f"pipelines, scanned paperwork, mobile camera uploads, and multi-page contract submissions routinely "
        f"arrive with arbitrary or inconsistent rotation angles.",
        f"Our production pipeline combines high-throughput local convolutional neural network inference "
        f"(MobileNet/ResNet/EfficientNet via ONNX Runtime), RapidOCR text orientation validation for low-confidence "
        f"cases, and Gemini Multimodal LLM verification.",
        f"For page {page_num}, the page was intentionally oriented at {rotation_desc}. The automated system identifies "
        f"the visual upright angle and applies lossless affine rotation to render all document content correctly readable.",
        f"Document processing statistics indicate over 99.4% classification accuracy across corporate documents, "
        f"tax declarations, receipts, technical schematics, and academic publications."
    ]

    y = 230
    for p in paragraphs:
        # Simple word wrap
        words = p.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 70:
                draw.text((60, y), line, fill=(30, 41, 59), font=font_body)
                y += 36
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            draw.text((60, y), line, fill=(30, 41, 59), font=font_body)
            y += 50

    draw.text((60, height - 80), f"Confidential — Internal Use Only — Page {page_num} of 4", fill=(148, 163, 184), font=font_body)
    return img


def create_samples():
    print("Generating invoice_90.png...")
    inv = draw_invoice()
    # Rotate 90 degrees clockwise (PIL rotate with -90, or ROTATE_270 for 90 CW)
    inv_90 = inv.transpose(Image.Transpose.ROTATE_270)
    inv_90.save(SAMPLES_DIR / "invoice_90.png", "PNG", optimize=True)

    print("Generating blank.png...")
    blank = Image.new("RGB", (1200, 1600), color=(255, 255, 255))
    blank.save(SAMPLES_DIR / "blank.png", "PNG", optimize=True)

    print("Generating mixed_rotations.pdf...")
    doc = fitz.open()
    
    # Page 1: 0 deg (Upright)
    p1_img = draw_page_content(1, "0° Upright")
    page1 = doc.new_page(width=p1_img.width, height=p1_img.height)
    # Save temp image to insert
    tmp1 = SAMPLES_DIR / "_p1.png"
    p1_img.save(tmp1)
    page1.insert_image(page1.rect, filename=str(tmp1))
    tmp1.unlink()

    # Page 2: 90 deg clockwise
    p2_img = draw_page_content(2, "90° Clockwise")
    p2_rot = p2_img.transpose(Image.Transpose.ROTATE_270)
    page2 = doc.new_page(width=p2_rot.width, height=p2_rot.height)
    tmp2 = SAMPLES_DIR / "_p2.png"
    p2_rot.save(tmp2)
    page2.insert_image(page2.rect, filename=str(tmp2))
    tmp2.unlink()

    # Page 3: 180 deg upside down
    p3_img = draw_page_content(3, "180° Upside Down")
    p3_rot = p3_img.transpose(Image.Transpose.ROTATE_180)
    page3 = doc.new_page(width=p3_rot.width, height=p3_rot.height)
    tmp3 = SAMPLES_DIR / "_p3.png"
    p3_rot.save(tmp3)
    page3.insert_image(page3.rect, filename=str(tmp3))
    tmp3.unlink()

    # Page 4: 270 deg clockwise (90 counter-clockwise)
    p4_img = draw_page_content(4, "270° Clockwise")
    p4_rot = p4_img.transpose(Image.Transpose.ROTATE_90)
    page4 = doc.new_page(width=p4_rot.width, height=p4_rot.height)
    tmp4 = SAMPLES_DIR / "_p4.png"
    p4_rot.save(tmp4)
    page4.insert_image(page4.rect, filename=str(tmp4))
    tmp4.unlink()

    pdf_bytes = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    with open(SAMPLES_DIR / "mixed_rotations.pdf", "wb") as f:
        f.write(pdf_bytes)

    print("All sample files generated successfully in", SAMPLES_DIR)


if __name__ == "__main__":
    create_samples()

