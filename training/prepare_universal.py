"""Global, realistically balanced multilingual & multi-domain dataset preparer.

Realistic global document distribution:
- 60% (~2,500) Standard global documents (DocLayNet: multi-column reports, manuals, legal, science)
- 6%  (~250)   Receipts & bills (CORD v2: crumpled, mobile photos on tables, narrow paper)
- 4%  (~150)   Complex forms, questionnaires, and stamps (FUNSD)
- 30% (~1,200) Global World Scripts:
    * Cyrillic (Russian, Belarusian, Ukrainian, Bulgarian) - 300
    * Arabic / Middle Eastern (Arabic, Farsi, Urdu) - 300
    * Indic / South Asian (Devanagari, Hindi, Bengali) - 300
    * East Asian / CJK (Chinese, Japanese, Korean) - 300
"""
import argparse
import hashlib
import io
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from datasets import Image as HFImage, load_dataset

from training.dataset import (
    CORD_REPO, DOCLAYNET_REPO, SPLITS, _configure_windows_tls,
)
from training.prepare_data import save_manifest, save_page

FUNSD_REPO = "nielsr/funsd"


def get_system_font(script, size=22):
    """Find appropriate Windows system font for each global script."""
    font_paths = {
        "cyrillic": ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\calibri.ttf"],
        "arabic": ["C:\\Windows\\Fonts\\seguiui.ttf", "C:\\Windows\\Fonts\\arial.ttf"],
        "indic": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\arial.ttf"],
        "cjk": ["C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simsun.ttc"],
        "latin": ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\times.ttf"],
    }
    for path in font_paths.get(script, []):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_global_multilingual_corpus(count_per_script=300):
    """Generate realistic multilingual documents across 4 major world script families."""
    docs = []

    corpus_data = {
        "cyrillic": [
            ("МИНИСТЕРСТВО ОБРАЗОВАНИЯ", ["ДИПЛОМ О ВЫСШЕМ ОБРАЗОВАНИИ", "Решением государственной экзаменационной комиссии", "Присуждена степень бакалавра техники и технологий", "Регистрационный номер: 2024-8841", "Город: Минск / Москва"]),
            ("СВИДЕТЕЛЬСТВО О ГОСУДАРСТВЕННОЙ РЕГИСТРАЦИИ", ["ЕДИНЫЙ ГОСУДАРСТВЕННЫЙ РЕЕСТР", "Индивидуальный предприниматель", "Основной государственный регистрационный номер", "Дата внесения записи: 12 марта 2022 года"]),
            ("СПРАВКА С МЕСТА РАБОТЫ", ["Дана для предъявления по месту требования", "Работает в должности ведущего инженера-разработчика", "Среднемесячная заработная плата составляет", "Руководитель предприятия: _________________"]),
            ("АКТ ПРИЕМА-ПЕРЕДАЧИ ТОВАРОВ", ["Номер документа: 451/2023", "Поставщик передал, а Покупатель принял продукцию", "Претензий по количеству и качеству не имеется", "Печать и подпись сторон: _________________"]),
        ],
        "arabic": [
            ("جمهورية مصر العربية - وزارة المالية", ["شهادة تسجيل رسمية للشركات", "رقم السجل التجاري الموحد: 8492041", "تاريخ الإصدار: 15 ربيع الأول 1445 هـ", "المقر الرئيسي: القاهرة"]),
            ("المملكة العربية السعودية - عقد رسمي", ["الهيئة العامة للزكاة والضريبة والجمارك", "فاتورة ضريبية معتمدة وفق الأنظمة", "المبلغ الإجمالي المستحق للدفع: 4,500 ريال", "رمز الاستجابة السريعة والختم الإلكتروني"]),
            ("عقد اتفاق وتعيين خدمات مهنية", ["الطرف الأول: شركة التكنولوجيا والخدمات", "الطرف الثاني: المستشار المالي المستقل", "تعتبر هذه الوثيقة نافذة من تاريخ التوقيع", "التوقيع والاعتماد الرسمي: ________________"]),
            ("شهادة إتمام وتخرج أكاديمي", ["جامعة الدول للعلوم والتكنولوجيا", "بناء على قرار مجلس الكلية الموقر", "تقرر منح درجة البكالوريوس بتقدير امتياز", "عميد الكلية ورئيس الجامعة: _____________"]),
        ],
        "indic": [
            ("भारत सरकार / GOVERNMENT OF INDIA", ["प्रमाण पत्र / OFFICIAL CERTIFICATE", "पहचान प्राधिकरण द्वारा प्रमाणित दस्तावेज", "पंजीकरण संख्या: 9841-2041-8841", "जारी करने की तिथि: 12 अगस्त 2023"]),
            ("विश्वविद्यालय परीक्षा परिणाम पत्र", ["अखिल भारतीय उच्च शिक्षा परिषद", "छात्र का नाम और अनुक्रमांक विवरण", "अंक तालिका एवं योग्यता प्रमाण पत्र", "परीक्षा नियंत्रक हस्ताक्षर: ________________"]),
            ("व्यापार एवं वाणिज्य पंजीकरण विवरणी", ["राष्ट्रीय लघु उद्योग निगम लिमिटेड", "करदाता पहचान संख्या (GSTIN) विवरणी", "अधिकृत हस्ताक्षरकर्ता: ________________", "स्थान: नई दिल्ली, भारत"]),
            ("भूमि एवं संपत्ति राजस्व अभिलेख", ["तहसील कार्यालय राजस्व विभाग", "स्वामित्व एवं अधिकार अभिलेख प्रतिलिपि", "प्रमाणित किया जाता है कि अभिलेख सत्य है", "राजस्व अधिकारी मोहर एवं हस्ताक्षर"]),
        ],
        "cjk": [
            ("中华人民共和国 国家税务总局", ["增值税专用发票 / 全国统一发票", "发票代码: 031002100411", "购买方名称: 科技有限公司", "合计金额 (大写): 壹万贰仟伍佰元整", "开票人及财务专用章: _____________"]),
            ("商事登记证明书 (企业法人)", ["统一社会信用代码: 91310000X4819", "法定代表人: 李明", "经营范围: 计算机软件开发与技术服务", "登记机关: 市场监督管理局"]),
            ("株式会社 業務委託基本契約書", ["甲: 株式会社イノベーション", "乙: グローバルテクノロジー合同会社", "契約締結日: 2023年4月1日", "代表取締役 印鑑: ________________"]),
            ("不動産売買重要事項説明書", ["東京都千代田区大手町一丁目", "物件概要及び権利関係に関する説明", "宅地建物取引士による記名押印", "説明年月日: 令和5年10月15日"]),
        ],
    }

    for script, templates in corpus_data.items():
        font_large = get_system_font(script, size=24)
        font_body = get_system_font(script, size=16)

        for i in range(count_per_script):
            title, lines = random.choice(templates)
            bg = random.choice([
                (252, 250, 246), (245, 248, 250), (255, 255, 255), (248, 248, 248), (255, 253, 248)
            ])
            w = random.choice([550, 600, 700])
            h = int(w * random.uniform(1.3, 1.45))
            img = Image.new("RGB", (w, h), color=bg)
            draw = ImageDraw.Draw(img)

            # Security patterns / watermarks / headers / stamps
            if random.random() < 0.6:
                cx, cy = w // 2, h - 90
                for r in range(15, 75, 12):
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(215, 225, 220), width=1)

            if random.random() < 0.5:
                draw.rectangle([15, 15, w - 15, h - 15], outline=(195, 205, 200), width=2)

            # Title placement (standard header or certificate/form layout)
            title_y = random.choice([40, 70, 140])
            bbox = draw.textbbox((0, 0), title, font=font_large)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, title_y), title, font=font_large, fill=(20, 25, 30))

            # Lines
            y = title_y + 45
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font_body)
                tw = bbox[2] - bbox[0]
                tx = (w - tw) // 2 if script in ("cyrillic", "cjk") else (w - tw - 40 if script == "arabic" else 40)
                draw.text((tx, y), line, font=font_body, fill=(40, 45, 50))
                y += 32

            # Red official stamp / seal (common globally in Asia, Middle East, Europe)
            if random.random() < 0.5:
                sx, sy = random.choice([(w - 120, h - 120), (80, h - 120), (w // 2, h - 140)])
                draw.ellipse([sx - 35, sy - 35, sx + 35, sy + 35], outline=(185, 45, 45), width=2)

            docs.append((img, {
                "group_id": f"global:{script}:{i}",
                "language": script,
                "category": "official_document",
                "upright_verified": True,
                "correction_cw": 0,
            }))

    return docs


def prepare_balanced_universal(output_dir="data/v2_universal_4k", doclaynet_count=2500, cord_count=250):
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    manifest_path = folder / "manifest.jsonl"
    rows = []

    print(f"\n=======================================================")
    print(f" Preparing Realistic Universal Global Dataset in {folder}")
    print(f" Target: 60% Corporate/Academic, 30% Global World Scripts, 6% Receipts, 4% Forms")
    print(f"=======================================================\n")

    # 1. Global Multilingual Scripts (Cyrillic, Arabic, Indic, CJK) - 1,200 pages
    print("1. Generating Global Multilingual Documents (Cyrillic, Arabic, Indic, CJK)...")
    for img, rec in generate_global_multilingual_corpus(count_per_script=300):
        rel, dig = save_page(img, folder, 1024)
        split = "train" if random.random() < 0.75 else ("validation" if random.random() < 0.6 else "test")
        rows.append({**rec, "path": rel, "sha256": dig, "split": split, "source": f"global_{rec['language']}"})
    print(f"   -> Added 1,200 global multi-script documents.\n")

    # 2. FUNSD Forms (Noisy real administrative forms) - ~150 pages
    print("2. Loading FUNSD real administrative forms...")
    try:
        funsd_ds = load_dataset(FUNSD_REPO, split="train")
        for i, item in enumerate(funsd_ds):
            rel, dig = save_page(item["image"], folder, 1024)
            split = "train" if i < 110 else "validation"
            rows.append({
                "path": rel, "sha256": dig, "split": split, "source": "funsd",
                "group_id": f"funsd:{item.get('id', i)}", "upright_verified": True,
                "correction_cw": 0, "category": "form", "language": "en"
            })
        print(f"   -> Added {len(funsd_ds)} FUNSD form pages.\n")
    except Exception as exc:
        print(f"   -> FUNSD skipped: {exc}\n")

    # 3. CORD Receipts (Balanced proportion: ~250 receipts)
    print(f"3. Loading {cord_count} CORD receipts (realistic small proportion)...")
    _configure_windows_tls()
    budgets_cord = [int(cord_count * 0.7), int(cord_count * 0.15)]
    budgets_cord.append(cord_count - sum(budgets_cord))
    for split, limit in zip(SPLITS, budgets_cord):
        try:
            ds = load_dataset(CORD_REPO, split=split, streaming=True, columns=["image"])
            ds = ds.cast_column("image", HFImage(decode=False)).shuffle(seed=42, buffer_size=64)
            count = 0
            for item in ds:
                raw = item["image"]
                with Image.open(io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]) as image:
                    rel, dig = save_page(image, folder, 1024)
                rows.append({
                    "path": rel, "sha256": dig, "split": split, "source": "cord",
                    "group_id": f"cord:{split}:{count}", "upright_verified": True,
                    "correction_cw": 0, "category": "receipt", "language": "ko-id-en"
                })
                count += 1
                if count >= limit:
                    break
            print(f"   CORD {split}: {count} pages.")
        except Exception as exc:
            print(f"   CORD {split} failed: {exc}")
    print()

    # 4. DocLayNet (Global Corporate, Legal, Academic, Scientific) - ~2,500 pages
    print(f"4. Loading {doclaynet_count} DocLayNet standard global documents...")
    budgets_doc = [int(doclaynet_count * 0.7), int(doclaynet_count * 0.15)]
    budgets_doc.append(doclaynet_count - sum(budgets_doc))
    for split, limit in zip(SPLITS, budgets_doc):
        try:
            ds = load_dataset(DOCLAYNET_REPO, split=split, streaming=True, columns=["image", "metadata"])
            ds = ds.cast_column("image", HFImage(decode=False)).shuffle(seed=42, buffer_size=64)
            count = 0
            for item in ds:
                raw = item["image"]
                with Image.open(io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]) as image:
                    rel, dig = save_page(image, folder, 1024)
                meta = item.get("metadata", {})
                docname = meta.get("original_filename") or meta.get("doc_name") or f"doclaynet:{count}"
                rows.append({
                    "path": rel, "sha256": dig, "split": split, "source": "doclaynet",
                    "group_id": f"doclaynet:{docname}", "upright_verified": True,
                    "correction_cw": 0, "category": meta.get("doc_category", "document"),
                    "language": "en"
                })
                count += 1
                if count >= limit:
                    break
            print(f"   DocLayNet {split}: {count} pages.")
        except Exception as exc:
            print(f"   DocLayNet {split} failed: {exc}")

    save_manifest(manifest_path, rows)
    print(f"\n=======================================================")
    print(f" SUCCESS! Balanced Universal Dataset: {len(rows)} pages")
    print(f" Manifest saved to: {manifest_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/v2_universal_4k")
    parser.add_argument("--doclaynet", type=int, default=2500)
    parser.add_argument("--cord", type=int, default=250)
    args = parser.parse_args()
    prepare_balanced_universal(args.output_dir, args.doclaynet, args.cord)
