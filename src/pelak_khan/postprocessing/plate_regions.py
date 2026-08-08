from __future__ import annotations

from typing import Any

# Public Iranian plate-code references are not an official registry and some
# county codes depend on BOTH the two-digit Iran code and the plate letter.
# This baseline intentionally returns a precise city only for codes that are
# broadly and consistently associated with one city in public references.
# For shared county codes we return the province/area and mark the result as
# province-level/ambiguous instead of guessing a city.

CITY_CODE_MAP: dict[str, tuple[str, str]] = {
    # Tehran city series
    "10": ("تهران", "تهران"),
    "11": ("تهران", "تهران"),
    "22": ("تهران", "تهران"),
    "33": ("تهران", "تهران"),
    "44": ("تهران", "تهران"),
    "55": ("تهران", "تهران"),
    "66": ("تهران", "تهران"),
    "77": ("تهران", "تهران"),
    "88": ("تهران", "تهران"),
    "99": ("تهران", "تهران"),

    # Province capitals / well-known city codes
    "12": ("خراسان رضوی", "مشهد"),
    "13": ("اصفهان", "اصفهان"),
    "14": ("خوزستان", "اهواز"),
    "15": ("آذربایجان شرقی", "تبریز"),
    "16": ("قم", "قم"),
    "17": ("آذربایجان غربی", "ارومیه"),
    "18": ("همدان", "همدان"),
    "19": ("کرمانشاه", "کرمانشاه"),
    "30": ("البرز", "کرج"),
    "31": ("لرستان", "خرم‌آباد"),
    "45": ("کرمان", "کرمان"),
    "46": ("گیلان", "رشت"),
    "47": ("مرکزی", "اراک"),
    "48": ("بوشهر", "بوشهر"),
    "49": ("کهگیلویه و بویراحمد", "یاسوج"),
    "51": ("کردستان", "سنندج"),
    "54": ("یزد", "یزد"),
    "59": ("گلستان", "گرگان"),
    "62": ("مازندران", "ساری"),
    "63": ("فارس", "شیراز"),
    "71": ("چهارمحال و بختیاری", "شهرکرد"),
    "79": ("قزوین", "قزوین"),
    "84": ("هرمزگان", "بندرعباس"),
    "85": ("سیستان و بلوچستان", "زاهدان"),
    "86": ("سمنان", "سمنان"),
    "87": ("زنجان", "زنجان"),
    "91": ("اردبیل", "اردبیل"),
    "98": ("ایلام", "ایلام"),
}

# Codes commonly used by counties/other cities of the province. Exact city
# often depends on the Persian letter, so we deliberately avoid inventing one.
PROVINCE_CODE_MAP: dict[str, str] = {
    "21": "تهران / البرز (کدهای قدیمی و شهرستانی؛ نیازمند حرف پلاک)",
    "23": "اصفهان",
    "24": "خوزستان",
    "25": "آذربایجان شرقی",
    "26": "خراسان شمالی",
    "27": "آذربایجان غربی",
    "28": "همدان",
    "29": "کرمانشاه",
    "32": "خراسان رضوی",
    "34": "خوزستان",
    "35": "آذربایجان شرقی",
    "36": "خراسان رضوی",
    "37": "آذربایجان غربی",
    "38": "البرز",
    "39": "کرمان",
    "41": "لرستان",
    "42": "خراسان رضوی",
    "43": "اصفهان",
    "52": "خراسان جنوبی",
    "53": "اصفهان",
    "56": "گیلان",
    "57": "مرکزی",
    "58": "بوشهر",
    "61": "کردستان",
    "64": "یزد",
    "65": "کرمان",
    "67": "اصفهان",
    "68": "البرز / تهران (کد قدیمی؛ نیازمند بررسی حرف)",
    "69": "گلستان",
    "72": "مازندران",
    "73": "فارس",
    "74": "خراسان رضوی",
    "75": "کرمان",
    "76": "گیلان",
    "78": "تهران (شهرستان‌های استان؛ نیازمند حرف پلاک)",
    "81": "چهارمحال و بختیاری",
    "82": "مازندران",
    "83": "فارس",
    "89": "قزوین",
    "92": "مازندران",
    "93": "فارس",
    "94": "هرمزگان",
    "95": "سیستان و بلوچستان",
    "96": "سمنان",
    "97": "زنجان",
}

# A small set of high-confidence letter-specific refinements. We can expand
# this table later as we verify each province against stronger references.
LETTER_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    # East Azerbaijan examples from public plate tables.
    ("25", "ب"): ("آذربایجان شرقی", "مراغه"),
    ("25", "ج"): ("آذربایجان شرقی", "مرند"),
    ("25", "د"): ("آذربایجان شرقی", "میانه"),
    ("25", "س"): ("آذربایجان شرقی", "اهر / هوراند"),
    ("25", "ص"): ("آذربایجان شرقی", "سراب"),
    ("25", "ط"): ("آذربایجان شرقی", "جلفا"),
    ("25", "ق"): ("آذربایجان شرقی", "هشترود"),
    ("25", "ل"): ("آذربایجان شرقی", "بناب"),
    ("25", "م"): ("آذربایجان شرقی", "بستان‌آباد"),
    ("25", "ن"): ("آذربایجان شرقی", "شبستر"),
    ("25", "و"): ("آذربایجان شرقی", "کلیبر / خداآفرین"),
    ("25", "ه"): ("آذربایجان شرقی", "هریس"),
    ("25", "ی"): ("آذربایجان شرقی", "آذرشهر"),
    ("35", "ب"): ("آذربایجان شرقی", "اسکو"),
    ("35", "ص"): ("آذربایجان شرقی", "ملکان"),
    ("35", "ق"): ("آذربایجان شرقی", "مراغه"),
    ("35", "ط"): ("آذربایجان شرقی", "مرند"),
    ("35", "ن"): ("آذربایجان شرقی", "میانه"),
    ("35", "و"): ("آذربایجان شرقی", "اهر / هوراند"),
    ("35", "ه"): ("آذربایجان شرقی", "تبریز"),
    ("35", "ی"): ("آذربایجان شرقی", "تبریز"),
    ("35", "ل"): ("آذربایجان شرقی", "تبریز"),
    ("35", "م"): ("آذربایجان شرقی", "تبریز"),
}


def extract_plate_parts(text: str) -> dict[str, str] | None:
    """Return normalized standard-plate segments from DDLDDDDD."""
    raw = str(text or "").strip().replace(" ", "")
    if len(raw) != 8:
        return None
    return {
        "left": raw[:2],
        "letter": raw[2],
        "middle": raw[3:6],
        "iran_code": raw[6:8],
    }


def lookup_plate_region(text: str) -> dict[str, Any]:
    parts = extract_plate_parts(text)
    if not parts:
        return {
            "known": False,
            "iran_code": None,
            "province": None,
            "city": None,
            "label": "نامشخص",
            "precision": "unknown",
        }

    code = parts["iran_code"]
    letter = parts["letter"]

    override = LETTER_OVERRIDES.get((code, letter))
    if override:
        province, city = override
        return {
            "known": True,
            "iran_code": code,
            "province": province,
            "city": city,
            "label": f"{city}، {province}",
            "precision": "code+letter",
        }

    direct = CITY_CODE_MAP.get(code)
    if direct:
        province, city = direct
        return {
            "known": True,
            "iran_code": code,
            "province": province,
            "city": city,
            "label": f"{city}، {province}" if city != province else city,
            "precision": "code",
        }

    province = PROVINCE_CODE_MAP.get(code)
    if province:
        return {
            "known": True,
            "iran_code": code,
            "province": province,
            "city": None,
            "label": province,
            "precision": "province",
        }

    return {
        "known": False,
        "iran_code": code,
        "province": None,
        "city": None,
        "label": f"ایران {code} — نامشخص",
        "precision": "unknown",
    }
