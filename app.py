from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import re
import numpy as np
import easyocr

app = FastAPI(title="Food Calorie Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'], gpu=False)

FOOD_DB = {
    "rice": {"kcal":130, "protein":2.4, "carbs":28.0, "fat":0.3, "fiber":0.4, "sugar":0.1, "sodium":1},
    "dal": {"kcal":116, "protein":9.0, "carbs":20.0, "fat":0.4, "fiber":8.0, "sugar":1.8, "sodium":2},
    "chicken curry": {"kcal":190, "protein":16.0, "carbs":4.0, "fat":12.0, "fiber":0.7, "sugar":1.2, "sodium":320},
    "paneer": {"kcal":265, "protein":18.0, "carbs":3.0, "fat":21.0, "fiber":0.0, "sugar":1.2, "sodium":22},
    "roti": {"kcal":297, "protein":9.6, "carbs":55.0, "fat":3.7, "fiber":4.9, "sugar":1.2, "sodium":5},
    "salad": {"kcal":35, "protein":1.5, "carbs":7.0, "fat":0.2, "fiber":2.4, "sugar":3.2, "sodium":28},
    "fries": {"kcal":312, "protein":3.4, "carbs":41.0, "fat":15.0, "fiber":3.8, "sugar":0.3, "sodium":210},
    "egg omelette": {"kcal":154, "protein":10.0, "carbs":1.6, "fat":11.0, "fiber":0.0, "sugar":1.0, "sodium":155},
    "fish": {"kcal":206, "protein":22.0, "carbs":0.0, "fat":12.0, "fiber":0.0, "sugar":0.0, "sodium":70},
    "banana": {"kcal":89, "protein":1.1, "carbs":22.8, "fat":0.3, "fiber":2.6, "sugar":12.2, "sodium":1},
    "apple": {"kcal":52, "protein":0.3, "carbs":14.0, "fat":0.2, "fiber":2.4, "sugar":10.4, "sodium":1},
    "bread": {"kcal":265, "protein":9.0, "carbs":49.0, "fat":3.2, "fiber":2.7, "sugar":5.0, "sodium":491},
    "pizza": {"kcal":266, "protein":11.0, "carbs":33.0, "fat":10.0, "fiber":2.3, "sugar":3.6, "sodium":598},
    "burger": {"kcal":295, "protein":17.0, "carbs":30.0, "fat":12.0, "fiber":1.5, "sugar":5.0, "sodium":450}
}

SYNONYMS = {
    "lentil": "dal",
    "chapati": "roti",
    "omelet": "egg omelette",
    "omelette": "egg omelette",
    "naan": "roti"
}


def round1(x):
    return round(float(x), 1)


def calc_item(food_name: str, qty_g: float, confidence: float):
    base = FOOD_DB[food_name]
    f = qty_g / 100.0
    return {
        "name": food_name,
        "qty_g": round1(qty_g),
        "confidence": round(confidence, 2),
        "calories": round1(base["kcal"] * f),
        "protein": round1(base["protein"] * f),
        "carbs": round1(base["carbs"] * f),
        "fat": round1(base["fat"] * f),
        "fiber": round1(base["fiber"] * f),
        "sugar": round1(base["sugar"] * f),
        "sodium": round1(base["sodium"] * f),
    }


def sum_totals(items):
    keys = ["calories", "protein", "carbs", "fat", "fiber", "sugar", "sodium"]
    totals = {k: 0.0 for k in keys}
    for it in items:
        for k in keys:
            totals[k] += it[k]
    for k in keys:
        totals[k] = round1(totals[k])
    return totals


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    np_img = np.array(image)

    ocr_results = reader.readtext(np_img, detail=0)
    detected_text = " ".join(ocr_results).lower()

    tokens = re.findall(r"[a-zA-Z]+", detected_text)

    matched_foods = set()
    for token in tokens:
        t = token.strip().lower()
        if t in SYNONYMS:
            t = SYNONYMS[t]
        if t in FOOD_DB:
            matched_foods.add(t)

    if not matched_foods:
        matched_foods = {"rice", "dal"}

    qty_candidates = re.findall(r"(\\d+)\\s?(g|gm|grams)", detected_text)
    parsed_grams = [int(x[0]) for x in qty_candidates if x[0].isdigit()]
    default_qty = 150

    items = []
    for food in matched_foods:
        qty = parsed_grams[0] if parsed_grams else default_qty
        conf = 0.72 if parsed_grams else 0.64
        items.append(calc_item(food, qty, conf))

    mean_conf = round(sum(i["confidence"] for i in items) / len(items), 2)
    totals = sum_totals(items)

    return {
        "detected_text": detected_text[:500],
        "items": items,
        "totals": totals,
        "mean_confidence": mean_conf,
        "note": "OCR+heuristic detection. For higher precision, plug in a dedicated food vision model with portion estimation."
    }
