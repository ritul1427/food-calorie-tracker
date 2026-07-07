from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import random

app = FastAPI(title="Food Calorie Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FOOD_DB = {
    "rice": {"kcal":130, "protein":2.4, "carbs":28.0, "fat":0.3, "fiber":0.4, "sugar":0.1, "sodium":1},
    "dal": {"kcal":116, "protein":9.0, "carbs":20.0, "fat":0.4, "fiber":8.0, "sugar":1.8, "sodium":2},
    "paneer": {"kcal":265, "protein":18.0, "carbs":3.0, "fat":21.0, "fiber":0.0, "sugar":1.2, "sodium":22},
    "roti": {"kcal":297, "protein":9.6, "carbs":55.0, "fat":3.7, "fiber":4.9, "sugar":1.2, "sodium":5},
    "salad": {"kcal":35, "protein":1.5, "carbs":7.0, "fat":0.2, "fiber":2.4, "sugar":3.2, "sodium":28},
    "egg omelette": {"kcal":154, "protein":10.0, "carbs":1.6, "fat":11.0, "fiber":0.0, "sugar":1.0, "sodium":155},
    "banana": {"kcal":89, "protein":1.1, "carbs":22.8, "fat":0.3, "fiber":2.6, "sugar":12.2, "sodium":1},
}

def round1(x): return round(float(x), 1)

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
    _ = Image.open(io.BytesIO(content)).convert("RGB")  # validate image

    # Lightweight mock inference for free-tier hosting
    picks = random.sample(list(FOOD_DB.keys()), k=2)
    items = [calc_item(name, qty_g=150, confidence=0.65) for name in picks]
    totals = sum_totals(items)
    mean_conf = round(sum(i["confidence"] for i in items) / len(items), 2)

    return {
        "detected_text": "",
        "items": items,
        "totals": totals,
        "mean_confidence": mean_conf,
        "note": "Lightweight estimation mode (no OCR/ML) for free-tier deployment."
    }
