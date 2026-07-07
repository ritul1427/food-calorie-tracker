from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, json, base64, difflib, re
from typing import Dict, Any, List
import requests
from openai import OpenAI

app = FastAPI(title="Food Calorie Analyzer API (Dynamic Nutrition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID", "")
NUTRITIONIX_APP_KEY = os.getenv("NUTRITIONIX_APP_KEY", "")
CACHE_FILE = "nutrition_cache.json"

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Optional starter seed (still allowed, but not required)
FOOD_DB: Dict[str, Dict[str, float]] = {
    "rice": {"kcal":130, "protein":2.4, "carbs":28.0, "fat":0.3, "fiber":0.4, "sugar":0.1, "sodium":1},
    "dal": {"kcal":116, "protein":9.0, "carbs":20.0, "fat":0.4, "fiber":8.0, "sugar":1.8, "sodium":2},
    "roti": {"kcal":297, "protein":9.6, "carbs":55.0, "fat":3.7, "fiber":4.9, "sugar":1.2, "sodium":5},
    "khandvi": {"kcal":160, "protein":6.0, "carbs":20.0, "fat":6.0, "fiber":2.0, "sugar":2.0, "sodium":220},
}

def load_cache() -> Dict[str, Dict[str, float]]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

CACHE = load_cache()

def norm_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def round1(x: float) -> float:
    return round(float(x), 1)

def nutritionix_query_food(food_name: str) -> Dict[str, float] | None:
    if not NUTRITIONIX_APP_ID or not NUTRITIONIX_APP_KEY:
        return None

    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_APP_KEY,
        "Content-Type": "application/json",
    }

    # Ask per 100g explicitly
    payload = {"query": f"100g {food_name}"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        foods = data.get("foods", [])
        if not foods:
            return None
        f = foods[0]

        # Nutritionix sodium is usually mg already
        result = {
            "kcal": float(f.get("nf_calories", 0) or 0),
            "protein": float(f.get("nf_protein", 0) or 0),
            "carbs": float(f.get("nf_total_carbohydrate", 0) or 0),
            "fat": float(f.get("nf_total_fat", 0) or 0),
            "fiber": float(f.get("nf_dietary_fiber", 0) or 0),
            "sugar": float(f.get("nf_sugars", 0) or 0),
            "sodium": float(f.get("nf_sodium", 0) or 0),  # mg
        }
        return result
    except Exception:
        return None

def get_nutrition_dynamic(food_name: str) -> Dict[str, float]:
    key = norm_name(food_name)

    # 1) local seed
    if key in FOOD_DB:
        return FOOD_DB[key]

    # 2) cache
    if key in CACHE:
        return CACHE[key]

    # 3) direct nutrition API
    val = nutritionix_query_food(key)

    # 4) fallback query variants for Indian dishes
    if not val:
        for q in [f"{key} indian", f"{key} recipe", key.replace("bhaji", "sabzi")]:
            val = nutritionix_query_food(q)
            if val:
                break

    # 5) fuzzy fallback from known keys (last resort)
    if not val:
        all_keys = list(FOOD_DB.keys()) + list(CACHE.keys())
        match = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.86)
        if match:
            m = match[0]
            return FOOD_DB.get(m) or CACHE.get(m)

    # 6) unknown safe fallback
    if not val:
        val = {"kcal": 120, "protein": 4, "carbs": 15, "fat": 4, "fiber": 2, "sugar": 2, "sodium": 120}

    CACHE[key] = val
    save_cache(CACHE)
    return val

def detect_food_items_with_openai(image_bytes: bytes) -> List[Dict[str, Any]]:
    if not client:
        # Fallback when no OpenAI key
        return [{"name": "unknown dish", "confidence": 0.35, "qty_g": 150}]

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = """
Identify visible food items in this meal image.
Return STRICT JSON:
{
  "items":[
    {"name":"khandvi","confidence":0.0,"qty_g":150}
  ]
}
Rules:
- lowercase names
- confidence between 0 and 1
- qty_g integer estimate per item
- max 3 items
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an accurate food recognition assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        temperature=0
    )

    content = resp.choices[0].message.content
    data = json.loads(content)
    items = data.get("items", [])
    cleaned = []
    for it in items[:3]:
        name = norm_name(str(it.get("name", "unknown dish")))
        conf = float(it.get("confidence", 0.5) or 0.5)
        qty = int(it.get("qty_g", 150) or 150)
        conf = max(0.0, min(1.0, conf))
        qty = max(1, min(1000, qty))
        cleaned.append({"name": name, "confidence": conf, "qty_g": qty})
    return cleaned or [{"name": "unknown dish", "confidence": 0.35, "qty_g": 150}]

def calc_item(food_name: str, qty_g: float, confidence: float) -> Dict[str, Any]:
    n = get_nutrition_dynamic(food_name)  # per 100g
    f = qty_g / 100.0
    return {
        "name": food_name,
        "qty_g": round1(qty_g),
        "confidence": round(confidence, 2),
        "calories": round1(n["kcal"] * f),
        "protein": round1(n["protein"] * f),
        "carbs": round1(n["carbs"] * f),
        "fat": round1(n["fat"] * f),
        "fiber": round1(n["fiber"] * f),
        "sugar": round1(n["sugar"] * f),
        "sodium": round1(n["sodium"] * f),
    }

def sum_totals(items: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["calories", "protein", "carbs", "fat", "fiber", "sugar", "sodium"]
    t = {k: 0.0 for k in keys}
    for it in items:
        for k in keys:
            t[k] += float(it.get(k, 0) or 0)
    for k in keys:
        t[k] = round1(t[k])
    return t

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()

    # Validate image
    try:
        _ = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return {"error": "invalid image file"}

    detected = detect_food_items_with_openai(content)
    items = [calc_item(d["name"], d.get("qty_g", 150), d.get("confidence", 0.5)) for d in detected]
    totals = sum_totals(items)
    mean_conf = round(sum(i["confidence"] for i in items) / len(items), 2) if items else 0.0

    return {
        "detected_text": "",
        "items": items,
        "totals": totals,
        "mean_confidence": mean_conf,
        "note": "dynamic nutrition: cache + nutrition API lookup"
    }
