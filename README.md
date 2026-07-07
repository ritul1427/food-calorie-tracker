# Food Calorie Tracker (Web + API)

This repository contains:
- `index.html`: Web app UI for uploading meal photos, viewing nutrient bifurcation, logging meals, and daily charts.
- `app.py`: FastAPI backend that accepts meal image uploads and returns estimated nutrients.
- `requirements.txt`: Python dependencies.
- `render.yaml`: One-click Render blueprint to deploy both backend and frontend.

## Quick Deploy on Render

1. Push this repository to GitHub.
2. Log in to Render and choose **New + > Blueprint**.
3. Select this repository.
4. Render will create:
   - `food-calorie-api` (FastAPI backend)
   - `food-calorie-web` (static frontend)
5. After deploy, set `API_BASE_URL` in `index.html` to your backend URL and redeploy static site.

## Local Run (optional)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Notes

- Current detection is OCR + heuristic fallback, useful for MVP workflow.
- For higher confidence/precision, integrate food detection + segmentation + portion estimation model.
