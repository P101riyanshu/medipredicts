## Run Instructions

1. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

2. Generate dataset:
```bash
python data/preprocess.py
```

3. Train models and save best model to backend:
```bash
python training/train.py
```

4. (Optional) Generate evaluation report:
```bash
python training/evaluate.py
```

5. Start backend API:
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

6. Start frontend static server (new terminal):
```bash
cd frontend
python -m http.server 5500
```

7. Open app:
- Frontend: http://localhost:5500
- API docs: http://localhost:8000/docs

### Frontend pages
- `/index.html` → prediction form
- `/models.html` → model metrics dashboard
- `/symptoms.html` → symptom importance explorer
- `/api-playground.html` → request/response tester
- `/about.html` → system overview
