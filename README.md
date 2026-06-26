# 🐱 vs 🐶 — Image Classifier

A small web app that uses a pre-trained machine learning model (MobileNetV2)
to guess whether an uploaded photo is a cat or a dog.

**Stack:** Flask (backend) · HTML/CSS/JS (frontend) · TensorFlow (ML model) · GitHub (version control) · Render (hosting)

---

## How it works

1. You upload a photo on the web page.
2. The Flask backend receives it and runs it through MobileNetV2 — a model
   already trained by Google on 1.4 million images.
3. We check the model's prediction against lists of cat/dog-related labels
   and show you the verdict, plus a confidence score.

No training required — this is called "transfer learning lite": using
someone else's trained model as-is, rather than training your own from scratch.

**Why TensorFlow Lite instead of full TensorFlow?** Free and low-cost cloud
hosting (like Render's free or Starter tier) gives you only 512MB of RAM.
Full TensorFlow alone can need 400-600MB just to load — too tight a squeeze.
TensorFlow **Lite** is a stripped-down runtime built specifically for
running pre-trained models on constrained hardware (it's the same tech
used to run ML models on phones), so it fits comfortably in that budget.
The model itself (MobileNetV2) is identical either way — same predictions,
much lighter footprint.

---

## Running it on your own computer (optional, before deploying)

You don't have to do this — you can skip straight to deployment below.
But if you want to test it locally first:

```bash
# 1. Create a virtual environment (keeps dependencies isolated)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

> First run will take a minute or two — it's downloading the MobileNetV2
> model weights (~14MB) automatically.

---

## Deploying — step by step

### 1. Put this code on GitHub
- Create a free account at [github.com](https://github.com) if you don't have one.
- Create a new repository (e.g. `cat-dog-classifier`). Keep it **public** if
  you're on Render's free tier and want zero friction; private also works.
- Upload all the files in this folder to that repo (drag-and-drop works fine
  on GitHub's web interface, or use `git push` if you're comfortable with Git).

### 2. Deploy on Render
- Create a free account at [render.com](https://render.com).
- Click **New +** → **Web Service**.
- Connect your GitHub account and select your new repo.
- Render will detect the `render.yaml` file and pre-fill the settings. If it
  asks manually instead:
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `gunicorn app:app --timeout 120`
- Click **Create Web Service**.

### 3. Wait for the build
- First build takes a few minutes (TensorFlow is a big library).
- Once it says "Live," Render gives you a public URL like
  `https://cat-dog-classifier.onrender.com` — that's your live app!

> **Free tier note:** Render's free web services "spin down" after periods
> of inactivity, so the first request after a while might take 30-60 seconds
> to wake back up. This is normal — just a free-tier quirk, not a bug.

---

## Project structure

```
cat-dog-classifier/
├── app.py                # Flask backend + ML logic
├── templates/
│   └── index.html        # The web page
├── static/
│   └── style.css          # Styling
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
└── README.md
```

---

## Ideas for leveling this up later

- Swap in a model you've fine-tuned yourself on a custom cat/dog dataset
  (more accurate than relying on general ImageNet labels).
- Add a "history" of past uploads using a small database.
- Let users confirm/correct the prediction, and log corrections — your
  first taste of collecting feedback data for retraining.
