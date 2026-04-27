# 🌍 Smart Travel Recommendation System

A personalized travel recommendation web app built with Streamlit. Get destination and hotel suggestions based on your budget and travel preferences, with sentiment analysis on reviews.

## Features

- 📍 Destination recommendations by budget & travel type
- 🏨 Hotel picks with segment & sentiment tags
- 📊 Customer segmentation visualization
- 💬 Live review sentiment analyzer

## Tech Stack

- [Streamlit](https://streamlit.io/)
- Pandas, NumPy, Plotly, TextBlob, Scikit-learn

## Run Locally

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
streamlit run app.py
```

## Deploy on Render

- Build Command: `pip install -r requirements.txt && python -m textblob.download_corpora`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
