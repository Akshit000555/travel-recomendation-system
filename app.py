"""
🌍 Smart Travel Recommendation System
A premium Streamlit web app for travel & tourism recommendations.

Run locally:
    pip install streamlit pandas numpy plotly scikit-learn textblob
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import time


st.set_page_config(
    page_title="Smart Travel Recommendation System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
        padding: 2.5rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99,102,241,0.25);
    }
    .hero h1 {
        font-size: 2.4rem; font-weight: 800; margin: 0;
    }
    .hero p { opacity: 0.92; margin-top: 0.4rem; font-size: 1.05rem; }

    /* Card style */
    .travel-card {
        background: white;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        border: 1px solid #eef2f7;
        margin-bottom: 1rem;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .travel-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
    }
    .card-title { font-size: 1.25rem; font-weight: 700; color:#0f172a; margin-bottom:.4rem;}
    .card-sub   { color:#64748b; font-size:.92rem; margin-bottom:.6rem;}
    .badge {
        display:inline-block; padding:.25rem .7rem; border-radius:999px;
        font-size:.78rem; font-weight:600; margin-right:.35rem;
    }
    .badge-budget   { background:#dcfce7; color:#166534; }
    .badge-mid      { background:#dbeafe; color:#1e40af; }
    .badge-luxury   { background:#fef3c7; color:#92400e; }
    .badge-pos      { background:#dcfce7; color:#166534; }
    .badge-neu      { background:#e2e8f0; color:#334155; }
    .badge-neg      { background:#fee2e2; color:#991b1b; }

    /* Section headers */
    .section-h {
        font-size:1.5rem; font-weight:800; color:#0f172a;
        margin: 1.5rem 0 .8rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: white !important; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg,#6366f1,#8b5cf6);
        color:white; border:none; padding:.6rem 1.4rem;
        border-radius:10px; font-weight:600;
    }
    .stButton>button:hover { filter: brightness(1.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sample datasets (replace with your own CSVs if available)
# ---------------------------------------------------------------------------
@st.cache_data
def load_destinations() -> pd.DataFrame:
    data = [
        ["Goa",          "Goa",            "Beach",     15000,  95, "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?w=600"],
        ["Andaman",      "Andaman",        "Beach",     35000,  88, "https://images.unsplash.com/photo-1590523741831-ab7e8b8f9c7f?w=600"],
        ["Pondicherry",  "Tamil Nadu",     "Beach",     12000,  80, "https://images.unsplash.com/photo-1582719508461-905c673771fd?w=600"],
        ["Manali",       "Himachal",       "Mountain",  18000,  92, "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?w=600"],
        ["Leh-Ladakh",   "Ladakh",         "Mountain",  40000,  94, "https://images.unsplash.com/photo-1589308078059-be1415eab4c3?w=600"],
        ["Darjeeling",   "West Bengal",    "Mountain",  14000,  82, "https://images.unsplash.com/photo-1544461772-722f499fa1dd?w=600"],
        ["Mumbai",       "Maharashtra",    "City",      20000,  85, "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?w=600"],
        ["Delhi",        "Delhi",          "City",      15000,  87, "https://images.unsplash.com/photo-1587474260584-136574528ed5?w=600"],
        ["Jaipur",       "Rajasthan",      "City",      13000,  90, "https://images.unsplash.com/photo-1477587458883-47145ed94245?w=600"],
        ["Rishikesh",    "Uttarakhand",    "Adventure", 10000,  89, "https://images.unsplash.com/photo-1591017403286-fd8493524e1d?w=600"],
        ["Bir Billing",  "Himachal",       "Adventure", 16000,  78, "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=600"],
        ["Spiti Valley", "Himachal",       "Adventure", 30000,  84, "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600"],
        ["Udaipur",      "Rajasthan",      "Romantic",  25000,  93, "https://images.unsplash.com/photo-1599661046289-e31897846e41?w=600"],
        ["Kerala",       "Kerala",         "Romantic",  22000,  96, "https://images.unsplash.com/photo-1602215955643-2a7c2ab5b84d?w=600"],
        ["Shimla",       "Himachal",       "Romantic",  17000,  86, "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?w=600"],
    ]
    return pd.DataFrame(data, columns=["Name","State","Type","Budget","Popularity","Image"])


@st.cache_data
def load_hotels() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    destinations = ["Goa","Manali","Jaipur","Kerala","Udaipur","Leh-Ladakh",
                    "Rishikesh","Mumbai","Delhi","Andaman","Shimla","Darjeeling"]
    rows = []
    for d in destinations:
        for i in range(1, 6):
            price = int(rng.integers(1500, 25000))
            rating = round(float(rng.uniform(3.2, 4.9)), 1)
            if price < 4000:       seg = "Budget"
            elif price < 12000:    seg = "Mid-range"
            else:                  seg = "Luxury"
            sent = rng.choice(["Positive","Neutral","Negative"], p=[0.6,0.25,0.15])
            rows.append([f"{d} Stay {i}", d, rating, price, seg, sent])
    return pd.DataFrame(rows, columns=["Hotel","Location","Rating","Price","Segment","Sentiment"])


@st.cache_data
def hotel_type_map() -> dict:
    return {
        "Beach":     ["Goa","Andaman"],
        "Mountain":  ["Manali","Leh-Ladakh","Darjeeling","Shimla"],
        "City":      ["Mumbai","Delhi","Jaipur"],
        "Adventure": ["Rishikesh","Leh-Ladakh"],
        "Romantic":  ["Udaipur","Kerala","Shimla"],
    }


destinations_df = load_destinations()
hotels_df       = load_hotels()
type_to_cities  = hotel_type_map()

# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ Plan Your Trip")
    st.markdown("Tell us your preferences and we'll craft recommendations.")
    st.divider()

    budget = st.slider("💰 Your Budget (₹)", 5_000, 50_000, 20_000, step=1_000)
    travel_type = st.selectbox(
        "🏝️ Travel Type",
        ["Beach", "Mountain", "City", "Adventure", "Romantic"],
    )
    st.divider()
    st.caption("© 2025 Smart Travel • Built with Streamlit")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌍 Smart Travel Recommendation System</h1>
        <p>Discover your next adventure — personalized destinations, hotels & insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1) DESTINATION RECOMMENDATION
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">📍 Top Destination For You</div>', unsafe_allow_html=True)

with st.spinner("Finding the perfect destination..."):
    time.sleep(0.4)
    filt = destinations_df[
        (destinations_df["Type"] == travel_type) &
        (destinations_df["Budget"] <= budget)
    ].sort_values("Popularity", ascending=False)

col_a, col_b = st.columns([1.2, 1])

if filt.empty:
    st.warning("😕 No destinations match your budget. Try increasing it or changing travel type.")
    top_dest = None
else:
    top_dest = filt.iloc[0]
    with col_a:
        st.image(top_dest["Image"], use_column_width=True)
    with col_b:
        st.markdown(
            f"""
            <div class="travel-card">
                <div class="card-title">🏖️ {top_dest['Name']}</div>
                <div class="card-sub">{top_dest['State']} • {top_dest['Type']}</div>
                <p><b>Estimated Budget:</b> ₹{top_dest['Budget']:,}</p>
                <p><b>Popularity:</b> ⭐ {top_dest['Popularity']}/100</p>
                <span class="badge badge-mid">{top_dest['Type']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(f"✅ Recommended: **{top_dest['Name']}** — within your ₹{budget:,} budget!")

# ---------------------------------------------------------------------------
# 2) HOTEL RECOMMENDATION
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">🏨 Top Hotel Picks</div>', unsafe_allow_html=True)

target_city = top_dest["Name"] if top_dest is not None else None
candidate_cities = type_to_cities.get(travel_type, [])

hotel_subset = hotels_df[
    hotels_df["Location"].isin(candidate_cities + ([target_city] if target_city else []))
]
hotel_subset = hotel_subset.sort_values(["Rating","Price"], ascending=[False, True]).head(5)

if hotel_subset.empty:
    st.info("No hotels found for this selection.")
else:
    st.dataframe(hotel_subset, use_container_width=True, hide_index=True)

    st.markdown("#### 🏷️ Hotel Cards")
    cols = st.columns(min(5, len(hotel_subset)))
    for col, (_, h) in zip(cols, hotel_subset.iterrows()):
        seg_class = {"Budget":"badge-budget","Mid-range":"badge-mid","Luxury":"badge-luxury"}[h["Segment"]]
        sen_class = {"Positive":"badge-pos","Neutral":"badge-neu","Negative":"badge-neg"}[h["Sentiment"]]
        sen_emoji = {"Positive":"😊","Neutral":"😐","Negative":"😡"}[h["Sentiment"]]
        with col:
            st.markdown(
                f"""
                <div class="travel-card">
                    <div class="card-title">{h['Hotel']}</div>
                    <div class="card-sub">📍 {h['Location']}</div>
                    <p>⭐ <b>{h['Rating']}</b> &nbsp; • &nbsp; ₹{h['Price']:,}</p>
                    <span class="badge {seg_class}">{h['Segment']}</span>
                    <span class="badge {sen_class}">{sen_emoji} {h['Sentiment']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# 3) CUSTOMER SEGMENTATION VISUALIZATION
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">📊 Customer Segmentation — Price vs Rating</div>', unsafe_allow_html=True)

seg_fig = px.scatter(
    hotels_df, x="Price", y="Rating",
    color="Segment",
    color_discrete_map={"Budget":"#22c55e","Mid-range":"#3b82f6","Luxury":"#f59e0b"},
    hover_data=["Hotel","Location","Sentiment"],
    title="Hotel Segments by Price and Rating",
)
seg_fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
seg_fig.update_layout(
    plot_bgcolor="white", height=450,
    legend=dict(orientation="h", y=-0.2),
    xaxis_title="Price (₹)", yaxis_title="Rating (out of 5)",
)
st.plotly_chart(seg_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 4) SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">💬 Review Sentiment Analyzer</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1])

with c1:
    review_text = st.text_area(
        "Write a review and we'll analyze its sentiment:",
        placeholder="e.g. 'The stay was amazing, rooms were clean and staff very friendly!'",
        height=140,
    )
    if st.button("🔍 Analyze Sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.4)
                polarity = TextBlob(review_text).sentiment.polarity
                if polarity > 0.15:
                    st.success(f"😊 **Positive** review  (score: {polarity:.2f})")
                elif polarity < -0.15:
                    st.error(f"😡 **Negative** review  (score: {polarity:.2f})")
                else:
                    st.info(f"😐 **Neutral** review  (score: {polarity:.2f})")

with c2:
    sent_counts = hotels_df["Sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment","Count"]
    sent_fig = px.bar(
        sent_counts, x="Sentiment", y="Count", color="Sentiment",
        color_discrete_map={"Positive":"#22c55e","Neutral":"#94a3b8","Negative":"#ef4444"},
        title="Sentiment Distribution (Dataset)",
    )
    sent_fig.update_layout(plot_bgcolor="white", showlegend=False, height=380)
    st.plotly_chart(sent_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 5) DATA VISUALIZATIONS
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">📈 Explore the Data</div>', unsafe_allow_html=True)

v1, v2 = st.columns(2)

with v1:
    top_pop = destinations_df.sort_values("Popularity", ascending=False).head(10)
    fig = px.bar(
        top_pop, x="Popularity", y="Name", orientation="h",
        color="Popularity", color_continuous_scale="Blues",
        title="Top Destinations by Popularity",
    )
    fig.update_layout(plot_bgcolor="white", yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, use_container_width=True)

with v2:
    fig2 = px.histogram(
        hotels_df, x="Price", nbins=20, color="Segment",
        color_discrete_map={"Budget":"#22c55e","Mid-range":"#3b82f6","Luxury":"#f59e0b"},
        title="Hotel Price Distribution",
    )
    fig2.update_layout(plot_bgcolor="white", bargap=0.05, height=420)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("✨ Built with ❤️ using Streamlit, Plotly & TextBlob — plug in your own dataset to go live.")
