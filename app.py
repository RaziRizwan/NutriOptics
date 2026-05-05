import os
import json
import re
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import easyocr
from openai import OpenAI
import time
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NutriOptics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .stTitle {
    font-family: 'Syne', sans-serif !important;
}

/* Dark card style for metric containers */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    color: white;
}

.risk-red {
    background: linear-gradient(135deg, #3d0000, #6b0000);
    border-left: 4px solid #ff4444;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #ffcccc;
    font-size: 0.9rem;
}

.risk-yellow {
    background: linear-gradient(135deg, #3d2e00, #5c4400);
    border-left: 4px solid #ffcc00;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #fff3cc;
    font-size: 0.9rem;
}

.risk-green {
    background: linear-gradient(135deg, #003d0e, #005c15);
    border-left: 4px solid #00cc44;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #ccffe0;
    font-size: 0.9rem;
}

.scan-history-card {
    background: #1e1e2e;
    border: 1px solid #333355;
    border-radius: 12px;
    padding: 14px;
    margin: 6px 0;
    cursor: pointer;
    transition: border-color 0.2s;
}

.scan-history-card:hover {
    border-color: #6666ff;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.badge-red { background: #ff444422; color: #ff4444; border: 1px solid #ff4444; }
.badge-yellow { background: #ffcc0022; color: #ffcc00; border: 1px solid #ffcc00; }
.badge-green { background: #00cc4422; color: #00cc44; border: 1px solid #00cc44; }

.clarity-bar-container {
    background: #1e1e2e;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0ff;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

.compare-header {
    font-family: 'Syne', sans-serif;
    background: linear-gradient(90deg, #6666ff, #00ccaa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.5rem;
    font-weight: 800;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a14 100%);
}

[data-testid="stSidebar"] * {
    color: #ccccee !important;
}

/* Serving slider */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6666ff, #00ccaa) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False


# ─────────────────────────────────────────────
# OCR UTILITIES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)


def preprocess_image_for_ocr(image_array):
    """
    Auto-crop to the nutrition label region using contour detection,
    then enhance contrast for better OCR accuracy.
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Edge detection + contour to find label region
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest rectangular contour (likely the label)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area_ratio = (w * h) / (image_array.shape[0] * image_array.shape[1])

        # Only crop if it covers between 10% and 95% of image (avoids bad crops)
        if 0.10 < area_ratio < 0.95:
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_array.shape[1], x + w + padding)
            y2 = min(image_array.shape[0], y + h + padding)
            cropped = image_array[y1:y2, x1:x2]
            return cropped, True

    return image_array, False  # No good crop found


def extract_text_from_image(image, reader):
    image_array = np.array(image)

    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    processed_array, was_cropped = preprocess_image_for_ocr(image_array)
    results = reader.readtext(processed_array)

    text_blocks = []
    confidences = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            y_coord = bbox[0][1]
            text_blocks.append((y_coord, text))
            confidences.append(confidence)

    text_blocks.sort(key=lambda x: x[0])
    extracted_text = " ".join([text for _, text in text_blocks])

    avg_confidence = np.mean(confidences) if confidences else 0
    clarity_score = round(avg_confidence * 100, 1)

    return extracted_text.strip(), clarity_score, was_cropped


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def build_prompt(text, health_profile, servings):
    conditions = health_profile.get("conditions", [])
    allergies = health_profile.get("allergies", "").strip()
    goal = health_profile.get("goal", "General wellness")
    conditions_str = ", ".join(conditions) if conditions else "None"

    return f"""You are a clinical nutritionist. Analyze this food label for a person with:
- Medical conditions: {conditions_str}
- Allergies/intolerances: {allergies if allergies else "None"}
- Health goal: {goal}
- Serving size multiplier: {servings}x (adjust all nutrient values accordingly)

Food label text:
{text}

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{
  "summary": "One sentence personalized summary based on the user's profile",
  "overall_rating": <integer 1-10>,
  "diabetes": {{"status": "safe|caution|avoid", "note": "one-liner"}},
  "blood_pressure": {{"status": "safe|caution|avoid", "note": "one-liner"}},
  "heart_health": {{"status": "safe|caution|avoid", "note": "one-liner"}},
  "allergies": {{"status": "safe|caution|avoid", "note": "one-liner"}},
  "other_warnings": "one-liner or null",
  "nutrients": {{
    "sugar_g": <number or null>,
    "sodium_mg": <number or null>,
    "saturated_fat_g": <number or null>,
    "calories": <number or null>
  }},
  "highlights": {{
    "red": ["list of high-risk ingredients or concerns"],
    "yellow": ["list of moderate-risk items"],
    "green": ["list of beneficial ingredients or positives"]
  }}
}}"""


def get_hf_api_key():
    """Read the Hugging Face key from env vars or Streamlit secrets."""
    env_key = os.getenv("HF_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets.get("HF_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def query_llm(text, health_profile, servings=1):
    prompt = build_prompt(text, health_profile, servings)
    api_key = get_hf_api_key()
    if not api_key:
        st.error("Missing Hugging Face API key.")
        st.info("On Streamlit Cloud, add `HF_KEY` in Manage app -> Settings -> Secrets, then redeploy the app.")
        st.stop()

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key
    )
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:fireworks-ai",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback — return raw as summary
        return {
            "summary": raw[:300],
            "overall_rating": 5,
            "diabetes": {"status": "caution", "note": "Could not parse structured response."},
            "blood_pressure": {"status": "caution", "note": "Could not parse structured response."},
            "heart_health": {"status": "caution", "note": "Could not parse structured response."},
            "allergies": {"status": "safe", "note": "No allergy data parsed."},
            "other_warnings": None,
            "nutrients": {"sugar_g": None, "sodium_mg": None, "saturated_fat_g": None, "calories": None},
            "highlights": {"red": [], "yellow": [], "green": []}
        }


# ─────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────
STATUS_COLORS = {
    "safe": "#00cc44",
    "caution": "#ffcc00",
    "avoid": "#ff4444"
}
STATUS_ICONS = {"safe": "✅", "caution": "⚠️", "avoid": "🚫"}


def normalize_plotly_color(color):
    """Convert hex color strings with alpha to Plotly-compatible rgba strings."""
    if isinstance(color, str) and color.startswith("#"):
        hex_value = color.lstrip("#")
        if len(hex_value) == 8:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            a = int(hex_value[6:8], 16) / 255
            return f"rgba({r},{g},{b},{a:.3f})"
    return color


def render_gauge(label, value, max_val, unit, thresholds):
    """Render a plotly radial gauge for a nutrient."""
    if value is None:
        return None

    low, high = thresholds
    if value <= low:
        color = "#00cc44"
    elif value <= high:
        color = "#ffcc00"
    else:
        color = "#ff4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 18, "color": "white", "family": "DM Sans"}},
        title={"text": label, "font": {"size": 13, "color": "#aaaacc", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#444466", "tickfont": {"color": "#666688", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#333355",
            "steps": [
                {"range": [0, low], "color": normalize_plotly_color("#00cc4415")},
                {"range": [low, high], "color": normalize_plotly_color("#ffcc0015")},
                {"range": [high, max_val], "color": normalize_plotly_color("#ff444415")},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    return fig


def render_health_factor_row(label, factor_data):
    status = factor_data.get("status", "caution")
    note = factor_data.get("note", "")
    icon = STATUS_ICONS.get(status, "⚠️")
    css_class = f"risk-{'red' if status=='avoid' else 'yellow' if status=='caution' else 'green'}"
    st.markdown(f"""
    <div class="{css_class}">
        <strong>{icon} {label}</strong>: {note}
    </div>
    """, unsafe_allow_html=True)


def render_highlights(highlights):
    st.markdown('<div class="section-header">🔬 Ingredient Risk Breakdown</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔴 Avoid / High Risk**")
        items = highlights.get("red", [])
        if items:
            for item in items:
                st.markdown(f'<span class="badge badge-red">⛔ {item}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-green">None found</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("**🟡 Limit / Moderate**")
        items = highlights.get("yellow", [])
        if items:
            for item in items:
                st.markdown(f'<span class="badge badge-yellow">⚠️ {item}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-green">None found</span>', unsafe_allow_html=True)

    with col3:
        st.markdown("**🟢 Beneficial**")
        items = highlights.get("green", [])
        if items:
            for item in items:
                st.markdown(f'<span class="badge badge-green">✅ {item}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">None identified</span>', unsafe_allow_html=True)


def render_nutrient_gauges(nutrients, servings):
    st.markdown('<div class="section-header">📊 Nutrient Gauges</div>', unsafe_allow_html=True)
    st.caption(f"Values shown for **{servings} serving(s)**. Daily reference values (DRV) used as max scale.")

    # Scale nutrients by servings
    def scale(v):
        return round(v * servings, 1) if v is not None else None

    gauges = [
        ("Calories", scale(nutrients.get("calories")), 2000, "kcal", (400, 800)),
        ("Sugar", scale(nutrients.get("sugar_g")), 50, "g", (10, 25)),
        ("Sodium", scale(nutrients.get("sodium_mg")), 2300, "mg", (600, 1500)),
        ("Saturated Fat", scale(nutrients.get("saturated_fat_g")), 20, "g", (5, 13)),
    ]

    cols = st.columns(4)
    for col, (label, value, max_val, unit, thresholds) in zip(cols, gauges):
        with col:
            fig = render_gauge(label, value, max_val, unit, thresholds)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(f"**{label}**\n\n*Not detected*")


def render_analysis_results(result, extracted_text, clarity_score, was_cropped, servings, product_name=""):
    # ── Summary banner
    rating = result.get("overall_rating", 5)
    rating_color = "#00cc44" if rating >= 7 else "#ffcc00" if rating >= 4 else "#ff4444"
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-family:Syne; font-size:1.1rem; color:#aaaacc; margin-bottom:4px;">
                    {"📦 " + product_name if product_name else "📦 Scanned Product"}
                </div>
                <div style="font-size:1rem; color:#e0e0ff;">{result.get("summary", "")}</div>
            </div>
            <div style="text-align:center; min-width:80px;">
                <div style="font-family:Syne; font-size:2.5rem; font-weight:800; color:{rating_color};">{rating}</div>
                <div style="font-size:0.75rem; color:#888899;">/ 10</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── OCR Clarity
    clarity_color = "#00cc44" if clarity_score >= 70 else "#ffcc00" if clarity_score >= 40 else "#ff4444"
    st.markdown(f"""
    <div style="margin:8px 0; font-size:0.85rem; color:#aaaacc;">
        🔎 OCR Text Clarity: <strong style="color:{clarity_color}">{clarity_score}%</strong>
        {"&nbsp;&nbsp;✂️ <em>Label region auto-cropped</em>" if was_cropped else ""}
        {"&nbsp;&nbsp;⚠️ <em>Low clarity — try a clearer photo for better results</em>" if clarity_score < 40 else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Serving multiplier slider
    st.markdown('<div class="section-header">🍽️ Serving Size Adjustment</div>', unsafe_allow_html=True)
    new_servings = st.slider(
        "How many servings will you eat?",
        min_value=0.5, max_value=5.0, value=float(servings), step=0.5,
        key=f"serving_slider_{product_name}"
    )

    # ── Nutrient Gauges
    render_nutrient_gauges(result.get("nutrients", {}), new_servings)

    # ── Health Factors
    st.markdown('<div class="section-header">🩺 Personalized Health Factors</div>', unsafe_allow_html=True)
    factors = {
        "Diabetes": result.get("diabetes", {}),
        "Blood Pressure": result.get("blood_pressure", {}),
        "Heart Health": result.get("heart_health", {}),
        "Allergies / Intolerances": result.get("allergies", {}),
    }
    for label, data in factors.items():
        render_health_factor_row(label, data)

    other = result.get("other_warnings")
    if other:
        st.markdown(f'<div class="risk-yellow">⚠️ <strong>Other:</strong> {other}</div>', unsafe_allow_html=True)

    # ── Ingredient highlights
    render_highlights(result.get("highlights", {}))

    # ── Extracted text expander
    with st.expander("🔤 View Extracted Text", expanded=False):
        st.code(extracted_text, language=None)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("logo.jpeg", use_container_width=True)

    st.markdown("---")
    with st.expander("Instructions & Tips", expanded=False):
        st.markdown("""
        **How it works:**
        1. Set your health profile on the home page
        2. Upload a food nutrition label photo
        3. AI extracts and analyzes the label
        4. Review your personalized health insights
        """)
        st.markdown("""
        **Tips for best results:**
        - Use clear, well-lit photos
        - Ensure the full nutrition table is visible
        - Avoid blurry or angled images
        - Text must be readable and uncut
        """)

    st.markdown("---")

    # ── Scan history
    st.markdown("### 🕘 Scan History")
    if st.session_state.scan_history:
        for i, scan in enumerate(reversed(st.session_state.scan_history[-5:])):
            rating = scan["result"].get("overall_rating", "?")
            rating_color = "🟢" if rating >= 7 else "🟡" if rating >= 4 else "🔴"
            st.markdown(f"""
            <div class="scan-history-card">
                {rating_color} <strong>{scan['name']}</strong><br>
                <span style="font-size:0.75rem; color:#888899;">Score: {rating}/10 · {scan['time']}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.caption("No scans yet.")

    # ── Compare toggle
    st.markdown("---")
    compare_mode = st.toggle("⚖️ Compare Two Products", value=st.session_state.compare_mode)
    st.session_state.compare_mode = compare_mode

    st.markdown("---")
    st.markdown("""
    **About NutriOptics**  
    AI-powered food label analysis for smarter, healthier choices.
    """)
    st.markdown("""
    **Developed by Muhammad Razi**  
    <a href="https://github.com/razirizwan" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16" style="vertical-align:middle; margin-right:4px;"></a>
    <a href="https://www.linkedin.com/in/razirizwan/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" style="vertical-align:middle;"></a>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:Syne; font-weight:800; font-size:2.2rem; margin-bottom:0;">
    🔍 NutriOptics
</h1>
<p style="color:#8888aa; font-size:1rem; margin-top:4px;">See. Scan. Eat Healthy.</p>
""", unsafe_allow_html=True)

st.markdown("### 👤 Your Health Profile")
st.caption("Personalize the AI analysis to your conditions before scanning a label.")

profile_col1, profile_col2, profile_col3 = st.columns([1.2, 1, 1])
with profile_col1:
    conditions = st.multiselect(
        "Medical Conditions",
        ["Diabetes (Type 2)", "Type 1 Diabetes", "Hypertension", "Heart Disease",
         "High Cholesterol", "Celiac Disease", "Kidney Disease", "Obesity"],
        default=[],
        key="conditions"
    )

with profile_col2:
    allergies = st.text_input(
        "Allergies / Intolerances",
        placeholder="e.g. gluten, lactose, peanuts",
        key="allergies"
    )

with profile_col3:
    goal = st.selectbox(
        "Primary Health Goal",
        ["General wellness", "Weight loss", "Muscle gain",
         "Blood sugar control", "Heart health", "Low sodium diet"],
        key="goal"
    )

health_profile = {
    "conditions": conditions,
    "allergies": allergies,
    "goal": goal
}

st.markdown("---")


# ─────────────────────────────────────────────
# SINGLE SCAN MODE
# ─────────────────────────────────────────────
def run_scan(uploaded_file, label="Product", key_suffix=""):
    """Run OCR + LLM pipeline for a single uploaded file."""
    progress = st.progress(0, text="Starting...")
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded: {label}", use_container_width=True)

    progress.progress(20, text="🔍 Preprocessing image...")
    reader = load_ocr_reader()

    progress.progress(40, text="📝 Extracting text via OCR...")
    extracted_text, clarity_score, was_cropped = extract_text_from_image(image, reader)

    progress.progress(70, text="🤖 Analyzing with AI...")
    result = query_llm(extracted_text, health_profile, servings=1)

    progress.progress(100, text="✅ Done!")
    time.sleep(0.3)
    progress.empty()

    # Save to history
    import datetime
    st.session_state.scan_history.append({
        "name": label,
        "result": result,
        "extracted_text": extracted_text,
        "clarity_score": clarity_score,
        "was_cropped": was_cropped,
        "time": datetime.datetime.now().strftime("%H:%M")
    })

    return result, extracted_text, clarity_score, was_cropped


if not st.session_state.compare_mode:
    # ── Single product mode
    uploaded_file = st.file_uploader(
        "📂 Upload a nutrition label image",
        type=["png", "jpg", "jpeg"],
        key="single_upload"
    )

    if uploaded_file:
        product_name = st.text_input("Product name (optional)", placeholder="e.g. Oreo Original", key="product_name_single")
        if st.button("🔬 Analyze Label", type="primary", key="analyze_single"):
            with st.spinner(""):
                result, extracted_text, clarity_score, was_cropped = run_scan(
                    uploaded_file,
                    label=product_name or uploaded_file.name
                )
            st.success("✅ Analysis Complete!")
            render_analysis_results(result, extracted_text, clarity_score, was_cropped, servings=1, product_name=product_name)

else:
    # ─────────────────────────────────────────
    # COMPARE MODE
    # ─────────────────────────────────────────
    st.markdown('<div class="compare-header">⚖️ Side-by-Side Comparison</div>', unsafe_allow_html=True)
    st.caption("Upload two products to compare their health profiles head-to-head.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Product A")
        file_a = st.file_uploader("Upload Product A", type=["png", "jpg", "jpeg"], key="upload_a")
        name_a = st.text_input("Name (optional)", placeholder="e.g. Brand A Chips", key="name_a")

    with col_b:
        st.markdown("### Product B")
        file_b = st.file_uploader("Upload Product B", type=["png", "jpg", "jpeg"], key="upload_b")
        name_b = st.text_input("Name (optional)", placeholder="e.g. Brand B Chips", key="name_b")

    if file_a and file_b:
        if st.button("⚖️ Compare Both Products", type="primary"):
            col_a_res, col_b_res = st.columns(2)

            with col_a_res:
                st.markdown(f"### 🅰️ {name_a or 'Product A'}")
                with st.spinner("Analyzing A..."):
                    res_a, text_a, clarity_a, cropped_a = run_scan(file_a, label=name_a or "Product A", key_suffix="a")
                render_analysis_results(res_a, text_a, clarity_a, cropped_a, servings=1, product_name=name_a or "Product A")

            with col_b_res:
                st.markdown(f"### 🅱️ {name_b or 'Product B'}")
                with st.spinner("Analyzing B..."):
                    res_b, text_b, clarity_b, cropped_b = run_scan(file_b, label=name_b or "Product B", key_suffix="b")
                render_analysis_results(res_b, text_b, clarity_b, cropped_b, servings=1, product_name=name_b or "Product B")

            # ── Winner summary
            st.markdown("---")
            st.markdown('<div class="section-header">🏆 Verdict</div>', unsafe_allow_html=True)
            rating_a = res_a.get("overall_rating", 5)
            rating_b = res_b.get("overall_rating", 5)

            if rating_a > rating_b:
                winner = name_a or "Product A"
                winner_score = rating_a
                loser_score = rating_b
                emoji = "🅰️"
            elif rating_b > rating_a:
                winner = name_b or "Product B"
                winner_score = rating_b
                loser_score = rating_a
                emoji = "🅱️"
            else:
                winner = None

            if winner:
                st.success(f"{emoji} **{winner}** wins with a health score of **{winner_score}/10** vs {loser_score}/10 — the healthier choice for your profile.")
            else:
                st.info("⚖️ Both products scored equally. Review individual factors to decide.")
    elif file_a or file_b:
        st.warning("Please upload both products to compare.")
