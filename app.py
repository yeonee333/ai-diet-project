import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================
# 페이지 설정
# =====================================
st.set_page_config(
    page_title="NutriAI — 스마트 식단 관리",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# 고급 CSS 스타일 주입
# =====================================
st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background-color: #0a0a0f;
    color: #f0f0f8;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #15151f !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #f0f0f8 !important;
}

/* 메인 폰트 */
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
}

/* 헤더 */
h1 {
    background: linear-gradient(135deg, #7c6fff, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    color: #f0f0f8 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    border-left: 4px solid #7c6fff;
    padding-left: 12px;
    margin-top: 2rem !important;
}
h3 {
    color: #b39ddb !important;
    font-size: 1rem !important;
}

/* 카드 스타일 */
.nutricard {
    background: #15151f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* 메트릭 카드 */
[data-testid="metric-container"] {
    background: #1a1a24 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8080a0 !important;
    font-size: 11px !important;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f0f8 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* 입력 필드 */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background-color: #1a1a24 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0f0f8 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #7c6fff !important;
    box-shadow: 0 0 0 2px rgba(124,111,255,0.2) !important;
}

/* 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #7c6fff, #ff6b9d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,111,255,0.35) !important;
}

/* 파일 업로더 */
[data-testid="stFileUploader"] {
    background: #1a1a24 !important;
    border: 2px dashed rgba(124,111,255,0.4) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c6fff !important;
}

/* 코드 블록 (목표 표시) */
.stCodeBlock {
    background: #1a1a24 !important;
    border: 1px solid rgba(124,111,255,0.2) !important;
    border-radius: 10px !important;
}
.stCodeBlock code {
    color: #b39ddb !important;
}

/* info/success/warning/error 박스 */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* selectbox 드롭다운 */
[data-testid="stSelectbox"] div[role="listbox"] {
    background-color: #1a1a24 !important;
    border: 1px solid rgba(124,111,255,0.3) !important;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1a1a24;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8080a0 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background-color: #7c6fff !important;
    color: white !important;
}

/* progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c6fff, #ff6b9d) !important;
    border-radius: 10px !important;
}
.stProgress > div {
    background-color: #1a1a24 !important;
    border-radius: 10px !important;
}

/* 구분선 */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* 스크롤바 */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,111,255,0.3); border-radius: 3px; }

/* 이미지 */
[data-testid="stImage"] img {
    border-radius: 12px !important;
}

/* caption */
.stCaption {
    color: #8080a0 !important;
}

/* number input 버튼 */
[data-testid="stNumberInput"] button {
    background: #1a1a24 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8080a0 !important;
}

/* 사이드바 warning */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255,77,109,0.1) !important;
    border: 1px solid rgba(255,77,109,0.3) !important;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# =====================================
# 모델 불러오기
# =====================================
@st.cache_resource
def load_food_model():
    return tf.keras.models.load_model("food_model.h5")

try:
    model = load_food_model()
    model_loaded = True
except Exception as e:
    st.error(f"모델을 불러오는데 실패했습니다: {e}")
    model_loaded = False

# =====================================
# 세션 초기화
# =====================================
defaults = {
    "total_kcal": 0, "total_protein": 0, "total_carb": 0, "total_fat": 0,
    "foods": [], "night_count": 0,
    "weekly_scores": [], "weekly_calories": [], "weekly_details": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================
# 클래스 이름 및 영양성분 DB
# =====================================
class_names = [
    "bibimbap", "chicken_wings", "donuts", "dumplings", "fried_rice",
    "hamburger", "ice_cream", "omelette", "pho", "pizza",
    "ramen", "spaghetti", "sushi", "tacos", "waffles"
]
food_info = {
    "bibimbap":      {"name": "비빔밥",     "emoji": "🍚", "kcal": 550, "carb": 80, "protein": 18, "fat": 12},
    "chicken_wings": {"name": "치킨윙",     "emoji": "🍗", "kcal": 320, "carb": 8,  "protein": 24, "fat": 22},
    "donuts":        {"name": "도넛",       "emoji": "🍩", "kcal": 250, "carb": 32, "protein": 3,  "fat": 12},
    "dumplings":     {"name": "만두",       "emoji": "🥟", "kcal": 350, "carb": 45, "protein": 15, "fat": 12},
    "fried_rice":    {"name": "볶음밥",     "emoji": "🍳", "kcal": 700, "carb": 95, "protein": 20, "fat": 25},
    "hamburger":     {"name": "햄버거",     "emoji": "🍔", "kcal": 550, "carb": 45, "protein": 25, "fat": 30},
    "ice_cream":     {"name": "아이스크림", "emoji": "🍦", "kcal": 210, "carb": 25, "protein": 4,  "fat": 11},
    "omelette":      {"name": "오믈렛",     "emoji": "🍳", "kcal": 250, "carb": 5,  "protein": 18, "fat": 17},
    "pho":           {"name": "쌀국수",     "emoji": "🍜", "kcal": 450, "carb": 65, "protein": 20, "fat": 10},
    "pizza":         {"name": "피자",       "emoji": "🍕", "kcal": 850, "carb": 90, "protein": 35, "fat": 40},
    "ramen":         {"name": "라면",       "emoji": "🍜", "kcal": 500, "carb": 75, "protein": 10, "fat": 18},
    "spaghetti":     {"name": "스파게티",   "emoji": "🍝", "kcal": 700, "carb": 85, "protein": 25, "fat": 25},
    "sushi":         {"name": "초밥",       "emoji": "🍣", "kcal": 500, "carb": 65, "protein": 25, "fat": 10},
    "tacos":         {"name": "타코",       "emoji": "🌮", "kcal": 300, "carb": 30, "protein": 12, "fat": 15},
    "waffles":       {"name": "와플",       "emoji": "🧇", "kcal": 450, "carb": 60, "protein": 8,  "fat": 20},
}

# =====================================
# 등급 / 채점 함수
# =====================================
def get_grade(score):
    if score <= 30:   return "주의", "🔴", "#FF4B4B"
    elif score <= 70: return "보통", "🟡", "#FFA500"
    else:             return "충족", "🔵", "#1E90FF"

def grade_badge(score, label=""):
    grade, icon, color = get_grade(score)
    prefix = f"{label} " if label else ""
    return (
        f"<div style='display:inline-block;padding:7px 20px;border-radius:20px;"
        f"background:{color}22;border:2px solid {color};color:{color};"
        f"font-weight:700;font-size:15px;'>"
        f"{icon} {prefix}{grade} &nbsp;|&nbsp; {score}점</div>"
    )

def evaluate_score(tk, tp, tc, tf_, gk, gp, gc, gf):
    details = {}
    for label, actual, target in [("칼로리",tk,gk),("탄수화물",tc,gc),("단백질",tp,gp),("지방",tf_,gf)]:
        r = actual / max(target, 1)
        unit = "kcal" if label == "칼로리" else "g"
        tight = (0.9,1.1) if label == "칼로리" else (0.85,1.15)
        loose = (0.75,1.25) if label == "칼로리" else (0.6,1.4)
        if tight[0] <= r <= tight[1]:   s, st_ = 25, "충족"
        elif loose[0] <= r <= loose[1]: s, st_ = 15, "보통"
        else:                            s, st_ = 5,  "주의"
        details[label] = {"score":s,"status":st_,"actual":actual,"target":round(target),"unit":unit}
    return sum(d["score"] for d in details.values()), details

def macro_progress_html(label, actual, goal, color):
    pct = min(100, round(actual / max(goal,1) * 100)) if goal else 0
    return f"""
    <div style='margin-bottom:14px;'>
      <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
        <span style='font-size:13px;color:#8080a0;'>{label}</span>
        <span style='font-size:13px;color:{color};font-weight:600;'>{actual} / {round(goal) if goal else '--'}{'kcal' if label=='칼로리' else 'g'}</span>
      </div>
      <div style='background:#1a1a24;border-radius:6px;height:8px;overflow:hidden;'>
        <div style='width:{pct}%;height:100%;background:{color};border-radius:6px;transition:width 0.6s ease;'></div>
      </div>
    </div>"""

# =====================================
# 사이드바
# =====================================
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px;'>
      <div style='font-size:22px;font-weight:800;background:linear-gradient(135deg,#7c6fff,#ff6b9d);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>NutriAI</div>
      <div style='font-size:11px;color:#8080a0;letter-spacing:1px;'>스마트 식단 관리</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 오늘 요약
    st.markdown("**📊 오늘 현황**")
    st.metric("섭취 칼로리", f"{st.session_state.total_kcal} kcal")
    st.metric("등록 식사 수", f"{len(st.session_state.foods)}끼")

    if st.session_state.night_count >= 3:
        st.error(f"🚨 야식 {st.session_state.night_count}회! 야식을 줄여주세요.")

    st.markdown("---")
    if st.button("🔄 데이터 초기화"):
        st.session_state.clear()
        st.rerun()

# =====================================
# 타이틀
# =====================================
st.markdown("<h1>🍱 NutriAI 식단 관리 서비스</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8080a0;margin-bottom:32px;'>AI 기반 음식 인식 · 맞춤 영양 목표 · 하루 평가 · 주간 리포트</p>", unsafe_allow_html=True)

# =====================================
# 탭 구성
# =====================================
tab1, tab2, tab3, tab4 = st.tabs(["👤 프로필 & 목표", "📸 음식 등록", "💯 하루 평가", "📅 주간 리포트"])

# ============================================================
# TAB 1 — 프로필 & 목표
# ============================================================
with tab1:
    st.markdown("## 👤 프로필 설정")
    st.markdown("<p style='color:#8080a0;'>정보를 입력하면 BMR 기반으로 맞춤 목표가 자동 계산됩니다.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("성별", ["남", "여"], index=None, placeholder="선택하세요")
        age    = st.number_input("나이 (세)", min_value=1, max_value=100, value=None, placeholder="25")
    with col2:
        height = st.number_input("키 (cm)", min_value=100, max_value=250, value=None, placeholder="170")
        weight = st.number_input("몸무게 (kg)", min_value=30, max_value=200, value=None, placeholder="65")

    goal = st.selectbox("목표", ["다이어트 (체중 감량)", "유지 (체중 유지)", "벌크업 (근육 증가)"], index=None, placeholder="선택하세요")
    info_complete = None not in (gender, age, height, weight, goal)

    if info_complete:
        bmr = (10*weight + 6.25*height - 5*age + 5) if gender=="남" else (10*weight + 6.25*height - 5*age - 161)
        maintain = bmr * 1.55
        if "다이어트" in goal:   target_calorie = maintain - 500
        elif "벌크업" in goal:   target_calorie = maintain + 500
        else:                     target_calorie = maintain
        target_protein = weight * 1.5
        target_fat     = weight * 0.8
        target_carb    = max((target_calorie - target_protein*4 - target_fat*9) / 4, 0)

        st.markdown("## 🎯 오늘 목표")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔥 칼로리", f"{round(target_calorie)} kcal")
        c2.metric("🌾 탄수화물", f"{round(target_carb)}g")
        c3.metric("💪 단백질", f"{round(target_protein)}g")
        c4.metric("🥑 지방", f"{round(target_fat)}g")

        st.markdown("## 📊 섭취 현황")
        st.markdown(macro_progress_html("칼로리",   st.session_state.total_kcal,    target_calorie,  "#ff6b9d"), unsafe_allow_html=True)
        st.markdown(macro_progress_html("탄수화물", st.session_state.total_carb,    target_carb,     "#7c6fff"), unsafe_allow_html=True)
        st.markdown(macro_progress_html("단백질",   st.session_state.total_protein, target_protein,  "#00d4aa"), unsafe_allow_html=True)
        st.markdown(macro_progress_html("지방",     st.session_state.total_fat,     target_fat,      "#ffb347"), unsafe_allow_html=True)

        # 오늘 식사 목록
        if st.session_state.foods:
            st.markdown("## 🍽 오늘 등록된 식사")
            type_color = {"아침":"#7c6fff","점심":"#00d4aa","저녁":"#4fc3f7","야식":"#FF4B4B"}
            for meal_type, food_name in st.session_state.foods:
                color = type_color.get(meal_type, "#8080a0")
                st.markdown(
                    f"<div style='padding:10px 16px;margin:5px 0;border-left:3px solid {color};"
                    f"background:#1a1a24;border-radius:8px;display:flex;justify-content:space-between;'>"
                    f"<span>{food_name}</span>"
                    f"<span style='color:{color};font-size:12px;font-weight:600;'>{meal_type}</span>"
                    f"</div>", unsafe_allow_html=True
                )

        # 추천
        if st.session_state.foods and info_complete:
            pr = st.session_state.total_protein / max(target_protein,1)
            cr = st.session_state.total_carb    / max(target_carb,1)
            fr = st.session_state.total_fat     / max(target_fat,1)
            candidates = []
            if pr <= cr and pr <= fr:
                lack = "단백질"
                for f,fi in food_info.items(): candidates.append((fi["protein"]/max(fi["kcal"],1), fi["name"], fi["emoji"]))
            elif cr <= fr:
                lack = "탄수화물"
                for f,fi in food_info.items(): candidates.append((fi["carb"]/max(fi["kcal"],1), fi["name"], fi["emoji"]))
            else:
                lack = "지방"
                for f,fi in food_info.items(): candidates.append((fi["fat"]/max(fi["kcal"],1), fi["name"], fi["emoji"]))
            candidates.sort(reverse=True)
            top3 = candidates[:3]
            st.markdown("## ✨ 부족 영양소 추천")
            st.markdown(f"<p style='color:#8080a0;'>현재 <b style='color:#b39ddb;'>{lack}</b>이 가장 부족합니다. 아래 음식을 추천합니다.</p>", unsafe_allow_html=True)
            rc1, rc2, rc3 = st.columns(3)
            for col, (_, name, emoji) in zip([rc1,rc2,rc3], top3):
                col.markdown(
                    f"<div style='background:#1a1a24;border:1px solid rgba(124,111,255,0.2);border-radius:12px;"
                    f"padding:16px;text-align:center;'>"
                    f"<div style='font-size:36px;'>{emoji}</div>"
                    f"<div style='font-weight:600;margin-top:8px;'>{name}</div>"
                    f"</div>", unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<div style='background:#1a1a24;border:1px solid rgba(124,111,255,0.15);border-radius:12px;"
            "padding:24px;text-align:center;color:#8080a0;margin-top:16px;'>"
            "성별·나이·키·몸무게·목표를 모두 입력하면<br>맞춤 목표가 계산됩니다.</div>",
            unsafe_allow_html=True
        )
        # 전역 변수 기본값
        target_calorie = target_carb = target_protein = target_fat = None

# ============================================================
# TAB 2 — 음식 등록
# ============================================================
with tab2:
    st.markdown("## 📸 음식 사진 업로드")
    st.markdown("<p style='color:#8080a0;'>사진을 업로드하면 AI가 15가지 음식을 자동 인식합니다.</p>", unsafe_allow_html=True)

    meal_type = st.selectbox("🕐 식사 종류", ["아침", "점심", "저녁", "야식"])
    uploaded_file = st.file_uploader("사진 선택 (JPG, PNG)", type=["jpg","jpeg","png"])

    def register_meal(info, mtype):
        st.session_state.total_kcal    += info["kcal"]
        st.session_state.total_protein += info["protein"]
        st.session_state.total_carb    += info["carb"]
        st.session_state.total_fat     += info["fat"]
        st.session_state.foods.append((mtype, info["name"]))
        if mtype == "야식":
            st.session_state.night_count += 1
        st.success(f"✅ {info['emoji']} {info['name']}이(가) {mtype} 식사로 등록되었습니다!")

    if uploaded_file is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)

        with col_result:
            if model_loaded:
                with st.spinner("🤖 AI 분석 중..."):
                    img = image.resize((224, 224))
                    if img.mode != "RGB": img = img.convert("RGB")
                    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
                    prediction = model.predict(img_array)
                    pred_idx   = np.argmax(prediction)
                    food_key   = class_names[pred_idx]
                    confidence = prediction[0][pred_idx] * 100
                    info       = food_info[food_key]

                st.markdown(
                    f"<div style='background:#1a1a24;border:1px solid rgba(124,111,255,0.25);"
                    f"border-radius:14px;padding:20px;margin-bottom:14px;'>"
                    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px;'>"
                    f"<span style='font-size:48px;'>{info['emoji']}</span>"
                    f"<div><div style='font-size:20px;font-weight:700;'>{info['name']}</div>"
                    f"<div style='font-size:12px;color:#8080a0;margin-top:3px;'>신뢰도 {confidence:.1f}%</div>"
                    f"<div style='background:#1a1a24;border-radius:4px;height:4px;margin-top:6px;overflow:hidden;width:160px;'>"
                    f"<div style='width:{confidence:.0f}%;height:100%;background:linear-gradient(90deg,#7c6fff,#ff6b9d);'></div></div>"
                    f"</div></div>"
                    f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;'>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'>"
                    f"<div style='font-size:17px;font-weight:700;color:#ff6b9d;'>{info['kcal']}</div>"
                    f"<div style='font-size:10px;color:#8080a0;'>kcal</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'>"
                    f"<div style='font-size:17px;font-weight:700;color:#7c6fff;'>{info['carb']}</div>"
                    f"<div style='font-size:10px;color:#8080a0;'>탄수화물g</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'>"
                    f"<div style='font-size:17px;font-weight:700;color:#00d4aa;'>{info['protein']}</div>"
                    f"<div style='font-size:10px;color:#8080a0;'>단백질g</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'>"
                    f"<div style='font-size:17px;font-weight:700;color:#ffb347;'>{info['fat']}</div>"
                    f"<div style='font-size:10px;color:#8080a0;'>지방g</div></div>"
                    f"</div></div>", unsafe_allow_html=True
                )
                st.button("🍽️ 이 식사 등록하기", on_click=register_meal, args=(info, meal_type))
            else:
                st.warning("모델이 로드되지 않아 예측을 수행할 수 없습니다.")

# ============================================================
# TAB 3 — 하루 평가
# ============================================================
with tab3:
    st.markdown("## 💯 하루 평가")
    st.markdown(
        "<div style='background:#1a1a24;border-radius:10px;padding:12px 16px;margin-bottom:16px;"
        "display:flex;gap:20px;font-size:13px;'>"
        "<span>🔴 <b>주의</b> 0~30점</span>"
        "<span>🟡 <b>보통</b> 31~70점</span>"
        "<span>🔵 <b>충족</b> 71~100점</span>"
        "</div>", unsafe_allow_html=True
    )

    if not info_complete:
        st.warning("먼저 '프로필 & 목표' 탭에서 정보를 입력해주세요.")
    elif not st.session_state.foods:
        st.warning("등록된 식사가 없습니다. '음식 등록' 탭에서 식사를 추가해주세요.")
    else:
        # 오늘 요약 미리보기
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("섭취 칼로리", f"{st.session_state.total_kcal} kcal")
        col_b.metric("탄수화물",    f"{st.session_state.total_carb}g")
        col_c.metric("단백질",      f"{st.session_state.total_protein}g")
        col_d.metric("지방",        f"{st.session_state.total_fat}g")

        st.markdown("")
        if st.button("💯 오늘 하루 평가하기"):
            score, details = evaluate_score(
                st.session_state.total_kcal, st.session_state.total_protein,
                st.session_state.total_carb, st.session_state.total_fat,
                target_calorie, target_protein, target_carb, target_fat
            )
            # 7일 초과 리셋
            if len(st.session_state.weekly_scores) >= 7:
                st.session_state.weekly_scores   = []
                st.session_state.weekly_calories = []
                st.session_state.weekly_details  = []
                st.info("📌 7일 완료! 새 주간 기록을 시작합니다.")

            st.session_state.weekly_scores.append(score)
            st.session_state.weekly_calories.append(st.session_state.total_kcal)
            st.session_state.weekly_details.append(details)
            day_num = len(st.session_state.weekly_scores)

            grade, icon, color = get_grade(score)
            # 종합 결과
            st.markdown(
                f"<div style='background:#1a1a24;border:2px solid {color};border-radius:16px;"
                f"padding:24px;text-align:center;margin:16px 0;'>"
                f"<div style='font-size:13px;color:#8080a0;margin-bottom:8px;'>{day_num}일차 종합 점수</div>"
                f"<div style='font-size:52px;font-weight:800;color:{color};'>{score}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{color};margin-top:4px;'>{icon} {grade}</div>"
                f"</div>", unsafe_allow_html=True
            )

            # 항목별 카드
            st.markdown("**📋 항목별 상세 결과**")
            status_color = {"충족":"#1E90FF","보통":"#FFA500","주의":"#FF4B4B"}
            status_icon  = {"충족":"🔵","보통":"🟡","주의":"🔴"}
            cols = st.columns(4)
            for idx, (item, d) in enumerate(details.items()):
                with cols[idx]:
                    c = status_color[d["status"]]
                    i = status_icon[d["status"]]
                    st.markdown(
                        f"<div style='border:2px solid {c};border-radius:12px;padding:14px;"
                        f"text-align:center;background:{c}11;'>"
                        f"<div style='font-size:12px;color:#8080a0;margin-bottom:6px;'>{item}</div>"
                        f"<div style='font-size:18px;font-weight:700;color:{c};'>{i} {d['status']}</div>"
                        f"<div style='font-size:13px;margin-top:6px;'>{d['actual']} / {d['target']} {d['unit']}</div>"
                        f"<div style='font-size:11px;color:#8080a0;margin-top:3px;'>+{d['score']}점</div>"
                        f"</div>", unsafe_allow_html=True
                    )

            st.markdown("")
            lacking = [item for item, d in details.items() if d["status"] in ("주의","보통")]
            if lacking:
                st.warning(f"⚠️ 더 충족해야 할 항목: **{'  |  '.join(lacking)}**")
            else:
                st.success("🎉 모든 항목 충족! 완벽한 하루예요!")

            # 초기화
            st.session_state.total_kcal = st.session_state.total_protein = 0
            st.session_state.total_carb = st.session_state.total_fat = 0
            st.session_state.foods = []
            st.info(f"📌 {day_num}일차 저장 완료! {day_num+1}일차를 이어서 등록하세요.")

# ============================================================
# TAB 4 — 주간 리포트
# ============================================================
with tab4:
    st.markdown("## 📅 주간 리포트")

    if not st.session_state.weekly_scores:
        st.markdown(
            "<div style='background:#1a1a24;border-radius:12px;padding:40px;text-align:center;color:#8080a0;'>"
            "아직 저장된 기록이 없습니다.<br><small>하루 평가를 완료하면 여기에 표시됩니다.</small></div>",
            unsafe_allow_html=True
        )
    else:
        scores       = st.session_state.weekly_scores
        calories     = st.session_state.weekly_calories
        details_list = st.session_state.weekly_details
        weekly_avg   = sum(scores) / len(scores)

        # 주간 요약 카드
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 기록 일수",    f"{len(scores)}일 / 7일")
        c2.metric("⭐ 주간 평균",    f"{round(weekly_avg, 1)}점")
        c3.metric("🔥 평균 칼로리",  f"{round(sum(calories)/len(calories))} kcal")

        st.markdown(grade_badge(round(weekly_avg), "주간 평균"), unsafe_allow_html=True)
        st.markdown("")

        # 주의 3일 이상 경고
        caution_days = [i+1 for i, s in enumerate(scores) if s <= 30]
        if len(caution_days) >= 3:
            st.error(f"🚨 **주의 등급 {len(caution_days)}일** ({', '.join([f'{d}일차' for d in caution_days])}) — 식단 개선이 시급합니다!")
            all_lacking = {}
            for i in [d-1 for d in caution_days]:
                if i < len(details_list):
                    for item, d in details_list[i].items():
                        if d["status"] == "주의":
                            all_lacking[item] = all_lacking.get(item, 0) + 1
            if all_lacking:
                sl = sorted(all_lacking.items(), key=lambda x: -x[1])
                st.warning("📌 **자주 부족한 항목:** " + "  |  ".join([f"{k}({v}일)" for k,v in sl]))

        # 일자별 기록
        st.markdown("**📋 일자별 기록**")
        for i, (s, c) in enumerate(zip(scores, calories), start=1):
            g, ico, col = get_grade(s)
            day_d = details_list[i-1] if i-1 < len(details_list) else {}
            lacking_items = [item for item, d in day_d.items() if d["status"] in ("주의","보통")]
            lacking_str   = f" — 부족: {', '.join(lacking_items)}" if lacking_items else ""
            st.markdown(
                f"<div style='padding:10px 16px;margin:5px 0;border-left:4px solid {col};"
                f"background:{col}11;border-radius:8px;'>"
                f"<b>{i}일차</b> &nbsp; {ico} {g} &nbsp; <b>{s}점</b> &nbsp;|&nbsp; {c} kcal"
                f"<span style='color:#8080a0;font-size:12px;'>{lacking_str}</span>"
                f"</div>", unsafe_allow_html=True
            )

        # 종합 등급
        st.markdown("")
        if weekly_avg > 70:   st.success("🥇 매우 우수한 식습관을 유지 중입니다!")
        elif weekly_avg > 30: st.warning("🥈 양호하지만 개선의 여지가 있습니다.")
        else:                  st.error("🥉 식습관 개선이 필요합니다. 영양 균형을 맞춰보세요.")

        # 그래프 (matplotlib 다크 테마)
        st.markdown("**📈 주간 칼로리 & 점수 변화**")
        days = list(range(1, len(calories)+1))
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#15151f")
        bar_colors = [get_grade(s)[2] for s in scores]

        ax1.set_facecolor("#1a1a24")
        ax1.bar(days, calories, color=bar_colors, alpha=0.85, edgecolor="none", width=0.6)
        ax1.plot(days, calories, marker="o", color="#f0f0f8", linewidth=1.5, zorder=5)
        ax1.set_xticks(days); ax1.set_xticklabels([f"{d}일" for d in days])
        ax1.set_xlabel("Day", color="#8080a0"); ax1.set_ylabel("kcal", color="#8080a0")
        ax1.set_title("일별 칼로리", color="#f0f0f8", fontweight="bold")
        ax1.tick_params(colors="#8080a0"); ax1.spines[:].set_color("#333")
        ax1.legend(handles=[
            mpatches.Patch(color="#FF4B4B",label="주의"),
            mpatches.Patch(color="#FFA500",label="보통"),
            mpatches.Patch(color="#1E90FF",label="충족")
        ], fontsize=9, facecolor="#1a1a24", edgecolor="#333", labelcolor="#f0f0f8")

        ax2.set_facecolor("#1a1a24")
        ax2.bar(days, scores, color=bar_colors, alpha=0.85, edgecolor="none", width=0.6)
        ax2.plot(days, scores, marker="o", color="#f0f0f8", linewidth=1.5, zorder=5)
        ax2.axhline(y=70, color="#1E90FF", linestyle="--", linewidth=1, alpha=0.7, label="충족(70)")
        ax2.axhline(y=30, color="#FF4B4B", linestyle="--", linewidth=1, alpha=0.7, label="주의(30)")
        ax2.set_ylim(0, 110); ax2.set_xticks(days)
        ax2.set_xticklabels([f"{d}일" for d in days])
        ax2.set_xlabel("Day", color="#8080a0"); ax2.set_ylabel("점수", color="#8080a0")
        ax2.set_title("일별 평가 점수", color="#f0f0f8", fontweight="bold")
        ax2.tick_params(colors="#8080a0"); ax2.spines[:].set_color("#333")
        ax2.legend(fontsize=9, facecolor="#1a1a24", edgecolor="#333", labelcolor="#f0f0f8")

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.style.use("default")
