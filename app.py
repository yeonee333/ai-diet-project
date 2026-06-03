import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import keras

st.set_page_config(
    page_title="NutriAI - 스마트 식단 관리",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
.stApp { background-color: #0a0a0f; color: #f0f0f8; }
[data-testid="stSidebar"] { background-color: #15151f !important; border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] * { color: #f0f0f8 !important; }
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
h1 { background: linear-gradient(135deg, #7c6fff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem !important; font-weight: 800 !important; margin-bottom: 0 !important; }
h2 { color: #f0f0f8 !important; font-size: 1.2rem !important; font-weight: 700 !important; border-left: 4px solid #7c6fff; padding-left: 12px; margin-top: 1.5rem !important; margin-bottom: 0.5rem !important; }
[data-testid="metric-container"] { background: #1a1a24 !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; padding: 14px !important; }
[data-testid="metric-container"] label { color: #8080a0 !important; font-size: 11px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f0f8 !important; font-size: 22px !important; font-weight: 700 !important; }
[data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] > div > div > input { background-color: #1a1a24 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: #f0f0f8 !important; }
.stButton > button { background: linear-gradient(135deg, #7c6fff, #ff6b9d) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 24px !important; font-weight: 600 !important; font-size: 14px !important; width: 100%; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(124,111,255,0.35) !important; }
[data-testid="stFileUploader"] { background: #1a1a24 !important; border: 2px dashed rgba(124,111,255,0.35) !important; border-radius: 12px !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(124,111,255,0.3); border-radius: 3px; }
[data-testid="stImage"] img { border-radius: 12px !important; }
[data-testid="stNumberInput"] button { background: #1a1a24 !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #8080a0 !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #1a1a24; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8080a0 !important; border-radius: 8px !important; padding: 8px 20px !important; }
.stTabs [aria-selected="true"] { background-color: #7c6fff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_food_model():
    return keras.saving.load_model("food_model.h5", compile=False)

try:
    model = load_food_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

defaults = {
    "total_kcal": 0,
    "total_protein": 0,
    "total_carb": 0,
    "total_fat": 0,
    "foods": [],
    "night_count": 0,
    "weekly_scores": [],
    "weekly_calories": [],
    "weekly_details": [],
    "target_calorie": None,
    "target_carb": None,
    "target_protein": None,
    "target_fat": None,
    "last_pred": None,
    "upload_reset_key": 0,
    "meal_type_idx": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

class_names = [
    "bibimbap","chicken_wings","donuts","dumplings","fried_rice",
    "hamburger","ice_cream","omelette","pho","pizza",
    "ramen","spaghetti","sushi","tacos","waffles"
]
food_info = {
    "bibimbap":      {"name":"비빔밥",    "emoji":"🍚","kcal":550,"carb":80,"protein":18,"fat":12},
    "chicken_wings": {"name":"치킨윙",    "emoji":"🍗","kcal":320,"carb":8, "protein":24,"fat":22},
    "donuts":        {"name":"도넛",      "emoji":"🍩","kcal":250,"carb":32,"protein":3, "fat":12},
    "dumplings":     {"name":"만두",      "emoji":"🥟","kcal":350,"carb":45,"protein":15,"fat":12},
    "fried_rice":    {"name":"볶음밥",    "emoji":"🍳","kcal":700,"carb":95,"protein":20,"fat":25},
    "hamburger":     {"name":"햄버거",    "emoji":"🍔","kcal":550,"carb":45,"protein":25,"fat":30},
    "ice_cream":     {"name":"아이스크림","emoji":"🍦","kcal":210,"carb":25,"protein":4, "fat":11},
    "omelette":      {"name":"오믈렛",    "emoji":"🍳","kcal":250,"carb":5, "protein":18,"fat":17},
    "pho":           {"name":"쌀국수",    "emoji":"🍜","kcal":450,"carb":65,"protein":20,"fat":10},
    "pizza":         {"name":"피자",      "emoji":"🍕","kcal":850,"carb":90,"protein":35,"fat":40},
    "ramen":         {"name":"라면",      "emoji":"🍜","kcal":500,"carb":75,"protein":10,"fat":18},
    "spaghetti":     {"name":"스파게티",  "emoji":"🍝","kcal":700,"carb":85,"protein":25,"fat":25},
    "sushi":         {"name":"초밥",      "emoji":"🍣","kcal":500,"carb":65,"protein":25,"fat":10},
    "tacos":         {"name":"타코",      "emoji":"🌮","kcal":300,"carb":30,"protein":12,"fat":15},
    "waffles":       {"name":"와플",      "emoji":"🧇","kcal":450,"carb":60,"protein":8, "fat":20},
}

def get_grade(score):
    if score <= 30:   return "주의", "#FF4B4B"
    elif score <= 70: return "보통", "#FFA500"
    else:             return "충족", "#1E90FF"

def grade_badge_html(score, label=""):
    g, color = get_grade(score)
    prefix = f"{label} " if label else ""
    return (
        f"<div style='display:inline-block;padding:7px 20px;border-radius:20px;"
        f"background:{color}22;border:2px solid {color};color:{color};"
        f"font-weight:700;font-size:15px;'>{prefix}{g} | {score}점</div>"
    )

def evaluate_score(tk, tp, tc, tf_, gk, gp, gc, gf):
    details = {}
    for label, actual, target in [("칼로리",tk,gk),("탄수화물",tc,gc),("단백질",tp,gp),("지방",tf_,gf)]:
        r    = actual / max(target, 1)
        unit = "kcal" if label == "칼로리" else "g"
        t1   = (0.9,1.1)   if label == "칼로리" else (0.85,1.15)
        t2   = (0.75,1.25) if label == "칼로리" else (0.6,1.4)
        if t1[0] <= r <= t1[1]:   s, st_ = 25, "충족"
        elif t2[0] <= r <= t2[1]: s, st_ = 15, "보통"
        else:                      s, st_ = 5,  "주의"
        details[label] = {"score":s,"status":st_,"actual":actual,"target":round(target),"unit":unit}
    return sum(d["score"] for d in details.values()), details

def macro_bar(label, actual, goal, color):
    pct  = min(100, round(actual/max(goal,1)*100)) if goal else 0
    unit = "kcal" if label == "칼로리" else "g"
    return (
        f"<div style='margin-bottom:14px;'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:5px;'>"
        f"<span style='font-size:13px;color:#8080a0;'>{label}</span>"
        f"<span style='font-size:13px;color:{color};font-weight:600;'>"
        f"{actual} / {round(goal) if goal else '--'}{unit}</span></div>"
        f"<div style='background:#1a1a24;border-radius:6px;height:8px;overflow:hidden;'>"
        f"<div style='width:{pct}%;height:100%;background:{color};border-radius:6px;'></div>"
        f"</div></div>"
    )

# ===== 사이드바 =====
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px;'>
      <div style='font-size:22px;font-weight:800;background:linear-gradient(135deg,#7c6fff,#ff6b9d);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>NutriAI</div>
      <div style='font-size:11px;color:#8080a0;letter-spacing:1px;margin-top:3px;'>스마트 식단 관리</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**오늘 현황**")
    tc = st.session_state.target_calorie
    st.metric("섭취 칼로리", f"{st.session_state.total_kcal} kcal",
              delta=f"남은 {round(tc - st.session_state.total_kcal)}kcal" if tc else None)
    st.metric("등록 식사", f"{len(st.session_state.foods)}끼")
    if st.session_state.night_count >= 3:
        st.error(f"야식 {st.session_state.night_count}회! 야식을 줄여주세요.")
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("데이터 초기화", key="reset_btn"):
        st.session_state.clear()
        st.rerun()

# ===== 타이틀 =====
st.markdown("<h1>NutriAI 식단 관리 서비스</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8080a0;margin-bottom:24px;'>AI 기반 음식 인식 · 맞춤 영양 목표 · 하루 평가 · 주간 리포트</p>", unsafe_allow_html=True)

target_calorie = st.session_state.target_calorie
target_carb    = st.session_state.target_carb
target_protein = st.session_state.target_protein
target_fat     = st.session_state.target_fat
info_complete  = all(v is not None for v in [target_calorie, target_carb, target_protein, target_fat])

# ===== 탭 =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "프로필 & 목표", "음식 등록", "오늘 현황", "영양소 추천", "하루 평가", "주간 리포트"
])

# ===== TAB1: 프로필 & 목표 =====
with tab1:
    st.markdown("## 프로필 설정")
    st.markdown("<p style='color:#8080a0;'>정보를 입력하면 BMR 기반으로 맞춤 목표가 자동 계산됩니다.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("성별", ["남","여"], index=None, placeholder="선택하세요")
        age    = st.number_input("나이 (세)", min_value=1, max_value=100, value=None, placeholder="입력하세요")
    with col2:
        height = st.number_input("키 (cm)",     min_value=100, max_value=250, value=None, placeholder="입력하세요")
        weight = st.number_input("몸무게 (kg)", min_value=30,  max_value=200, value=None, placeholder="입력하세요")

    goal = st.selectbox("목표", ["다이어트 (체중 감량)","유지 (체중 유지)","벌크업 (근육 증가)"], index=None, placeholder="선택하세요")

    if st.button("목표 계산하기", key="calc_btn"):
        if None in (gender, age, height, weight, goal):
            st.warning("모든 항목을 입력해주세요.")
        else:
            bmr = (10*weight + 6.25*height - 5*age + 5) if gender == "남" \
                  else (10*weight + 6.25*height - 5*age - 161)
            maintain = bmr * 1.55
            if "다이어트" in goal:   tc = maintain - 500
            elif "벌크업" in goal:   tc = maintain + 500
            else:                     tc = maintain
            tp    = weight * 1.5
            tf_   = weight * 0.8
            tcarb = max((tc - tp*4 - tf_*9) / 4, 0)
            st.session_state.target_calorie = tc
            st.session_state.target_carb    = tcarb
            st.session_state.target_protein = tp
            st.session_state.target_fat     = tf_
            target_calorie = tc
            target_carb    = tcarb
            target_protein = tp
            target_fat     = tf_
            info_complete  = True
            st.success("목표가 설정되었습니다!")

    st.markdown("## 오늘 목표")
    if info_complete:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("칼로리",   f"{round(target_calorie)} kcal")
        c2.metric("탄수화물", f"{round(target_carb)}g")
        c3.metric("단백질",   f"{round(target_protein)}g")
        c4.metric("지방",     f"{round(target_fat)}g")
    else:
        st.code("칼로리: - kcal  |  탄수화물: - g  |  단백질: - g  |  지방: - g")
        st.caption("성별, 나이, 키, 몸무게, 목표를 모두 입력하면 목표치가 계산됩니다.")

# ===== TAB2: 음식 등록 =====
with tab2:
    st.markdown("## 음식 등록")
    st.markdown("<p style='color:#8080a0;'>사진을 업로드하면 AI가 15가지 음식을 자동 인식합니다.</p>", unsafe_allow_html=True)

    meal_options = ["아침","점심","저녁","야식"]
    meal_colors  = {"아침":"#7c6fff","점심":"#00d4aa","저녁":"#4fc3f7","야식":"#FF4B4B"}

    st.markdown("**식사 종류**")
    m1, m2, m3, m4 = st.columns(4)
    for col, opt in zip([m1,m2,m3,m4], meal_options):
        selected = (st.session_state.meal_type_idx == meal_options.index(opt))
        c = meal_colors[opt]
        bg = f"{c}33" if selected else "#1a1a24"
        border = c if selected else "rgba(255,255,255,0.1)"
        fw = "700" if selected else "400"
        col.markdown(
            f"<div style='border:2px solid {border};background:{bg};border-radius:10px;"
            f"padding:10px;text-align:center;color:{c if selected else '#8080a0'};"
            f"font-weight:{fw};font-size:14px;margin-bottom:4px;'>{opt}</div>",
            unsafe_allow_html=True
        )
        if col.button(opt, key=f"meal_btn_{opt}"):
            st.session_state.meal_type_idx = meal_options.index(opt)
            st.rerun()

    meal_type = meal_options[st.session_state.meal_type_idx]
    uploaded_file = st.file_uploader(
        "사진 선택 (JPG, PNG)",
        type=["jpg","jpeg","png"],
        key=f"uploader_{st.session_state.upload_reset_key}"
    )

    if uploaded_file is not None:
        col_img, col_res = st.columns([1,1])
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)
        with col_res:
            if model_loaded:
                if (st.session_state.last_pred is None or
                        st.session_state.last_pred.get("filename") != uploaded_file.name):
                    with st.spinner("AI 분석 중..."):
                        img = image.resize((224,224))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        arr  = np.expand_dims(np.array(img)/255.0, axis=0)
                        pred = model.predict(arr)
                        idx  = np.argmax(pred)
                        key  = class_names[idx]
                        conf = float(pred[0][idx] * 100)
                        info = food_info[key]
                        st.session_state.last_pred = {
                            "filename": uploaded_file.name,
                            "info": info,
                            "conf": conf
                        }
                else:
                    info = st.session_state.last_pred["info"]
                    conf = st.session_state.last_pred["conf"]

                st.markdown(
                    f"<div style='background:#1a1a24;border:1px solid rgba(124,111,255,0.25);"
                    f"border-radius:14px;padding:20px;margin-bottom:14px;'>"
                    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>"
                    f"<span style='font-size:48px;'>{info['emoji']}</span>"
                    f"<div><div style='font-size:20px;font-weight:700;'>{info['name']}</div>"
                    f"<div style='font-size:12px;color:#8080a0;margin-top:3px;'>신뢰도 {conf:.1f}%</div>"
                    f"<div style='background:#0a0a0f;border-radius:4px;height:4px;margin-top:6px;width:160px;overflow:hidden;'>"
                    f"<div style='width:{conf:.0f}%;height:100%;background:linear-gradient(90deg,#7c6fff,#ff6b9d);'></div></div>"
                    f"</div></div>"
                    f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;'>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'><div style='font-size:17px;font-weight:700;color:#ff6b9d;'>{info['kcal']}</div><div style='font-size:10px;color:#8080a0;'>kcal</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'><div style='font-size:17px;font-weight:700;color:#7c6fff;'>{info['carb']}</div><div style='font-size:10px;color:#8080a0;'>탄수화물g</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'><div style='font-size:17px;font-weight:700;color:#00d4aa;'>{info['protein']}</div><div style='font-size:10px;color:#8080a0;'>단백질g</div></div>"
                    f"<div style='text-align:center;background:#0a0a0f;border-radius:8px;padding:10px;'><div style='font-size:17px;font-weight:700;color:#ffb347;'>{info['fat']}</div><div style='font-size:10px;color:#8080a0;'>지방g</div></div>"
                    f"</div></div>", unsafe_allow_html=True
                )
                if st.button("이 식사 등록하기", key="register_btn"):
                    st.session_state.total_kcal    += info["kcal"]
                    st.session_state.total_protein += info["protein"]
                    st.session_state.total_carb    += info["carb"]
                    st.session_state.total_fat     += info["fat"]
                    st.session_state.foods.append((meal_type, info["name"]))
                    if meal_type == "야식":
                        st.session_state.night_count += 1
                    st.session_state.upload_reset_key += 1
                    st.session_state.meal_type_idx = 0
                    st.session_state.last_pred = None
                    st.success(f"{info['emoji']} {info['name']}이(가) {meal_type} 식사로 등록되었습니다!")
                    st.rerun()
            else:
                st.error(f"모델 로딩 실패: {model_error}")
    else:
        st.markdown(
            "<div style='background:#1a1a24;border-radius:12px;padding:32px;text-align:center;color:#8080a0;'>"
            "사진을 업로드하면 AI가 자동으로 음식을 인식합니다.</div>",
            unsafe_allow_html=True
        )

# ===== TAB3: 오늘 현황 =====
with tab3:
    st.markdown("## 오늘의 식단 현황")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("섭취 칼로리", f"{st.session_state.total_kcal} kcal",
              delta=f"목표 {round(target_calorie)}kcal" if info_complete else "목표 미설정")
    c2.metric("탄수화물", f"{st.session_state.total_carb}g")
    c3.metric("단백질",   f"{st.session_state.total_protein}g")
    c4.metric("지방",     f"{st.session_state.total_fat}g")

    st.markdown("## 영양소 섭취 현황")
    st.markdown(macro_bar("칼로리",   st.session_state.total_kcal,    target_calorie,  "#ff6b9d"), unsafe_allow_html=True)
    st.markdown(macro_bar("탄수화물", st.session_state.total_carb,    target_carb,     "#7c6fff"), unsafe_allow_html=True)
    st.markdown(macro_bar("단백질",   st.session_state.total_protein, target_protein,  "#00d4aa"), unsafe_allow_html=True)
    st.markdown(macro_bar("지방",     st.session_state.total_fat,     target_fat,      "#ffb347"), unsafe_allow_html=True)

    if not info_complete:
        st.info("프로필 & 목표 탭에서 정보를 먼저 설정해주세요!")

    if st.session_state.foods:
        st.markdown("## 오늘 등록된 식사")
        type_color = {"아침":"#7c6fff","점심":"#00d4aa","저녁":"#4fc3f7","야식":"#FF4B4B"}
        for mt, fn in st.session_state.foods:
            color = type_color.get(mt, "#8080a0")
            st.markdown(
                f"<div style='padding:10px 16px;margin:5px 0;border-left:3px solid {color};"
                f"background:#1a1a24;border-radius:8px;display:flex;justify-content:space-between;'>"
                f"<span style='font-size:14px;'>{fn}</span>"
                f"<span style='color:{color};font-size:12px;font-weight:600;'>{mt}</span>"
                f"</div>", unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style='background:#1a1a24;border-radius:12px;padding:32px;text-align:center;color:#8080a0;margin-top:16px;'>"
            "아직 등록된 식사가 없습니다.<br><small>음식 등록 탭에서 추가하세요.</small></div>",
            unsafe_allow_html=True
        )

# ===== TAB4: 영양소 추천 =====
with tab4:
    st.markdown("## 영양소 추천")
    st.markdown("<p style='color:#8080a0;'>현재 식단에서 가장 부족한 영양소를 채울 음식 3가지를 추천합니다.</p>", unsafe_allow_html=True)

    if not st.session_state.foods:
        st.info("음식을 먼저 등록해주세요.")
    elif not info_complete:
        st.info("프로필 & 목표 탭에서 먼저 설정해주세요.")
    else:
        pr = st.session_state.total_protein / max(target_protein,1)
        cr = st.session_state.total_carb    / max(target_carb,1)
        fr = st.session_state.total_fat     / max(target_fat,1)

        st.markdown("## 현재 영양소 현황")
        st.markdown(macro_bar("탄수화물", st.session_state.total_carb,    target_carb,    "#7c6fff"), unsafe_allow_html=True)
        st.markdown(macro_bar("단백질",   st.session_state.total_protein, target_protein, "#00d4aa"), unsafe_allow_html=True)
        st.markdown(macro_bar("지방",     st.session_state.total_fat,     target_fat,     "#ffb347"), unsafe_allow_html=True)

        kcal_ratio = st.session_state.total_kcal / max(target_calorie, 1)
        if kcal_ratio > 1.1:
            st.warning(f"칼로리를 {round(st.session_state.total_kcal - target_calorie)}kcal 초과했습니다! 오늘은 가벼운 식사를 권장합니다.")

        candidates = []
        if pr <= cr and pr <= fr:
            lack = "단백질"
            for f, fi in food_info.items():
                candidates.append((fi["protein"]/max(fi["kcal"],1), fi))
        elif cr <= fr:
            lack = "탄수화물"
            for f, fi in food_info.items():
                candidates.append((fi["carb"]/max(fi["kcal"],1), fi))
        else:
            lack = "지방"
            for f, fi in food_info.items():
                candidates.append((fi["fat"]/max(fi["kcal"],1), fi))
        candidates.sort(key=lambda x: -x[0])
        top3 = [fi for _, fi in candidates[:3]]

        st.markdown("## 추천 식품")
        st.markdown(f"<p style='color:#8080a0;'>현재 <b style='color:#b39ddb;'>{lack}</b>이 가장 부족합니다.</p>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        for col, fi in zip([rc1,rc2,rc3], top3):
            col.markdown(
                f"<div style='background:#1a1a24;border:1px solid rgba(124,111,255,0.2);border-radius:14px;"
                f"padding:20px;text-align:center;'>"
                f"<div style='font-size:40px;margin-bottom:10px;'>{fi['emoji']}</div>"
                f"<div style='font-weight:700;font-size:16px;margin-bottom:8px;'>{fi['name']}</div>"
                f"<div style='font-size:12px;color:#ff6b9d;font-weight:600;'>{fi['kcal']} kcal</div>"
                f"<div style='font-size:11px;color:#8080a0;margin-top:6px;'>"
                f"탄 {fi['carb']}g / 단 {fi['protein']}g / 지 {fi['fat']}g</div>"
                f"</div>", unsafe_allow_html=True
            )

# ===== TAB5: 하루 평가 =====
with tab5:
    st.markdown("## 하루 평가")
    st.markdown(
        "<div style='background:#1a1a24;border-radius:10px;padding:12px 16px;margin-bottom:20px;"
        "display:flex;gap:24px;font-size:13px;'>"
        "<span><b style='color:#FF4B4B;'>빨강</b> 주의 (0~30점)</span>"
        "<span><b style='color:#FFA500;'>노랑</b> 보통 (31~70점)</span>"
        "<span><b style='color:#1E90FF;'>파랑</b> 충족 (71~100점)</span>"
        "</div>", unsafe_allow_html=True
    )

    if not info_complete:
        st.warning("먼저 프로필 & 목표 탭에서 정보를 입력해주세요.")
    elif not st.session_state.foods:
        st.warning("등록된 식사가 없습니다. 음식 등록 탭에서 추가해주세요.")
    else:
        is_over = st.session_state.total_kcal > target_calorie
        diff    = abs(round(st.session_state.total_kcal - target_calorie))
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("섭취 칼로리", f"{st.session_state.total_kcal} kcal",
                  delta=f"{'초과' if is_over else '부족'} {diff}kcal",
                  delta_color="inverse" if is_over else "normal")
        c2.metric("탄수화물", f"{st.session_state.total_carb}g")
        c3.metric("단백질",   f"{st.session_state.total_protein}g")
        c4.metric("지방",     f"{st.session_state.total_fat}g")
        st.markdown("")

        if st.button("오늘 하루 평가하기", key="eval_btn"):
            score, details = evaluate_score(
                st.session_state.total_kcal, st.session_state.total_protein,
                st.session_state.total_carb, st.session_state.total_fat,
                target_calorie, target_protein, target_carb, target_fat
            )
            if len(st.session_state.weekly_scores) >= 7:
                st.session_state.weekly_scores   = []
                st.session_state.weekly_calories = []
                st.session_state.weekly_details  = []
                st.info("7일 완료! 새 주간 기록을 시작합니다.")

            st.session_state.weekly_scores.append(score)
            st.session_state.weekly_calories.append(st.session_state.total_kcal)
            st.session_state.weekly_details.append(details)
            day_num = len(st.session_state.weekly_scores)
            g, color = get_grade(score)

            st.markdown(
                f"<div style='background:#1a1a24;border:2px solid {color};border-radius:16px;"
                f"padding:28px;text-align:center;margin:16px 0;'>"
                f"<div style='font-size:13px;color:#8080a0;margin-bottom:8px;'>{day_num}일차 종합 점수</div>"
                f"<div style='font-size:56px;font-weight:800;color:{color};line-height:1;'>{score}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{color};margin-top:8px;'>{g}</div>"
                f"</div>", unsafe_allow_html=True
            )

            st.markdown("**항목별 상세 결과**")
            sc = {"충족":"#1E90FF","보통":"#FFA500","주의":"#FF4B4B"}
            cols = st.columns(4)
            for idx, (item, d) in enumerate(details.items()):
                with cols[idx]:
                    c_ = sc[d["status"]]
                    st.markdown(
                        f"<div style='border:2px solid {c_};border-radius:12px;padding:14px;"
                        f"text-align:center;background:{c_}11;'>"
                        f"<div style='font-size:12px;color:#8080a0;margin-bottom:6px;'>{item}</div>"
                        f"<div style='font-size:18px;font-weight:700;color:{c_};'>{d['status']}</div>"
                        f"<div style='font-size:13px;margin-top:6px;'>{d['actual']} / {d['target']} {d['unit']}</div>"
                        f"<div style='font-size:11px;color:#8080a0;margin-top:3px;'>+{d['score']}점</div>"
                        f"</div>", unsafe_allow_html=True
                    )

            st.markdown("")
            lacking = [item for item, d in details.items() if d["status"] in ("주의","보통")]
            if lacking:
                st.warning(f"더 충족해야 할 항목: {' | '.join(lacking)}")
            else:
                st.success("모든 항목 충족! 완벽한 하루예요!")

            st.session_state.total_kcal    = 0
            st.session_state.total_protein = 0
            st.session_state.total_carb    = 0
            st.session_state.total_fat     = 0
            st.session_state.foods         = []
            st.info(f"{day_num}일차 저장 완료! {day_num+1}일차를 이어서 등록하세요.")

# ===== TAB6: 주간 리포트 =====
with tab6:
    st.markdown("## 주간 리포트")

    if not st.session_state.weekly_scores:
        st.markdown(
            "<div style='background:#1a1a24;border-radius:12px;padding:48px;text-align:center;color:#8080a0;'>"
            "아직 저장된 기록이 없습니다.<br><small>하루 평가를 완료하면 여기에 표시됩니다.</small></div>",
            unsafe_allow_html=True
        )
    else:
        scores = st.session_state.weekly_scores
        cals   = st.session_state.weekly_calories
        dlist  = st.session_state.weekly_details
        avg    = sum(scores) / len(scores)

        c1,c2,c3 = st.columns(3)
        c1.metric("기록 일수",   f"{len(scores)}일 / 7일")
        c2.metric("주간 평균",   f"{round(avg,1)}점")
        c3.metric("평균 칼로리", f"{round(sum(cals)/len(cals))} kcal")

        st.markdown(grade_badge_html(round(avg), "주간 평균"), unsafe_allow_html=True)
        st.markdown("")

        caution_days = [i+1 for i, s in enumerate(scores) if s <= 30]
        if len(caution_days) >= 3:
            st.error(f"주의 등급 {len(caution_days)}일 ({', '.join([f'{d}일차' for d in caution_days])}) - 식단 개선이 시급합니다!")
            all_lack = {}
            for i in [d-1 for d in caution_days]:
                if i < len(dlist):
                    for item, d in dlist[i].items():
                        if d["status"] == "주의":
                            all_lack[item] = all_lack.get(item, 0) + 1
            if all_lack:
                sl = sorted(all_lack.items(), key=lambda x: -x[1])
                st.warning("자주 부족한 항목: " + " | ".join([f"{k}({v}일)" for k, v in sl]))

        st.markdown("**일자별 기록**")
        for i, (s, c) in enumerate(zip(scores, cals), start=1):
            g, col = get_grade(s)
            d_ = dlist[i-1] if i-1 < len(dlist) else {}
            lack_items = [item for item, d in d_.items() if d["status"] in ("주의","보통")]
            lack_str   = " - 부족: " + ", ".join(lack_items) if lack_items else ""
            st.markdown(
                f"<div style='padding:10px 16px;margin:5px 0;border-left:4px solid {col};"
                f"background:{col}11;border-radius:8px;'>"
                f"<b>{i}일차</b> &nbsp; {g} &nbsp; <b>{s}점</b> &nbsp;|&nbsp; {c} kcal"
                f"<span style='color:#8080a0;font-size:12px;'>{lack_str}</span>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("")
        if avg > 70:   st.success("매우 우수한 식습관을 유지 중입니다!")
        elif avg > 30: st.warning("양호하지만 개선의 여지가 있습니다.")
        else:           st.error("식습관 개선이 필요합니다.")

        st.markdown("**주간 칼로리 & 점수 변화**")
        days = list(range(1, len(cals)+1))
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), facecolor="#15151f")
        bc = [get_grade(s)[1] for s in scores]

        for ax, data, title, ylabel in [
            (ax1, cals,   "일별 칼로리", "kcal"),
            (ax2, scores, "일별 평가 점수", "점수")
        ]:
            ax.set_facecolor("#1a1a24")
            ax.bar(days, data, color=bc, alpha=0.85, edgecolor="none", width=0.6)
            ax.plot(days, data, marker="o", color="#f0f0f8", linewidth=1.5, zorder=5)
            ax.set_xticks(days)
            ax.set_xticklabels([f"{d}일" for d in days])
            ax.set_xlabel("Day", color="#8080a0")
            ax.set_ylabel(ylabel, color="#8080a0")
            ax.set_title(title, color="#f0f0f8", fontweight="bold")
            ax.tick_params(colors="#8080a0")
            ax.spines[:].set_color("#333")

        ax2.axhline(y=70, color="#1E90FF", linestyle="--", linewidth=1, alpha=0.7, label="충족(70)")
        ax2.axhline(y=30, color="#FF4B4B", linestyle="--", linewidth=1, alpha=0.7, label="주의(30)")
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=9, facecolor="#1a1a24", edgecolor="#333", labelcolor="#f0f0f8")
        ax1.legend(handles=[
            mpatches.Patch(color="#FF4B4B", label="주의"),
            mpatches.Patch(color="#FFA500", label="보통"),
            mpatches.Patch(color="#1E90FF", label="충족"),
        ], fontsize=9, facecolor="#1a1a24", edgecolor="#333", labelcolor="#f0f0f8")

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.style.use("default")
