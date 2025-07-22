import streamlit as st
st.set_page_config(page_title="학교 도서관 분석 및 예측", layout="wide")

st.title("📚 학교급별 도서관 이용자 수 분석 및 예측")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# ✅ 한글 폰트 설정
# ---------------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams["font.family"] = font_prop.get_name()
    mpl.rcParams["axes.unicode_minus"] = False
else:
    font_prop = None

# ---------------------------
# ✅ 데이터 로드 및 전처리 (전체 연도)
# ---------------------------
@st.cache_data
def load_all_data():
    df = pd.read_csv("학교도서관현황_20250717223352.csv", encoding="cp949")
    df = df[df["학교급별(1)"].isin(["초등학교", "중학교", "고등학교"])]

    rows = []
    for year in range(2011, 2024):  # 2011 ~ 2023
        rows.append(df[["학교급별(1)", f"{year}.1", f"{year}.2", f"{year}.3"]]
                    .assign(연도=year)
                    .rename(columns={
                        "학교급별(1)": "학교급",
                        f"{year}.1": "장서수",
                        f"{year}.2": "사서수",
                        f"{year}.3": "방문자수"
                    }))
    df_all = pd.concat(rows, ignore_index=True)

    for col in ["장서수", "사서수", "방문자수"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    return df_all

df_all = load_all_data()

# ---------------------------
# ✅ 학교급별 연도별 방문자 수 변화
# ---------------------------
st.subheader("📊 학교급별 연도별 방문자 수 변화")
st.markdown("2011년부터 2023년까지 학교급별 1관당 방문자 수 변화를 보여줍니다.")

fig, ax = plt.subplots(figsize=(12, 6))
for school in df_all["학교급"].unique():
    data = df_all[df_all["학교급"] == school]
    ax.plot(data["연도"], data["방문자수"], marker="o", label=school)

ax.set_title("학교급별 연도별 방문자 수 변화 (2011~2023)", fontproperties=font_prop)
ax.set_xlabel("연도", fontproperties=font_prop)
ax.set_ylabel("1관당 방문자 수", fontproperties=font_prop)
ax.legend(prop=font_prop)
yticks = ax.get_yticks()
ax.set_yticklabels([f"{int(t):,}" for t in yticks], fontproperties=font_prop)
st.pyplot(fig)

latest = df_all[df_all["연도"] == 2023].sort_values(by="방문자수", ascending=False).iloc[0]
st.success(f"✅ 2023년 기준 **{latest['학교급']}**이(가) 가장 많은 방문자수를 기록했습니다. (약 **{int(latest['방문자수']):,}명**)")

# ---------------------------
# ✅ 머신러닝 (RandomForest)
# ---------------------------
st.subheader("🔍 전체 연도 기반 방문자 수 예측 및 변수 중요도")
st.markdown("장서수와 사서수가 방문자 수에 어떤 영향을 미치는지 분석했습니다.")

X = df_all[["장서수", "사서수"]]
y = df_all["방문자수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.markdown(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

importance = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots(figsize=(6, 4))
importance.sort_values().plot.barh(ax=ax2, color="skyblue")
ax2.set_title("RandomForest 변수 중요도", fontproperties=font_prop)
ax2.set_xlabel("중요도", fontproperties=font_prop)
ax2.set_ylabel("변수", fontproperties=font_prop)
ax2.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
st.pyplot(fig2)

# ---------------------------
# ✅ 통계 데이터 출력
# ---------------------------
st.subheader("📄 전체 연도 통계 데이터")
st.markdown("2011년부터 2023년까지 학교급별 도서관 운영 데이터를 확인할 수 있습니다.")
st.dataframe(df_all)
