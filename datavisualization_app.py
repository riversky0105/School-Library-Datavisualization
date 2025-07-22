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

# -------------------------------
# ✅ 한글 폰트 설정
# -------------------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# -------------------------------
# ✅ 데이터 불러오기 & 전처리
# -------------------------------
@st.cache_data
def load_school_lib_data():
    df = pd.read_csv("학교도서관현황_20250717223352.csv", encoding="cp949")
    df = df[df["학교급별(1)"].isin(["초등학교", "중학교", "고등학교"])].reset_index(drop=True)

    # 연도별 필요한 컬럼만 추출 (예: 2023년 기준)
    cols = ["학교급별(1)", "2023.1", "2023.2", "2023.3"]
    df = df[cols]
    df.columns = ["학교급", "장서수", "사서수", "방문자수"]

    # 숫자로 변환
    for col in ["장서수", "사서수", "방문자수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_stat = load_school_lib_data()

# -------------------------------
# ✅ 학교급별 방문자 수 시각화
# -------------------------------
st.subheader("📊 학교급별 도서관 방문자 수")
st.markdown("2023년 기준 학교급별 1관당 방문자 수입니다.")

df_sorted = df_stat.sort_values(by="방문자수", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_sorted["학교급"], df_sorted["방문자수"], color="skyblue")
ax.set_title("학교급별 도서관 방문자 수 (2023)", fontproperties=font_prop)
ax.set_xlabel("학교급", fontproperties=font_prop)
ax.set_ylabel("1관당 방문자 수", fontproperties=font_prop)
yticks = ax.get_yticks()
ax.set_yticklabels([f"{int(y):,}" for y in yticks], fontproperties=font_prop)
st.pyplot(fig)

top = df_sorted.iloc[0]
st.success(f"✅ 2023년 기준 **{top['학교급']}**이(가) 가장 많은 방문자수를 기록했습니다. (약 **{int(top['방문자수']):,}명**)")

# -------------------------------
# ✅ 머신러닝 예측 (RandomForest)
# -------------------------------
st.subheader("🔍 방문자 수 예측 및 변수 중요도")
st.markdown("장서수와 사서수가 방문자 수에 미치는 영향을 분석했습니다.")

X = df_stat[["장서수", "사서수"]]
y = df_stat["방문자수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# -------------------------------
# ✅ 원본 데이터 표시
# -------------------------------
st.subheader("📄 분석에 사용된 데이터")
st.dataframe(df_stat)
