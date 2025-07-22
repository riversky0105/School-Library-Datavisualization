import streamlit as st
st.set_page_config(page_title="학교 & 공공 도서관 통합 분석", layout="wide")

st.title("📚 학교 도서관 & 공공 도서관 통합 분석 및 예측")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import urllib.request

# ---------------------------
# ✅ 한글 폰트 설정 (자동 다운로드)
# ---------------------------
font_dir = "fonts"
os.makedirs(font_dir, exist_ok=True)
font_path = os.path.join(font_dir, "NanumGothicCoding.ttf")

if not os.path.exists(font_path):
    url = "https://github.com/naver/nanumfont/releases/download/VER2.0/NanumGothicCoding.ttf"
    urllib.request.urlretrieve(url, font_path)

font_prop = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = font_prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False

# ---------------------------
# ✅ 학교 도서관 데이터 로드
# ---------------------------
@st.cache_data
def load_school_data():
    df = pd.read_csv("학교도서관현황_20250717223352.csv", encoding="cp949")
    df = df[df["학교급별(1)"].isin(["초등학교", "중학교", "고등학교"])]

    rows = []
    for year in range(2011, 2024):
        base_cols = [f"{year}.1", f"{year}.2", f"{year}.3"]
        existing_cols = [col for col in base_cols if col in df.columns]
        budget_col = f"{year}.4" if f"{year}.4" in df.columns else None

        cols_to_use = ["학교급별(1)"] + existing_cols
        if budget_col:
            cols_to_use.append(budget_col)

        temp_df = df[cols_to_use].assign(연도=year)
        rename_dict = {
            "학교급별(1)": "학교급",
            f"{year}.1": "장서수",
            f"{year}.2": "사서수",
            f"{year}.3": "방문자수",
        }
        if budget_col:
            rename_dict[budget_col] = "예산"

        temp_df = temp_df.rename(columns=rename_dict)
        if "예산" not in temp_df.columns:
            temp_df["예산"] = np.nan

        rows.append(temp_df)

    df_all = pd.concat(rows, ignore_index=True)
    for col in ["장서수", "사서수", "방문자수", "예산"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all["구분"] = "학교 도서관"
    return df_all

# ---------------------------
# ✅ 공공 도서관 데이터 로드
# ---------------------------
@st.cache_data
def load_public_data():
    df = pd.read_csv("공공도서관 자치구별 통계 파일.csv", encoding="cp949", header=1)
    df = df[df["자치구별(2)"] != "소계"]

    df = df.rename(columns={
        "자치구별(2)": "자치구",
        "소계": "개소수",
        "소계.1": "좌석수",
        "도서": "장서수",
        "소계.2": "방문자수",
        "소계.4": "사서수",
        "소계.5": "예산"
    })

    df = df[["자치구", "장서수", "사서수", "방문자수", "예산"]]
    for col in ["장서수", "사서수", "방문자수", "예산"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["구분"] = "공공 도서관"
    return df

# ---------------------------
# ✅ 데이터 선택
# ---------------------------
df_school = load_school_data()
df_public = load_public_data()

option = st.radio("분석할 데이터셋을 선택하세요:", ["학교 도서관", "공공 도서관", "통합 비교"])

if option == "학교 도서관":
    df = df_school.copy()
elif option == "공공 도서관":
    df = df_public.copy()
else:
    df = pd.concat([df_school, df_public], ignore_index=True)

# ---------------------------
# ✅ 상관계수 히트맵
# ---------------------------
st.subheader("📊 변수 간 상관관계 (히트맵)")
st.markdown(f"선택한 데이터셋({option})의 변수 간 상관관계를 보여줍니다.")

corr = df[["장서수", "사서수", "예산", "방문자수"]].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True)
ax.set_title(f"{option} 변수 간 상관관계", fontproperties=font_prop)
st.pyplot(fig)

# ---------------------------
# ✅ 머신러닝 (RandomForest + Linear Regression)
# ---------------------------
st.subheader("🔍 방문자 수 예측 및 변수 중요도")
st.markdown(f"{option} 데이터에서 장서수, 사서수, 예산이 방문자 수에 미치는 영향을 분석했습니다.")

X = df[["장서수", "사서수", "예산"]].fillna(df[["장서수", "사서수", "예산"]].median())
y = df["방문자수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.markdown(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

importance = pd.Series(rf_model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots(figsize=(6, 4))
importance.sort_values().plot.barh(ax=ax2, color="skyblue")
ax2.set_title(f"{option} RandomForest 변수 중요도", fontproperties=font_prop)
ax2.set_xlabel("중요도", fontproperties=font_prop)
ax2.set_ylabel("변수", fontproperties=font_prop)
ax2.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
st.pyplot(fig2)

# Linear Regression 비교 (교차 검증)
lr_model = LinearRegression()
rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring="r2")

st.subheader("📌 모델 성능 비교 (5-Fold 교차 검증)")
st.markdown(f"""
✅ **RandomForest 평균 R²**: `{rf_scores.mean():.4f}`  
✅ **Linear Regression 평균 R²**: `{lr_scores.mean():.4f}`
""")

# ---------------------------
# ✅ 데이터 테이블 출력
# ---------------------------
st.subheader("📄 사용된 데이터")
st.dataframe(df)
