import streamlit as st
st.set_page_config(page_title="도서관 통합 분석", layout="wide")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import folium
import requests
import json
from shapely.geometry import shape
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import urllib.request

# ---------------------------
# ✅ 한글 폰트 설정
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
# ✅ 데이터 로드 함수
# ---------------------------
@st.cache_data
def load_public_data():
    df = pd.read_csv("공공도서관 자치구별 통계 파일.csv", encoding="cp949", header=1)
    df = df[df.iloc[:, 0] != "소계"]
    df.columns = [
        '자치구명','개소수','좌석수','자료수_도서','자료수_비도서','자료수_연속간행물',
        '도서관 방문자수','연간대출책수','직원수','직원수_남','직원수_여','예산'
    ]
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

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
    return df_all

# ---------------------------
# ✅ 모드 선택
# ---------------------------
mode = st.sidebar.radio("분석 모드를 선택하세요", ["서울시 공공도서관 분석", "학교·공공 도서관 통합 분석"])

# ---------------------------
# ✅ ① 서울시 공공도서관 분석
# ---------------------------
if mode == "서울시 공공도서관 분석":
    st.title("📚 서울특별시 자치구별 도서관 이용자 수 분석 및 예측")
    df_stat = load_public_data()

    df_users = df_stat[['자치구명','도서관 방문자수']].copy()
    df_users.columns = ['구','이용자수']
    df_users['이용자수'] = df_users['이용자수'].astype(int)
    df_users_sorted = df_users.sort_values(by='이용자수', ascending=False).reset_index(drop=True)

    # 📊 그래프 시각화
    st.subheader("📊 자치구별 도서관 이용자 수 그래프 시각화")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_users_sorted['구'], df_users_sorted['이용자수'], color='skyblue')
    ax.set_title("📌 자치구별 이용자 수", fontproperties=font_prop)
    ax.set_xlabel("자치구", fontproperties=font_prop)
    ax.set_ylabel("이용자 수", fontproperties=font_prop)
    ax.set_xticks(range(len(df_users_sorted)))
    ax.set_xticklabels(df_users_sorted['구'], rotation=45, fontproperties=font_prop)
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(t):,}" for t in yticks], fontproperties=font_prop)
    st.pyplot(fig)

    # 🗺 지도 시각화
    st.subheader("🗺️ 자치구별 도서관 이용자 수 지도 시각화")
    geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
    res = requests.get(geo_url)
    seoul_geo = res.json()

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    folium.GeoJson(seoul_geo, name="경계", style_function=lambda f: {
        'fillColor': '#dddddd',
        'color': 'black',
        'weight': 3,
        'fillOpacity': 0.2
    }).add_to(m)

    min_v, max_v = df_users['이용자수'].min(), df_users['이용자수'].max()
    for feature in seoul_geo['features']:
        gu = feature['properties']['name']
        if gu in df_users['구'].values:
            center = shape(feature['geometry']).centroid
            val = df_users[df_users['구'] == gu]['이용자수'].values[0]
            norm = (val - min_v) / (max_v - min_v)
            folium.CircleMarker(
                location=[center.y, center.x],
                radius=10 + 30 * norm,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"{gu}: {val:,}명"
            ).add_to(m)
    folium.LayerControl().add_to(m)
    folium_static(m)

    top = df_users_sorted.iloc[0]
    st.success(f"✅ 도서관을 가장 많이 이용한 구는 **{top['구']}**입니다, 이용자 수는 총 **{top['이용자수']:,}명**입니다.")

    # 🔍 변수 중요도 분석
    st.subheader("🔍 변수 중요도 분석")
    X = df_stat.drop(columns=['자치구명', '도서관 방문자수'])
    y = df_stat['도서관 방문자수']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.markdown(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax2, color='skyblue')
    ax2.set_title("RandomForest 변수 중요도", fontproperties=font_prop)
    ax2.set_xlabel("중요도", fontproperties=font_prop)
    ax2.set_ylabel("변수", fontproperties=font_prop)
    ax2.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
    xticks = ax2.get_xticks()
    ax2.set_xticklabels([f"{x:.1f}" for x in xticks], fontproperties=font_prop)
    st.pyplot(fig2)

    st.subheader("📄 자치구별 통계 데이터")
    st.dataframe(df_stat)

# ---------------------------
# ✅ ② 학교·공공 도서관 통합 분석
# ---------------------------
else:
    st.title("📚 학교 & 공공 도서관 통합 분석 및 예측")

    df_school = load_school_data()
    df_public = load_public_data()

    option = st.radio("분석할 데이터셋을 선택하세요:", ["학교 도서관", "공공 도서관", "통합 비교"])

    if option == "학교 도서관":
        df = df_school.copy()
    elif option == "공공 도서관":
        df = df_public.copy()
    else:
        df = pd.concat([df_school, df_public], ignore_index=True)

    # 상관계수 히트맵 (matplotlib)
    st.subheader("📊 변수 간 상관관계")
    corr = df[["장서수", "사서수", "예산", "방문자수"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(corr, cmap="Blues")
    plt.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, fontproperties=font_prop)
    ax.set_yticklabels(corr.columns, fontproperties=font_prop)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", va='center', ha='center', color='black', fontproperties=font_prop)
    ax.set_title(f"{option} 변수 간 상관관계", fontproperties=font_prop)
    st.pyplot(fig)

    # 머신러닝 분석
    st.subheader("🔍 방문자 수 예측 및 변수 중요도")
    X = df[["장서수", "사서수", "예산"]].fillna(df[["장서수", "사서수", "예산"]].median())
    y = df["방문자수"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

    lr_model = LinearRegression()
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
    lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring="r2")

    st.subheader("📌 모델 성능 비교 (5-Fold 교차 검증)")
    st.markdown(f"""
    ✅ **RandomForest 평균 R²**: `{rf_scores.mean():.4f}`  
    ✅ **Linear Regression 평균 R²**: `{lr_scores.mean():.4f}`
    """)

    st.subheader("📄 사용된 데이터")
    st.dataframe(df)
