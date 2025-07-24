import streamlit as st
st.set_page_config(page_title="서울시 도서관 통합 분석 및 예측", layout="wide")

st.title("📚 서울특별시 도서관 통합 분석 및 예측")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import requests
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# ✅ 1. 한글 폰트 설정
# ---------------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# ---------------------------
# ✅ 2. 데이터 불러오기 함수 (학교 + 자치구)
# ---------------------------
@st.cache_data
def load_school_data():
    base_path = os.path.dirname(__file__)
    file1 = os.path.join(base_path, "문화체육관광부_국가도서관통계_전국학교도서관통계_20231231.csv")
    file3 = os.path.join(base_path, "서울시 학교별 학교도서관 현황.csv")

    df1 = pd.read_csv(file1, encoding="cp949")
    df3 = pd.read_csv(file3, encoding="cp949")

    # 서울시 데이터 필터링
    df1_seoul = df1[df1['행정구역'] == '서울'].copy()
    df3_seoul = df3[df3['시도교육청'] == '서울특별시교육청'].copy()

    # 주요 컬럼만 추출
    df1_seoul = df1_seoul[['도서관명', '시군구', '장서수(인쇄)', '사서수', '대출자수', '대출권수', '도서예산(자료구입비)']]
    df1_seoul.columns = ['학교명', '지역', '장서수', '사서수', '대출자수', '대출권수', '자료구입비']

    df3_seoul = df3_seoul[['학교명', '지역', '자료구입비예산액', '도서관대출자료수']]
    df3_seoul.columns = ['학교명', '지역', '자료구입비', '대출자료수']

    # 병합
    df_merged = pd.merge(df1_seoul, df3_seoul, on=['학교명', '지역'], how='outer')

    # 자료구입비 합산
    if '자료구입비_x' in df_merged.columns and '자료구입비_y' in df_merged.columns:
        df_merged['자료구입비'] = df_merged[['자료구입비_x', '자료구입비_y']].sum(axis=1)
        df_merged.drop(columns=['자료구입비_x', '자료구입비_y'], inplace=True)

    # 숫자형 변환 및 결측치 처리
    numeric_cols = ['장서수', '사서수', '대출자수', '대출권수', '자료구입비', '대출자료수']
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)

    return df_merged

@st.cache_data
def load_gu_data():
    df = pd.read_csv("공공도서관 자치구별 통계 파일.csv", encoding='cp949', header=1)
    df = df[df.iloc[:, 0] != '소계']
    df.columns = [
        '자치구명','개소수','좌석수','자료수_도서','자료수_비도서','자료수_연속간행물',
        '도서관 방문자수','연간대출책수','직원수','직원수_남','직원수_여','예산'
    ]
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

df_school = load_school_data()
df_gu = load_gu_data()

# ---------------------------
# ✅ 3. 학교 데이터 머신러닝 분석
# ---------------------------
st.subheader("🤖 학교도서관 이용자 수 예측 (RandomForest)")
st.markdown("학교 도서관의 **대출자수(이용자 수)**를 예측하고, 어떤 변수가 영향을 많이 주는지 분석했습니다.")

X = df_school[['장서수', '사서수', '자료구입비']]
y = df_school['대출자수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

# 변수 중요도
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importance.plot.barh(ax=ax, color='skyblue')
ax.set_title("📌 변수 중요도 (대출자수에 대한 영향)", fontproperties=font_prop)
ax.set_xlabel("중요도", fontproperties=font_prop)
ax.set_ylabel("변수", fontproperties=font_prop)
if font_prop:
    ax.set_yticklabels(importance.index, fontproperties=font_prop)
st.pyplot(fig)

top_var = importance.sort_values(ascending=False).index[0]
st.info(f"📊 **대출자수(이용자 수)에 가장 큰 영향을 미치는 변수는 `{top_var}`입니다.**")

# ---------------------------
# ✅ 4. 서울시 자치구별 지도 시각화 (shapely 제거)
# ---------------------------
st.subheader("🗺️ 서울시 자치구별 도서관 이용자 수 지도")
st.markdown("서울특별시 각 자치구의 도서관 방문자 수를 지도 위에 시각화했습니다. 마커 크기가 클수록 방문자 수가 많습니다.")

df_users = df_gu[['자치구명', '도서관 방문자수']].copy()
df_users.columns = ['구', '이용자수']

geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
res = requests.get(geo_url)
seoul_geo = res.json()

# 중심 좌표 계산 함수 (shapely 없이)
def get_center(geometry):
    coords = geometry['coordinates'][0]
    lon = sum([c[0] for c in coords]) / len(coords)
    lat = sum([c[1] for c in coords]) / len(coords)
    return lat, lon

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
        lat, lon = get_center(feature['geometry'])
        val = df_users[df_users['구'] == gu]['이용자수'].values[0]
        norm = (val - min_v) / (max_v - min_v)
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 + 30 * norm,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"{gu}: {val:,}명"
        ).add_to(m)

folium.LayerControl().add_to(m)
folium_static(m)

top = df_users.sort_values(by='이용자수', ascending=False).iloc[0]
st.success(f"✅ 도서관을 가장 많이 이용한 구는 **{top['구']}**, 이용자 수는 총 **{top['이용자수']:,}명**입니다.")

# ---------------------------
# ✅ 5. 학교 통합 데이터 테이블 (그래프 아래 배치)
# ---------------------------
st.subheader("📄 서울시 학교도서관 통합 데이터")
st.markdown("문화체육관광부, 교육통계, 서울시 데이터를 통합한 결과입니다.")
st.dataframe(df_school)
