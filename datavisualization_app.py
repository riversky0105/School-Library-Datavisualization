import streamlit as st
st.set_page_config(page_title="서울시 학교도서관 이용 분석 및 예측", layout="wide")

st.title("📚 서울특별시 학교도서관 이용자 수 분석 및 예측")

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
# ✅ 1. 한글 폰트 설정
# ---------------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# ---------------------------
# ✅ 2. 데이터 불러오기 함수
# ---------------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv("문화체육관광부_국가도서관통계_전국학교도서관통계_20231231.csv", encoding="cp949")
    df2 = pd.read_csv("학교도서관현황_20250717223352.csv", encoding="cp949")
    df3 = pd.read_csv("서울시 학교별 학교도서관 현황.csv", encoding="cp949")

    # 서울시 데이터 필터링
    df1_seoul = df1[df1['행정구역'] == '서울'].copy()
    df3_seoul = df3[df3['시도교육청'] == '서울특별시교육청'].copy()

    # 주요 컬럼만 추출
    df1_seoul = df1_seoul[['도서관명', '시군구', '장서수(인쇄)', '사서수', '대출자수', '대출권수', '도서예산(자료구입비)']]
    df1_seoul.columns = ['학교명', '지역', '장서수', '사서수', '대출자수', '대출권수', '자료구입비']

    df3_seoul = df3_seoul[['학교명', '지역', '자료구입비예산액', '도서관대출자료수', '전년도전체학생수', '도서관대여학생수', '1인당대출자료수']]
    df3_seoul.columns = ['학교명', '지역', '자료구입비', '대출자료수', '전체학생수', '대여학생수', '1인당대출자료수']

    # 병합
    df_merged = pd.merge(df1_seoul, df3_seoul, on=['학교명', '지역'], how='outer')

    # 자료구입비 합산
    if '자료구입비_x' in df_merged.columns and '자료구입비_y' in df_merged.columns:
        df_merged['자료구입비'] = df_merged[['자료구입비_x', '자료구입비_y']].sum(axis=1)
        df_merged.drop(columns=['자료구입비_x', '자료구입비_y'], inplace=True)

    # 숫자형 변환 및 결측치 처리
    numeric_cols = ['장서수', '사서수', '대출자수', '대출권수', '자료구입비',
                    '대출자료수', '전체학생수', '대여학생수', '1인당대출자료수']
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)

    return df_merged

df = load_data()

# ---------------------------
# ✅ 3. 데이터 테이블 표시
# ---------------------------
st.subheader("📄 서울시 학교도서관 통합 데이터")
st.markdown("문화체육관광부, 교육통계, 서울시 데이터를 통합한 결과입니다.")
st.dataframe(df)

# ---------------------------
# ✅ 4. 머신러닝 학습 및 평가
# ---------------------------
st.subheader("🤖 머신러닝 예측 (RandomForest)")
st.markdown("학교 도서관의 **대출자수(이용자 수)**를 예측하고, 어떤 변수가 영향을 많이 주는지 분석했습니다.")

# 독립 변수와 종속 변수
X = df[['장서수', '사서수', '자료구입비', '전체학생수', '대여학생수']]
y = df['대출자수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

# ---------------------------
# ✅ 5. 변수 중요도 시각화
# ---------------------------
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importance.plot.barh(ax=ax, color='skyblue')
ax.set_title("📌 변수 중요도 (대출자수에 대한 영향)", fontproperties=font_prop)
ax.set_xlabel("중요도", fontproperties=font_prop)
ax.set_ylabel("변수", fontproperties=font_prop)
if font_prop:
    ax.set_yticklabels(importance.index, fontproperties=font_prop)
st.pyplot(fig)

# ---------------------------
# ✅ 6. 주요 인사이트
# ---------------------------
top_var = importance.sort_values(ascending=False).index[0]
st.info(f"📊 **대출자수(이용자 수)에 가장 큰 영향을 미치는 변수는 `{top_var}`입니다.**")
