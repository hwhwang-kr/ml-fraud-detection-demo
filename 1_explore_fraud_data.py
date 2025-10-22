"""
2_explore_fraud_data.py
----------------------------------------
신용카드 사기 거래 탐지 프로젝트 - 데이터 탐색(EDA) 단계
----------------------------------------
목표:
- Impala에서 데이터 로드
- 기본 통계 및 사기율 확인
- 주요 패턴 시각화 (시간대별, 카테고리별, 국가별, 금액분포)
"""

import cml.data_v1 as cmldata
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

# 폰트 등록
font_path = '/home/cdsw/fonts/NotoSansKR-VariableFont_wght.ttf'
font_manager.fontManager.addfont(font_path)
rc('font', family='Noto Sans KR')
rc('axes', unicode_minus=False)  

# ==========================
# 1️⃣ Impala 연결
# ==========================
CONNECTION_NAME = "default-impala-aws"
conn = cmldata.get_connection(CONNECTION_NAME)

print("="*70)
print("✅ Impala 연결 성공")
print("="*70)

# ==========================
# 2️⃣ 분석용 데이터 로드 
# ==========================
QUERY = """
SELECT *
FROM hwhwang_finance_db.credit_card_transactions
WHERE rand() <= 0.02
"""
df = conn.get_pandas_dataframe(QUERY)
print(f"데이터 로드 완료 ✅ : {df.shape[0]:,}건, {df.shape[1]}컬럼")
print(df.head())

# ==========================
# 3️⃣ 기본 통계 & 사기율
# ==========================
total = len(df)
fraud = df['is_fraud'].sum()
fraud_rate = round(df['is_fraud'].mean() * 100, 3)

print("\n[데이터 요약]")
print(f"- 전체 거래 수: {total:,}")
print(f"- 사기 거래 수: {fraud:,}")
print(f"- 사기율: {fraud_rate:.3f}%")

# ==========================
# 4️⃣ 거래 금액 분포 (정상 vs 사기)
# ==========================
plt.figure(figsize=(10,5))
sns.histplot(df[df['is_fraud']==0]['transaction_amount'], bins=50,
             color='skyblue', label='정상', alpha=0.6)
sns.histplot(df[df['is_fraud']==1]['transaction_amount'], bins=50,
             color='red', label='사기', alpha=0.6)
plt.xlim(0, 1000000)
plt.legend()
plt.title("거래금액 분포 (정상 vs 사기)")
plt.xlabel("거래금액")
plt.ylabel("거래건수")
plt.tight_layout()
plt.show()

# ==========================
# 5️⃣ 시간대별 사기율
# ==========================
hourly = df.groupby('transaction_hour')['is_fraud'].mean().reset_index()
hourly['fraud_rate'] = hourly['is_fraud'] * 100

plt.figure(figsize=(10,5))
sns.barplot(x='transaction_hour', y='fraud_rate', data=hourly, hue='transaction_hour', palette='magma', legend=False)
plt.title("시간대별 사기율 (%)")
plt.xlabel("거래 시간대")
plt.ylabel("사기율 (%)")
plt.tight_layout()
plt.show()

# ==========================
# 6️⃣ 카테고리별 사기율 Top 10
# ==========================
cat = df.groupby('merchant_category')['is_fraud'].mean().reset_index()
cat['fraud_rate'] = cat['is_fraud'] * 100
cat = cat.sort_values('fraud_rate', ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(y='merchant_category', x='fraud_rate', data=cat, hue='merchant_category', palette='Reds_r', legend=False)
plt.title("가맹점 카테고리별 사기율 TOP 10")
plt.xlabel("사기율 (%)")
plt.ylabel("가맹점 카테고리")
plt.tight_layout()
plt.show()

# ==========================
# 7️⃣ 국가별 사기율
# ==========================
country = df.groupby('transaction_country')['is_fraud'].mean().reset_index()
country['fraud_rate'] = country['is_fraud'] * 100

plt.figure(figsize=(8,5))
sns.barplot(x='transaction_country', y='fraud_rate', data=country, hue='transaction_country', palette='coolwarm', legend=False)
plt.title("국가별 사기율 (%)")
plt.xlabel("거래 국가")
plt.ylabel("사기율 (%)")
plt.tight_layout()
plt.show()

# ==========================
# 8️⃣ 종료
# ==========================
print("\n✅ 데이터 탐색(EDA) 완료 - 주요 패턴 시각화 완료")
conn.close()
