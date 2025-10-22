"""
retrain_job.py
----------------------------------------
신용카드 사기 거래 탐지 - 주기적 재학습 Job
----------------------------------------
"""
import cml.data_v1 as cmldata
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ==========================
# 디렉토리 생성
# ==========================
os.makedirs("models", exist_ok=True)
os.makedirs("job_logs", exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print("="*70)
print(f"🔄 모델 재학습 Job 시작 - {timestamp}")
print("="*70)

# ==========================
# 1️⃣ Impala 연결 및 최신 데이터 로드
# ==========================
CONNECTION_NAME = "default-impala-aws"
conn = cmldata.get_connection(CONNECTION_NAME)

print("✅ Impala 연결 성공")

# 최근 90일 데이터로 학습 (조정 가능)
QUERY = """
SELECT *
FROM hwhwang_finance_db.credit_card_transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), 90)
"""
df = conn.get_pandas_dataframe(QUERY)
conn.close()

print(f"데이터 로드 완료 ✅ : {df.shape[0]:,}건, {df.shape[1]}컬럼")

# ==========================
# 2️⃣ Feature Engineering
# ==========================
print("\n[전처리 시작]")

cat_cols = ['merchant_category', 'transaction_country', 'customer_income_level']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

drop_cols = ['transaction_id', 'customer_id', 'transaction_date']
df = df.drop(columns=drop_cols)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("전처리 완료 ✅")

# ==========================
# 3️⃣ 데이터 분할
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================
# 4️⃣ 모델 정의
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
}

# ==========================
# 5️⃣ MLflow 실험 설정
# ==========================
mlflow.set_experiment("fraud_detection_retrain_job")

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    
    with mlflow.start_run(run_name=f"{name}_{timestamp}"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        
        # 로그 기록
        mlflow.log_param("model_type", name)
        mlflow.log_param("training_date", timestamp)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Precision_Fraud", report['1']['precision'])
        mlflow.log_metric("Recall_Fraud", report['1']['recall'])
        mlflow.log_metric("F1_Fraud", report['1']['f1-score'])
        mlflow.sklearn.log_model(model, "model")
        
        results.append({
            "Model": name,
            "AUC": auc,
            "Precision_Fraud": report['1']['precision'],
            "Recall_Fraud": report['1']['recall'],
            "F1_Fraud": report['1']['f1-score']
        })
        
        print(f"✅ {name} 완료 | AUC: {auc:.3f} | F1 (Fraud): {report['1']['f1-score']:.3f}")

# ==========================
# 6️⃣ 모델 비교 결과
# ==========================
result_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
print("\n" + "="*70)
print("[모델 비교 결과]")
print("="*70)
print(result_df.to_string(index=False))

best_model_name = result_df.iloc[0]['Model']
best_auc = result_df.iloc[0]['AUC']
best_model = models[best_model_name]

print(f"\n🏆 최고 성능 모델: {best_model_name} (AUC: {best_auc:.4f})")

# ==========================
# 7️⃣ 기존 모델과 비교 및 업데이트
# ==========================
model_filename = f"best_model_{best_model_name.replace(' ', '_')}.pkl"
model_updated = False

try:
    # 기존 모델 로드
    old_model = joblib.load(model_filename)
    old_scaler = joblib.load("scaler.pkl")
    
    # 기존 모델 성능 측정
    X_test_rescaled = old_scaler.transform(X_test)  # 기존 scaler 사용
    old_probs = old_model.predict_proba(X_test_rescaled)[:, 1]
    old_auc = roc_auc_score(y_test, old_probs)
    
    print(f"\n📊 기존 모델 AUC: {old_auc:.4f}")
    print(f"📊 새 모델 AUC: {best_auc:.4f}")
    
    # 성능 비교 (최소 0.005 이상 개선되었을 때만 교체)
    if best_auc > old_auc + 0.005:
        improvement = ((best_auc - old_auc) / old_auc) * 100
        print(f"\n✅ 성능 개선 감지! ({improvement:.2f}% 향상)")
        print("🔄 모델 업데이트 진행...")
        
        # 기존 모델 백업
        backup_filename = f"models/backup_{model_filename.replace('.pkl', '')}_{timestamp}.pkl"
        joblib.dump(old_model, backup_filename)
        print(f"💾 기존 모델 백업: {backup_filename}")
        
        # 새 모델로 교체
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoders, "label_encoders.pkl")
        
        model_updated = True
        print(f"✅ 새 모델 배포 완료: {model_filename}")
        
    else:
        print(f"\n⚠️ 성능 개선 미미 (차이: {best_auc - old_auc:.4f})")
        print("기존 모델 유지")
        
except FileNotFoundError:
    print("\n⚠️ 기존 모델 없음. 새 모델을 초기 배포합니다.")
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    model_updated = True
    print(f"✅ 초기 모델 저장: {model_filename}")

# ==========================
# 8️⃣ 버전 관리 (항상 저장)
# ==========================
versioned_model = f"models/model_{best_model_name.replace(' ', '_')}_{timestamp}.pkl"
joblib.dump(best_model, versioned_model)
print(f"💾 버전 모델 저장: {versioned_model}")

# ==========================
# 9️⃣ Job 실행 로그 저장
# ==========================
job_log = {
    "timestamp": timestamp,
    "execution_time": datetime.now().isoformat(),
    "data_range": "90 days",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "best_model": best_model_name,
    "best_auc": float(best_auc),
    "all_results": results,
    "model_updated": model_updated,
    "old_auc": float(old_auc) if 'old_auc' in locals() else None,
    "improvement": float(best_auc - old_auc) if 'old_auc' in locals() else None
}

log_filename = f"job_logs/retrain_log_{timestamp}.json"
with open(log_filename, "w") as f:
    json.dump(job_log, f, indent=2)

print(f"\n📝 실행 로그 저장: {log_filename}")

# ==========================
# 🔟 최종 요약
# ==========================
print("\n" + "="*70)
print("✅ 재학습 Job 완료")
print("="*70)
print(f"🏆 최고 성능: {best_model_name} (AUC: {best_auc:.4f})")
print(f"📦 모델 업데이트: {'Yes ✅' if model_updated else 'No ⚠️'}")
print(f"💾 로그 파일: {log_filename}")
print(f"🕐 완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)