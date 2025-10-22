"""
3_model_training_experiments_final.py
----------------------------------------
신용카드 사기 거래 탐지 프로젝트 - 모델 학습 및 실험 단계 (MLflow 연동 포함)
----------------------------------------
목표:
- Impala에서 데이터 로드
- 데이터 전처리 및 Feature Engineering
- 여러 알고리즘 학습 및 비교
- MLflow(CML Experiments)로 실험 자동 추적
- 최적 모델 선정 및 저장
"""

import cml.data_v1 as cmldata
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ==========================
# 1️⃣ Impala 연결 및 데이터 로드
# ==========================
CONNECTION_NAME = "default-impala-aws"
conn = cmldata.get_connection(CONNECTION_NAME)

print("="*70)
print("✅ Impala 연결 성공")
print("="*70)

QUERY = """
SELECT *
FROM hwhwang_finance_db.credit_card_transactions
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
mlflow.set_experiment("fraud_detection_experiment")

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        # -----------------------------
        # ✅ ① 최적 threshold 탐색
        # -----------------------------
        from sklearn.metrics import precision_recall_curve, f1_score

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        print(f"📏 Best threshold for {name}: {best_threshold:.3f}")

        # -----------------------------
        # ✅ ② 최적 threshold로 재예측
        # -----------------------------
        preds = (probs >= best_threshold).astype(int)

        # -----------------------------
        # ✅ ③ 지표 계산
        # -----------------------------
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        # -----------------------------
        # ✅ ④ MLflow 기록
        # -----------------------------
        mlflow.log_param("model_type", name)
        mlflow.log_param("best_threshold", float(best_threshold))
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Precision_Fraud", report['1']['precision'])
        mlflow.log_metric("Recall_Fraud", report['1']['recall'])
        mlflow.log_metric("F1_Fraud", report['1']['f1-score'])
        mlflow.sklearn.log_model(model, "model")

        # -----------------------------
        # ✅ ⑤ 결과 저장
        # -----------------------------
        results.append({
            "Model": name,
            "AUC": auc,
            "Precision_Fraud": report['1']['precision'],
            "Recall_Fraud": report['1']['recall'],
            "F1_Fraud": report['1']['f1-score'],
            "Best_Threshold": best_threshold
        })

        print(
            f"✅ {name} 완료 | AUC: {auc:.3f} | "
            f"F1 (Fraud): {report['1']['f1-score']:.3f} | "
            f"Threshold: {best_threshold:.3f}"
        )


# ==========================
# 6️⃣ 모델 비교 결과
# ==========================
result_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
print("\n[모델 비교 결과]")
print(result_df.to_markdown(index=False))

best_model_name = result_df.iloc[0]['Model']
best_model = models[best_model_name]

# ==========================
# 7️⃣ 모델 및 전처리 도구 저장
# ==========================
joblib.dump(best_model, f"best_model_{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print(f"\n💾 최적 모델 저장 완료: best_model_{best_model_name.replace(' ', '_')}.pkl")
print("💾 전처리 도구 (Scaler, LabelEncoders) 저장 완료")
print("\n✅ 모델 학습 및 Experiments 등록 완료 ✅")
