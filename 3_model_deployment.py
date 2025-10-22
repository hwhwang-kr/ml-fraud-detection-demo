import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import cml.models_v1 as models

print("=" * 50, file=sys.stderr)
print("LOADING MODEL FILES...", file=sys.stderr)
print("=" * 50, file=sys.stderr)

THRESHOLD = 0.109  

# 모델 및 전처리 로드
try:
    model = joblib.load("best_model_XGBoost.pkl")
    print("✓ Model loaded", file=sys.stderr)
except Exception as e:
    print(f"✗ Model load failed: {e}", file=sys.stderr)
    model = None

try:
    scaler = joblib.load("scaler.pkl")
    # Scaler가 기대하는 컬럼 출력
    if hasattr(scaler, 'feature_names_in_'):
        print(f"✓ Scaler loaded - expects: {list(scaler.feature_names_in_)}", file=sys.stderr)
    else:
        print("✓ Scaler loaded", file=sys.stderr)
except Exception as e:
    print(f"✗ Scaler load failed: {e}", file=sys.stderr)
    scaler = None

try:
    label_encoders = joblib.load("label_encoders.pkl")
    print(f"✓ Label encoders loaded: {list(label_encoders.keys())}", file=sys.stderr)
except Exception as e:
    print(f"✗ Label encoders load failed: {e}", file=sys.stderr)
    label_encoders = {}

print("=" * 50, file=sys.stderr)
print("INITIALIZATION COMPLETE", file=sys.stderr)
print("=" * 50, file=sys.stderr)
sys.stderr.flush()


@models.cml_model
def predict(args):
    """
    PBJ Runtime 예측 함수
    """
    print("\n" + "=" * 50, file=sys.stderr)
    print("PREDICT CALLED", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    sys.stderr.flush()
    
    start_time = datetime.now()
    
    try:
        print(f"[1] Input: {type(args)}", file=sys.stderr)
        sys.stderr.flush()
        
        # 입력 파싱
        if isinstance(args, str):
            args = json.loads(args)
        
        if "request" in args:
            data = args["request"]
        else:
            data = args
        
        # 데이터 추출
        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            return {"error": f"Invalid input format: {type(data)}"}
        
        print(f"[2] Processing {len(records)} records", file=sys.stderr)
        sys.stderr.flush()
        
        # DataFrame 생성
        df = pd.DataFrame(records)
        
        expected_columns = [
            'transaction_amount',
            'merchant_category',
            'card_present',
            'transaction_country',
            'customer_age',
            'customer_income_level',
            'transaction_hour',
            'is_weekend',
            'days_since_last_transaction',
            'avg_amount_last_30days',
            'num_transactions_24h'
        ]
        
        # 컬럼 검증
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing: {missing}", file=sys.stderr)
            sys.stderr.flush()
            return {"error": f"Missing columns: {missing}"}
        
        print(f"[3] Columns OK", file=sys.stderr)
        print(f"[3] Current columns: {list(df.columns)}", file=sys.stderr)
        sys.stderr.flush()
        
        # Label Encoding
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                except Exception as e:
                    print(f"[ERROR] Encoding {col}: {e}", file=sys.stderr)
                    print(f"Valid classes: {le.classes_[:5]}", file=sys.stderr)
                    sys.stderr.flush()
                    raise
        
        print(f"[4] Encoding done", file=sys.stderr)
        sys.stderr.flush()
        
        # ✅ Scaler가 기대하는 정확한 순서대로 컬럼 정렬
        if hasattr(scaler, 'feature_names_in_'):
            expected_columns = list(scaler.feature_names_in_)
            print(f"[5] Using scaler's column order: {expected_columns}", file=sys.stderr)
        else:
            print(f"[5] Scaler has no feature_names_in_, using predefined order", file=sys.stderr)
        
        sys.stderr.flush()
        
        # Scaling & Prediction
        df = df[expected_columns]
        X_scaled = scaler.transform(df)
        
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs > THRESHOLD).astype(int)
        
        print(f"[5] Prediction done", file=sys.stderr)
        sys.stderr.flush()
        
        # 결과 생성
        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            results.append({
                "row_id": i,
                "is_fraud_predicted": int(pred),
                "fraud_probability": round(float(prob), 4)
            })
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        print(f"[6] SUCCESS - {len(results)} predictions in {latency_ms:.0f}ms", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        sys.stderr.flush()
        
        return {
            "predictions": results,
            "metadata": {
                "total_predictions": len(results),
                "latency_ms": round(latency_ms, 2)
            }
        }
    
    except Exception as e:
        print(f"\n[EXCEPTION] {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        
        return {
            "error": str(e),
            "type": type(e).__name__
        }