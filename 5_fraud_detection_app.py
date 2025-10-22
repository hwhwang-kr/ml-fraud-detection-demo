import streamlit as st
import requests
import pandas as pd
import json
import os

st.set_page_config(page_title="💳 Fraud Detection - 실시간 예측", layout="wide")

# 사이드바 크기 조정
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 400px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💳 신용카드 사기 거래 실시간 예측")
st.write("모델 배포 API를 호출하여 입력된 거래의 사기 확률을 실시간으로 예측합니다.")

# ==========================
# 🔧 API 정보 설정
# ==========================
st.sidebar.header("🔐 API 설정")
API_URL = st.sidebar.text_input(
    "API URL", 
    value="https://modelservice.ml-dbfc64d1-783.go01-dem.ylcu-atmi.cloudera.site/model"
)

default_access_key = os.getenv("ACCESS_KEY", "")
ACCESS_KEY = st.sidebar.text_input(
    "Access Key", 
    value=default_access_key,
    type="password",
    help="필수 항목입니다. 환경변수 ACCESS_KEY로도 설정 가능합니다."
)

st.sidebar.markdown("---")

# ==========================
# 🎚️ 위험도 기준
# ==========================
st.sidebar.header("🎚️ 위험도 판단 기준")
st.sidebar.markdown("""
- ✅ **안전**: 10% 미만
- ⚠️ **의심**: 10% ~ 30%
- 🔶 **매우 의심**: 30% ~ 60%
- 🚨 **위험**: 60% 이상
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📘 사용 가이드")
st.sidebar.markdown("""
**1. API 설정**
- API URL과 Access Key 입력

**2. 거래 정보 입력**
- 테이블에서 거래 데이터 편집

**3. 예측 실행**
- 모델 예측 실행 버튼 클릭
- 사기 확률 및 위험도 확인
""")

# ==========================
# 🧾 입력 데이터 테이블 생성
# ==========================
st.subheader("거래 입력 데이터")

sample = {
    "transaction_amount": 23500,
    "merchant_category": "편의점",
    "transaction_country": "KR",
    "customer_income_level": "MEDIUM",
    "transaction_hour": 13,
    "num_transactions_24h": 2,
    "avg_amount_last_30days": 28000,
    "days_since_last_transaction": 1,
    "is_weekend": 0,
    "card_present": 1,
    "customer_age": 21
}

df = pd.DataFrame([sample])
edited_df = st.data_editor(df, num_rows="fixed", use_container_width=True)

# ==========================
# 🚀 모델 예측
# ==========================
if st.button("🚀 모델 예측 실행", type="primary"):
    if not ACCESS_KEY:
        st.error("⚠️ Access Key를 입력해주세요!")
        st.stop()
    
    if edited_df.empty:
        st.warning("⚠️ 최소 1개 이상의 거래 데이터를 입력해주세요.")
        st.stop()
    
    records = edited_df.to_dict(orient="records")
    single_record = records[0] if len(records) == 1 else records
    
    payload = {
        "accessKey": ACCESS_KEY,
        "request": single_record
    }
    
    headers = {"Content-Type": "application/json"}
    request_url = API_URL
    
    with st.spinner("📤 모델 호출 중..."):
        try:
            response = requests.post(
                request_url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"❌ HTTP 에러: {response.status_code}")
                st.code(response.text)
                st.stop()
            
            result = response.json()
            st.success("✅ 예측 완료")
            
            predictions = None
            metadata = None
            
            if "response" in result:
                response_data = result["response"]
                predictions = response_data.get("predictions", [])
                metadata = response_data.get("metadata", {})
            elif "predictions" in result:
                predictions = result["predictions"]
                metadata = result.get("metadata", {})
            
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                prob = pred.get("fraud_probability", 0)
                prob_percent = prob * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("사기 확률", f"{prob_percent:.2f}%")
                
                with col2:
                    if prob_percent >= 60:
                        risk_level = "🚨 위험"
                        risk_desc = "즉시 차단 권장"
                    elif prob_percent >= 30:
                        risk_level = "🔶 매우 의심"
                        risk_desc = "추가 확인 필수"
                    elif prob_percent >= 10:
                        risk_level = "⚠️ 의심"
                        risk_desc = "모니터링 권장"
                    else:
                        risk_level = "✅ 안전"
                        risk_desc = "승인 가능"
                    
                    st.metric("위험도", risk_level)
                
                with col3:
                    if metadata and "latency_ms" in metadata:
                        st.metric("응답 시간", f"{metadata['latency_ms']:.2f}ms")
                
                st.subheader("📊 위험도 평가")
                
                if prob_percent >= 60:
                    st.error(f"🚨 **위험 거래** ({prob_percent:.1f}%) - {risk_desc}")
                elif prob_percent >= 30:
                    st.warning(f"🔶 **매우 의심 거래** ({prob_percent:.1f}%) - {risk_desc}")
                elif prob_percent >= 10:
                    st.info(f"⚠️ **의심 거래** ({prob_percent:.1f}%) - {risk_desc}")
                else:
                    st.success(f"✅ **안전 거래** ({prob_percent:.1f}%) - {risk_desc}")
                
                st.progress(min(prob, 1.0))
                
            else:
                st.error("❌ 예측 결과를 찾을 수 없습니다.")
                st.json(result)
            
            with st.expander("📋 상세 응답 보기"):
                st.json(result)
            
            with st.expander("📤 전송된 요청 데이터"):
                st.json(payload)
                
        except requests.exceptions.Timeout:
            st.error("❌ API 요청 시간 초과 (30초)")
        except requests.exceptions.ConnectionError:
            st.error("❌ API 서버에 연결할 수 없습니다. URL을 확인해주세요.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API 호출 실패: {e}")
        except json.JSONDecodeError:
            st.error("❌ 응답을 JSON으로 파싱할 수 없습니다.")
            st.code(response.text)
        except Exception as e:
            st.error(f"❌ 예상치 못한 오류: {e}")
            import traceback
            st.code(traceback.format_exc())