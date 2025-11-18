# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1hi5afSTbEUk0rhkLM1e4UqKRZJjjbwsH")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {    

     labels[0]: {
       "texts": ["유니콘", "마린 남편", "히나 인형"],
       "images": ["https://image2.1004gundam.com/item_images/goods/380/1376413529.jpg"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },

     labels[1]: {
       "texts": ["마린 바라기", "구름 과자", "공주"],
       "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },

     labels[2]: {
       "texts": ["고죠 아내", "코스프레", "히메"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://youtu.be/XXXXXXXXXXX"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
