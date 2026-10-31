import base64
import csv
import datetime
import glob
import json
import random
import time
import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from plotly import express as px
from plotly import graph_objects as go
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="VisionAI - Deteksi Hewan Ternak", page_icon="🐄",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 20% -10%,#0f2a3a 0%,#0a111e 55%);color:#e6f1ff}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none}
.block-container{max-width:1200px;padding:1rem 2rem !important}
@keyframes scan{0%{top:0}100%{top:100%}}
@keyframes fadeIn{0%{opacity:0;transform:translateY(14px)}100%{opacity:1;transform:none}}
.brand{font-size:26px;font-weight:800;color:#fff;line-height:1.1}.brand b{color:#1de9c8}
.brand small{display:block;font-size:10px;letter-spacing:.3em;color:#8aa4b8;font-weight:500}
.status{text-align:right;color:#1de9c8;font-size:13px;font-weight:600;padding-top:10px}
.hero{border-radius:18px;overflow:hidden;border:1px solid #1b3a4a;background:#0b1626;animation:fadeIn .8s ease}
.hero-row{display:grid;grid-template-columns:repeat(4,1fr);height:300px}
.cell{position:relative;background-size:cover;background-position:center}
.cell:after{content:"";position:absolute;inset:0;background:linear-gradient(0deg,#0b1626 0%,#0b162600 65%)}
.frame{position:absolute;inset:14% 12% 24%;border:2px solid #1de9c8;border-radius:4px;z-index:2;overflow:hidden}
.frame:before{content:"";position:absolute;left:0;right:0;height:2px;background:#1de9c8;box-shadow:0 0 10px #1de9c8;animation:scan 2.6s linear infinite}
.tag{position:absolute;top:9%;left:12%;z-index:3;background:#1de9c8;color:#04201b;font-weight:800;font-size:12px;padding:2px 8px;letter-spacing:.08em}
.hero-txt{padding:18px 26px 22px}.hero-txt h1{margin:0;font-size:36px;line-height:1.1;color:#fff}.hero-txt h1 span{color:#1de9c8}
.hero-txt p{color:#9fb6c8;margin:8px 0 0}
.main-title{font-size:26px;font-weight:800;color:#fff;margin:8px 0 12px}.main-title b{color:#1de9c8}
.section-title{font-size:14px;font-weight:700;letter-spacing:.16em;color:#f5c56b;margin:8px 0 10px}
.explain-box,.card{background:#0f1b2d;border:1px solid #1b3a4a;border-radius:12px;padding:14px 16px;color:#c9d8e6}
.metric-box{background:#0f1b2d;border:1px solid #1b3a4a;border-radius:12px;padding:12px;text-align:center;color:#8aa4b8;font-weight:600}
.metric-box span{display:block;font-size:26px;color:#1de9c8}
.pal{text-align:center;color:#e6f1ff;font-size:13px;font-weight:600}
.pal div.sw{height:46px;border-radius:8px;margin-bottom:4px;border:1px solid #1b3a4a}
.login-h{font-size:42px;font-weight:800;color:#fff;margin:34px 0 4px}.login-h span{color:#1de9c8}
.collage{display:grid;grid-template-columns:1fr 1fr;gap:12px;height:440px}.collage .cell{border-radius:12px;border:1px solid #1b3a4a}
.stButton>button{background:#1de9c8;color:#04201b;font-weight:800;border:0;border-radius:10px;letter-spacing:.05em}
.stButton>button:hover{background:#5ff3da;color:#04201b}
[data-testid="stImage"] img{border:1px solid #1de9c8;border-radius:12px;box-shadow:0 0 18px #1de9c833}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #1de9c8;background:#0f1b2d}
center,.footer-center{color:#8aa4b8;font-size:13px}
@media(max-width:760px){.hero-row{grid-template-columns:1fr 1fr;height:420px}.hero-txt h1{font-size:26px}.collage{height:260px}}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:8px}
[data-testid="stSidebar"] div[role="radiogroup"]>label{background:#0f1b2d;border:1px solid #1b3a4a;border-radius:10px;padding:10px 12px;width:100%;transition:all .2s}
[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{border-color:#1de9c8}
[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked){border-color:#1de9c8;background:#12354a;box-shadow:0 0 10px #1de9c844}
.footer{display:flex;align-items:center;justify-content:center;gap:16px;margin-top:28px;padding:18px 10px 6px;border-top:1px solid #1b3a4a;color:#8aa4b8;font-size:13px;line-height:1.5}
.footer img{height:58px;border-radius:10px;background:#fff;padding:4px}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/best.pt"
CONF_THR = 0.30  # ambang YOLO (dikunci)
CNN_THR = 0.30   # ambang filter CNN (dikunci)
CNN_PATH = "model/model_cnn.h5"
CNN_LABELS = ["kerbau", "kuda", "nilgai", "sapi"]  # URUTAN HARUS SAMA DENGAN SAAT TRAINING CNN
CNN_RESCALE = True  # True bila saat training gambar dibagi 255
FONT = dict(color="#e6f1ff")
COLORS = ["#4f63c8", "#22b8cf", "#2fb344", "#f5a623", "#a855f7", "#e5484d"]

for k, v in {"page": "home", "user_name": "", "user_campus": "", "used_feature": False,
             "last_mode": None, "last_pred": None, "last_conf": None, "sample": None, "detect": False, "src_id": None}.items():
    st.session_state.setdefault(k, v)


def go_to(page: str) -> None:
    st.session_state["page"] = page


# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    yolo_model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    yolo_model = None


@st.cache_resource
def load_cnn():
    if not os.path.exists(CNN_PATH):
        return None
    from tensorflow.keras.models import load_model  # import malas: aplikasi tetap jalan tanpa TensorFlow
    return load_model(CNN_PATH, compile=False)

try:
    cnn_model = load_cnn()
except Exception as e:
    cnn_model = None
    st.sidebar.warning(f"Filter CNN tidak aktif: {e}")


def cnn_check(img: Image.Image) -> tuple[str, float]:
    """Tahap 1: klasifikasi CNN. Return (label teratas, probabilitas)."""
    shp = cnn_model.input_shape
    h, w, ch = (shp[1] or 224), (shp[2] or 224), (shp[3] if len(shp) > 3 else 3)
    x = img.resize((w, h))
    arr = np.asarray(x.convert("L") if ch == 1 else x, dtype=np.float32)
    if ch == 1:
        arr = arr[..., None]
    if CNN_RESCALE:
        arr = arr / 255.0
    probs = np.asarray(cnn_model.predict(arr[None], verbose=0)[0], dtype=float).ravel()
    i = int(np.argmax(probs))
    return (CNN_LABELS[i] if len(probs) == len(CNN_LABELS) else f"Kelas {i}"), float(probs[i])

# ==========================
# FUNGSI ANALISIS
# ==========================
NAMES = {"Hitam": (20, 20, 20), "Putih": (240, 240, 240), "Abu-abu": (128, 128, 128), "Merah": (200, 40, 40),
         "Hijau": (60, 160, 60), "Biru": (50, 80, 200), "Kuning": (230, 210, 60), "Oranye": (240, 140, 40),
         "Cokelat": (130, 85, 50), "Krem": (225, 210, 180), "Cyan": (50, 200, 210), "Ungu": (140, 70, 170)}


def color_name(rgb):
    return min(NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(NAMES[n], rgb)))


def analyze_colors(img: Image.Image, k: int = 5) -> dict:
    """Analisis warna seluruh gambar: palet K-Means, histogram RGB, kecerahan."""
    px_ = np.asarray(img.convert("RGB")).reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = px_[rng.choice(len(px_), min(20000, len(px_)), replace=False)].astype(np.float32)
    k = max(1, min(k, len(np.unique(sample, axis=0))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    palette = []
    for i in np.argsort(-counts):
        rgb = tuple(int(v) for v in centers[i])
        palette.append({"hex": "#%02X%02X%02X" % rgb, "pct": round(100 * counts[i] / counts.sum(), 1),
                        "name": color_name(rgb)})
    hist = {c: np.histogram(px_[:, i], bins=32, range=(0, 256))[0] for i, c in enumerate("RGB")}
    gray = px_ @ np.array([0.299, 0.587, 0.114])
    return {"palette": palette, "hist": hist, "mean_brightness": float(gray.mean()),
            "bright": np.histogram(gray, bins=32, range=(0, 256))[0]}


def to_png_bytes(arr_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, format="PNG")
    return buf.getvalue()


def style_fig(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(15,27,45,0.6)", font=FONT)
    return fig


# ==========================
# PAGES
# ==========================
SAMPLES = {"kerbau": "sample_images/contoh_buffelo.jpg", "sapi": "sample_images/contoh_cow.jpg",
           "nilgai": "sample_images/contoh_nilgai.jpg", "kuda": "sample_images/contoh_horse.jpg"}


@st.cache_data
def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def cells(keys) -> str:
    return "".join(f'<div class="cell" style="background-image:url(data:image/jpeg;base64,{b64(SAMPLES[k])})">'
                   f'<div class="frame"></div><span class="tag">{k.upper()}</span></div>' for k in keys)


def navbar():
    logged = bool(st.session_state["user_name"])
    c = st.columns([3, 1, 1, 2])
    c[0].markdown('<div class="brand">Vision<b>AI</b><small>ANIMAL DETECTION SYSTEM</small></div>', unsafe_allow_html=True)
    c[1].button("Beranda", key="nav_h", on_click=go_to, args=("home",), width="stretch")
    c[2].button("Deteksi", key="nav_d", on_click=go_to, args=("main" if logged else "register",), width="stretch")
    ok = yolo_model is not None
    c[3].markdown(f'<div class="status">● MODEL YOLOv8 {"SIAP" if ok else "TIDAK TERSEDIA"}</div>', unsafe_allow_html=True)


def halaman_awal():
    navbar()
    st.markdown(f"""<div class="hero"><div class="hero-row">{cells(SAMPLES)}</div><div class="hero-txt">
<h1>BUKAN SEKADAR GAMBAR,<br><span>INI ADALAH DATA</span></h1>
<p>Deteksi otomatis hewan: kerbau, sapi, nilgai, dan kuda dengan teknologi deep learning (YOLOv8).</p></div></div>""", unsafe_allow_html=True)
    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    mid.button("▶  PLAY — MULAI DETEKSI", key="play", width="stretch", on_click=go_to,
               args=("main" if st.session_state["user_name"] else "register",))
    st.markdown('<div class="section-title">TENTANG VISIONAI</div>', unsafe_allow_html=True)
    for col, (t, d) in zip(st.columns(3), [("Deteksi YOLOv8", "Bounding box, nama kelas, dan confidence untuk setiap objek."),
                                           ("Analisis Warna", "Palet warna dominan, histogram RGB, dan distribusi kecerahan."),
                                           ("Di Luar Kelas", "Gambar tanpa objek yang yakin ditolak dengan pesan khusus.")]):
        col.markdown(f'<div class="card"><b style="color:#1de9c8">{t}</b><br>{d}</div>', unsafe_allow_html=True)


def halaman_registrasi():
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="brand">Vision<b>AI</b><small>ANIMAL DETECTION SYSTEM</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="login-h">Selamat <span>Datang</span></div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#9fb6c8">Isi data singkat Anda untuk melanjutkan ke sistem deteksi hewan.</p>', unsafe_allow_html=True)
        with st.form("form_reg"):
            name = st.text_input("Nama Lengkap", value=st.session_state["user_name"], placeholder="Masukkan nama Anda")
            campus = st.text_input("Nama Kampus", value=st.session_state["user_campus"], placeholder="Masukkan nama kampus")
            if st.form_submit_button("MASUK  →", width="stretch"):
                if not name.strip() or not campus.strip():
                    st.warning("Mohon isi Nama dan Nama Kampus sebelum melanjutkan.")
                else:
                    st.session_state.update(user_name=name.strip(), user_campus=campus.strip(), page="main")
                    st.rerun()
        st.button("⬅️ Kembali ke Beranda", on_click=go_to, args=("home",))
        st.markdown('<hr style="border-color:#1b3a4a"><div class="footer-center">VisionAI © 2026 — Intelligent Animal Detection System</div>', unsafe_allow_html=True)
    right.markdown(f'<div class="collage">{cells(["sapi", "kuda"])}</div>'
                   '<p style="text-align:right;letter-spacing:.25em;color:#1de9c8;font-size:12px">DETECT / ANALYZE / PROTECT</p>', unsafe_allow_html=True)


def tampil_deteksi(img: Image.Image, conf_thr: float, col1, col2, cnn_thr: float = CNN_THR):
    if yolo_model is None:
        st.error("Model YOLO belum tersedia. Pastikan file model/best.pt ada.")
        return
    if cnn_model is not None:  # Tahap 1: CNN sebagai filter sebelum YOLOv8
        with st.spinner("🔍 Tahap 1: validasi gambar dengan CNN..."):
            label, c = cnn_check(img)
        if c < cnn_thr:
            kelas = ", ".join(str(n) for n in yolo_model.names.values())
            with col2:
                st.markdown(f'<div class="card" style="border-color:#ff6b57"><b style="color:#ff6b57;font-size:20px">IMAGE NOT RECOGNIZED</b><br>'
                            f'Maaf, gambar yang Anda unggah tidak termasuk ke dalam kelas hewan yang didukung oleh VisionAI ({kelas}).<br>'
                            f'Silakan unggah gambar hewan yang sesuai dengan kategori yang tersedia.<br><br>'
                            f'Status: <b>Rejected</b> • Hasil klasifikasi: {label} • Confidence: {c*100:.1f}% (minimum {cnn_thr*100:.0f}%)<br>'
                            f'Gambar tidak dapat diproses lebih lanjut.</div>', unsafe_allow_html=True)
            st.session_state.update(last_mode="detection", used_feature=True, last_conf=c, last_pred="di luar kelas (ditolak CNN)")
            return
        st.caption(f"✅ Tahap 1 (CNN) lolos: {label} ({c*100:.1f}%) → lanjut ke YOLOv8")
    with st.spinner("🚀 AI sedang mendeteksi objek... harap tunggu sebentar!"):
        bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])  # ultralytics membaca array sebagai BGR
        res = yolo_model.predict(bgr, conf=conf_thr, verbose=False)[0]
        result_rgb = res.plot()[:, :, ::-1]
    boxes = res.boxes
    with col2:
        if boxes is None or len(boxes) == 0:
            kelas = ", ".join(str(n) for n in yolo_model.names.values())
            st.error(f"🚫 Gambar tidak masuk ke dalam kelas ({kelas}). "
                     f"Tidak ada objek dengan confidence ≥ {conf_thr:.2f}.")
            st.session_state.update(last_mode="detection", used_feature=True, last_conf=None,
                                    last_pred="di luar kelas")
            return
        st.image(result_rgb, caption="📦 Hasil Deteksi Objek", width="stretch")
        st.download_button("⬇️ Unduh Gambar Hasil Deteksi", to_png_bytes(result_rgb), "hasil_deteksi.png", "image/png")
    rows = []
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        rows.append({"No.": i, "Kelas": res.names[int(b.cls[0])], "Confidence (%)": round(float(b.conf[0]) * 100, 2),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    df = pd.DataFrame(rows)
    confs = df["Confidence (%)"]
    st.success(f"✅ {len(df)} objek berhasil terdeteksi!")
    c = st.columns(4)
    for col, (lab, val) in zip(c, [("Total Objek", len(df)), ("Rata-rata Confidence", f"{confs.mean():.1f}%"),
                                   ("Tertinggi", f"{confs.max():.1f}%"), ("Terendah", f"{confs.min():.1f}%")]):
        col.markdown(f'<div class="metric-box">{lab}<span>{val}</span></div>', unsafe_allow_html=True)
    per = df["Kelas"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-title">Jumlah Objek per Kelas</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(px.bar(per, x="Kelas", y="Jumlah", color="Kelas", color_discrete_sequence=COLORS)),
                        width="stretch", config={"displayModeBar": False})
    with g2:
        st.markdown('<div class="section-title">Proporsi Kelas</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(px.pie(per, values="Jumlah", names="Kelas", hole=0.45, color_discrete_sequence=COLORS)),
                        width="stretch", config={"displayModeBar": False})
    st.subheader("📋 Rincian Hasil Deteksi")
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("> 💡 Koordinat dalam piksel (xyxy) pada gambar asli. Confidence menunjukkan keyakinan model, bukan akurasi sebenarnya.")
    st.session_state.update(last_mode="detection", used_feature=True, last_conf=float(confs.mean()),
                            last_pred=", ".join(f"{r.Kelas} ({r.Jumlah})" for r in per.itertuples()))


def tampil_warna(img: Image.Image):
    a = analyze_colors(img)
    st.markdown('<div class="section-title">🎨 Warna Dominan (K-Means, 5 cluster)</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">Analisis dilakukan pada <b>seluruh gambar</b>, bukan hanya objek terdeteksi. '
                'Ini deskripsi visual, bukan ciri khas suatu kelas hewan.</div>', unsafe_allow_html=True)
    cols = st.columns(len(a["palette"]))
    for col, p in zip(cols, a["palette"]):
        col.markdown(f'<div class="pal"><div class="sw" style="background:{p["hex"]}"></div>{p["hex"]}<br>{p["pct"]:.1f}%<br>{p["name"]}</div>',
                     unsafe_allow_html=True)
    x = list(range(0, 256, 8))
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-title">Histogram RGB</div>', unsafe_allow_html=True)
        fig = go.Figure([go.Scatter(x=x, y=a["hist"][c], name=c, line=dict(color=col)) for c, col in
                         zip("RGB", ["#e5484d", "#2fb344", "#3b82f6"])])
        fig.update_xaxes(title="Intensitas (0-255)"); fig.update_yaxes(title="Jumlah piksel")
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    with g2:
        st.markdown('<div class="section-title">Distribusi Kecerahan</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=x, y=a["bright"], marker_color="#4f63c8"))
        fig.update_xaxes(title="Kecerahan (0-255)"); fig.update_yaxes(title="Jumlah piksel")
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    st.write(f"Kecerahan rata-rata: **{a['mean_brightness']:.1f}** / 255")
    st.session_state.update(last_mode="color", used_feature=True, last_conf=None,
                            last_pred="Warna dominan " + ", ".join(p["hex"] for p in a["palette"][:3]))


def pick(k: str) -> None:
    st.session_state.update(sample=k, detect=False)


def start() -> None:
    st.session_state["detect"] = True


GAME_WIN_RATE = 0.3  # peluang 3 gambar cocok
GAME_WIN_IMG, GAME_LOSE_IMG = "assets/naga_menang.png", "assets/hantu_kalah.jpg"
HANTU_JS = r"""(function(){
var d=document,old=d.getElementById('hantu-overlay');if(old){old.remove();}
var ov=d.createElement('div');ov.id='hantu-overlay';
ov.style.cssText='position:fixed;left:0;top:0;width:100vw;height:100vh;z-index:2147483647;overflow:hidden;opacity:1;transition:opacity 1.2s ease-out';
var im=d.createElement('img');im.src=__IMG__;
im.style.cssText='position:absolute;left:0;top:0;width:100%;height:100%;object-fit:cover;transform:scale(0.05);transform-origin:50% 50%;transition:transform 0.7s cubic-bezier(.1,.9,.2,1)';
im.style.objectFit=__FIT__;im.style.background=__BG__;ov.appendChild(im);d.body.appendChild(ov);
requestAnimationFrame(function(){requestAnimationFrame(function(){im.style.transform='scale(1)';
setTimeout(function(){im.style.transition='transform 20s linear';im.style.transform='scale(1.15)';},800);});});
var done=false;function fade(){if(done){return;}done=true;ov.style.opacity='0';setTimeout(function(){ov.remove();},1300);}
var a=new Audio(__SND__);a.onended=fade;a.onerror=function(){setTimeout(fade,4000);};
var pr=a.play();if(pr&&pr.catch){pr.catch(function(){setTimeout(fade,4000);});}
setTimeout(fade,20000);
})();"""


@st.cache_data
def data_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def hantu_fullscreen(img_path: str, snd: str, fit: str = "cover", bg: str = "transparent") -> None:
    """Gambar zoom dari tengah hingga memenuhi layar, suara otomatis, fade-out saat suara selesai."""
    import streamlit.components.v1 as components
    js = HANTU_JS.replace("__IMG__", json.dumps(data_uri(img_path))).replace("__SND__", json.dumps(snd)).replace("__FIT__", json.dumps(fit)).replace("__BG__", json.dumps(bg))
    loader = ("<script>var P=window.parent;var s=P.document.createElement('script');"
              f"s.textContent={json.dumps(js)};P.document.head.appendChild(s);</script><!--{time.time()}-->")
    components.html(loader, height=0)
SND_WIN = "assets/mahjong.mp3" if os.path.exists("assets/mahjong.mp3") else "https://www.myinstants.com/media/sounds/mahjong_23mNzE1.mp3"
SND_LOSE = "assets/horror_scream.mp3" if os.path.exists("assets/horror_scream.mp3") else "https://www.myinstants.com/media/sounds/horror-scream-high-quality.mp3"


def tampil_game():
    """Roulette 3 gambar dari data train. 3 sama = menang; 2x gagal berturut-turut = hantu kedua."""
    pool = {d: sorted(glob.glob(f"game_images/{d}/*.jpg")) for d in sorted(os.listdir("game_images"))} if os.path.isdir("game_images") else {}
    pool = {k: v for k, v in pool.items() if v}
    st.markdown('<div class="section-title">🎰 MINIGAME ROULETTE</div>', unsafe_allow_html=True)
    if not pool:
        st.error("Folder game_images/ kosong atau tidak ditemukan.")
        return
    st.markdown('<div class="explain-box">Tekan <b>PUTAR</b>. Jika 3 gambar hewan cocok, kamu menang. Jika 2 kali percobaan berturut-turut gagal, '
                'sesuatu akan muncul. ⚠️ <b>Peringatan:</b> ada gambar seram dan suara keras — kecilkan volume dulu.</div>', unsafe_allow_html=True)
    st.session_state.setdefault("game_fail", 0)
    st.caption(f"Gagal berturut-turut: {st.session_state['game_fail']} / 2")
    ph = [c.empty() for c in st.columns(3)]
    for x, k in zip(ph, list(pool)[:3]):
        x.image(pool[k][0], caption="?", width="stretch")
    if not st.button("🎰 PUTAR", key="spin", width="stretch"):
        return
    allimg = [f for v in pool.values() for f in v]
    for _ in range(10):  # animasi putaran
        for x in ph:
            x.image(random.choice(allimg), width="stretch")
        time.sleep(0.12)
    names = list(pool)
    if random.random() < GAME_WIN_RATE:  # sesuai GAME_WIN_RATE: 3 gambar sama
        kinds = [random.choice(names)] * 3
    else:
        kinds = random.choices(names, k=3)
        while len(set(kinds)) == 1:
            kinds = random.choices(names, k=3)
    for x, k in zip(ph, kinds):
        x.image(random.choice(pool[k]), caption=k.upper(), width="stretch")
    if len(set(kinds)) == 1:
        st.session_state["game_fail"] = 0
        st.success(f"🎉 3 gambar cocok: {kinds[0].upper()}!")
        hantu_fullscreen(GAME_WIN_IMG, SND_WIN, "contain", "#ffffff")
    else:
        st.session_state["game_fail"] += 1
        if st.session_state["game_fail"] >= 2:
            st.session_state["game_fail"] = 0
            st.error("💀 Dua kali gagal...")
            hantu_fullscreen(GAME_LOSE_IMG, SND_LOSE)
        else:
            st.warning("Belum cocok. Coba sekali lagi...")


def halaman_main():
    menu = st.sidebar.radio("🧭 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Analisis Warna Gambar", "Minigame Roulette"])
    conf_thr = CONF_THR
    cnn_thr = CNN_THR
    navbar()
    st.markdown(f'<div class="explain-box">Halo <b>{st.session_state["user_name"]}</b> dari <b>{st.session_state["user_campus"]}</b> '
                '— pilih sampel gambar atau unggah gambarmu, lalu tekan tombol mulai.</div>', unsafe_allow_html=True)
    if menu.startswith("Minigame"):
        tampil_game()
        return
    left, right = st.columns([1, 3])
    with left:
        st.markdown('<div class="section-title">SAMPEL GAMBAR</div>', unsafe_allow_html=True)
        for k, pth in SAMPLES.items():
            a, b = st.columns([1, 1.2])
            a.image(pth, width="stretch")
            b.button(k.capitalize(), key=f"s_{k}", on_click=pick, args=(k,))
    img = None
    with right:
        st.markdown('<div class="section-title">UNGGAH GAMBAR</div>', unsafe_allow_html=True)
        up = st.file_uploader("Seret gambar ke sini atau pilih file (JPG • JPEG • PNG • WEBP)", type=["jpg", "jpeg", "png", "webp"])
        sample = st.session_state["sample"]
        src = up if up is not None else (SAMPLES.get(sample) if sample else None)
        sid = getattr(up, "file_id", up.name) if up is not None else sample
        if sid != st.session_state["src_id"]:
            st.session_state.update(src_id=sid, detect=False)
        if src is None:
            st.info("👆 Pilih sampel di kiri atau unggah gambar terlebih dahulu.")
        else:
            try:
                img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                st.image(img, caption=f"🖼️ Pratinjau ({img.width}×{img.height} px)", width="stretch")
                st.button("MULAI DETEKSI  →" if menu.startswith("Deteksi") else "MULAI ANALISIS  →", on_click=start, width="stretch")
            except Exception:
                st.error("File bukan gambar yang valid.")
    if img is not None and st.session_state["detect"]:
        if menu.startswith("Deteksi"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="🖼️ Gambar Asli", width="stretch")
            tampil_deteksi(img, conf_thr, col1, col2, cnn_thr)
        else:
            tampil_warna(img)
        if st.session_state["used_feature"]:
            st.button("Berikan Feedback", on_click=go_to, args=("feedback",))
    st.button("⬅️ Kembali ke Registrasi", on_click=go_to, args=("register",))


def halaman_feedback():
    st.markdown('<div class="main-title">💬 Feedback Pengguna</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Terima kasih telah menggunakan aplikasi — beri masukanmu di bawah ini</div>', unsafe_allow_html=True)
    with st.form("form_feedback"):
        rating = st.slider("Rating (1 - 5)", 1, 5, 5)
        comment = st.text_area("Komentar / Saran", height=120)
        if st.form_submit_button("Kirim Feedback"):
            row = {"timestamp": datetime.datetime.now().isoformat(), "user_name": st.session_state["user_name"],
                   "user_campus": st.session_state["user_campus"], "mode": st.session_state["last_mode"] or "",
                   "prediction": st.session_state["last_pred"] or "", "confidence": st.session_state["last_conf"] or "",
                   "rating": rating, "comment": comment}
            try:
                new = not os.path.exists("feedbacks.csv")
                with open("feedbacks.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                st.success("✅ Feedback berhasil dikirim — terima kasih!")
                st.session_state["used_feature"] = False
            except Exception as e:
                st.error(f"Gagal menyimpan feedback: {e}")
    st.button("⬅️ Kembali ke Dashboard", on_click=go_to, args=("main",))


# ==========================
# ROUTING
# ==========================
{"home": halaman_awal, "register": halaman_registrasi, "main": halaman_main,
 "feedback": halaman_feedback}.get(st.session_state["page"], halaman_awal)()

st.markdown(f'<div class="footer"><img src="data:image/jpeg;base64,{b64("model/LOGO UUSK.jpg")}" alt="Logo USK"><div>created by <b style="color:#e6f1ff">Siddiqia Almuntahza | 2308108010004</b><br>Proyek UAS Praktikum Pemrograman Big Data<br>Universitas Syiah Kuala</div></div>', unsafe_allow_html=True)
