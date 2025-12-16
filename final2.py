# =====================================================
# FINAL PROJECT LINEAR ALGEBRA
# Dyastiti – Group 7
# =====================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Matrix Image Lab",
    layout="wide"
)

# =====================================================
# LANGUAGE SELECTOR
# =====================================================
language = st.sidebar.selectbox(
    "Language / Bahasa",
    ["English", "Bahasa Indonesia"]
)

# =====================================================
# TEXT DICTIONARY (SINGLE SOURCE OF TRUTH)
# =====================================================
TEXT = {
    "English": {
        "home": "Home / Introduction",
        "tools": "Image Processing Tools",
        "team": "Team Members",

        "home_title": "Matrix Image Lab — 2D Transforms & Filters",
        "home_desc": "This application demonstrates linear algebra concepts through image transformations.",

        "what": "What this app does",
        "notes": "Notes",
        "visual": "Visual examples",

        "what_1": "Apply 2D transformations using 3×3 homogeneous matrices",
        "what_2": "Apply convolution-based image filters (Blur, Sharpen)",
        "what_3": "Optional background removal (HSV threshold or GrabCut)",
        "what_4": "Multi-page UI: Home, Tools, Team Members",

        "notes_1": "Transformations are constructed manually as 3×3 matrices",
        "notes_2": "Filters are implemented using explicit convolution loops",

        "upload_desc": "Upload an image, select transformations and filters, preview and download the result.",

        "transform": "Transform",
        "single": "Single transform",

        "translation": "Translation",
        "scaling": "Scaling",
        "rotation": "Rotation",
        "shearing": "Shearing",
        "reflection": "Reflection",
        "none": "None",

        "filters": "Filters (convolution)",
        "blur": "Blur (manual convolution)",
        "sharpen": "Sharpen (manual convolution)",

        "background": "Background removal (optional)",
        "enable_bkg": "Enable background removal",

        "preview": "Preview & Controls",
        "matrix": "Applied 3×3 homography (H)",
        "download": "Download result (PNG)",

        "name": "Name",
        "role": "Role"
    },

    "Bahasa Indonesia": {
        "home": "Beranda / Pendahuluan",
        "tools": "Alat Pengolahan Citra",
        "team": "Anggota Tim",

        "home_title": "Matrix Image Lab — Transformasi 2D & Filter",
        "home_desc": "Aplikasi ini mendemonstrasikan konsep aljabar linear melalui transformasi citra.",

        "what": "Fungsi Aplikasi",
        "notes": "Catatan",
        "visual": "Contoh Visual",

        "what_1": "Menerapkan transformasi 2D menggunakan matriks homogen 3×3",
        "what_2": "Menerapkan filter citra berbasis konvolusi (Blur, Sharpen)",
        "what_3": "Opsional penghapusan latar belakang (HSV atau GrabCut)",
        "what_4": "Antarmuka multi-halaman: Beranda, Tools, Tim",

        "notes_1": "Transformasi dibangun manual sebagai matriks 3×3",
        "notes_2": "Filter diimplementasikan dengan konvolusi manual",

        "upload_desc": "Unggah gambar, pilih transformasi dan filter, lalu unduh hasilnya.",

        "transform": "Transformasi",
        "single": "Transformasi tunggal",

        "translation": "Translasi",
        "scaling": "Skala",
        "rotation": "Rotasi",
        "shearing": "Shear",
        "reflection": "Refleksi",
        "none": "Tidak ada",

        "filters": "Filter (konvolusi)",
        "blur": "Blur (konvolusi manual)",
        "sharpen": "Penajaman (konvolusi manual)",

        "background": "Penghapusan latar (opsional)",
        "enable_bkg": "Aktifkan penghapusan latar",

        "preview": "Pratinjau & Kontrol",
        "matrix": "Matriks homografi 3×3",
        "download": "Unduh hasil (PNG)",

        "name": "Nama",
        "role": "Peran"
    }
}

# =====================================================
# TEAM MEMBERS DATA
# =====================================================
TEAM_MEMBERS = [
    {
        "name": "Dyastiti Eka Marlinda",
        "role": {
            "English": "Project Manager, System Architect & Back-End Engineer",
            "Bahasa Indonesia": "Manajer Proyek, Perancang Sistem, Pengembang Back-End"
        },
        "image": "assets/dyas.jpg"
    },
    {
        "name": "Lovyta Amelia",
        "role": {
            "English": "UI/UX specialist & Front-End Developer",
            "Bahasa Indonesia": "Pengembang UI/UX & Front-End"
        },
        "image": "assets/amel.jpg"
    },
    {
        "name": "Mutiara Rahemi Putri",
        "role": {
            "English": "Algorithm, Documentation & Presentation Lead",
            "Bahasa Indonesia": "Algoritma, Penanggung Jawab Dokumentasi & Presentasi"
        },
        "image": "assets/muti.jpg"
    },
    {
        "name": "Putri Wulan Sari",
        "role": {
            "English": "Image Processing, Testing & Debugging",
            "Bahasa Indonesia": "Pengolahan Gambar, Pengujian & Perbaikan Bug"
        },
        "image": "assets/putri.jpg"
    }
]

# =====================================================
# MATRIX FUNCTIONS
# =====================================================
def translation(tx, ty):
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]])

def scaling(sx, sy, center):
    cx, cy = center
    return translation(cx, cy) @ np.array([[sx,0,0],[0,sy,0],[0,0,1]]) @ translation(-cx, -cy)

def rotation(angle, center):
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    cx, cy = center
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return translation(cx, cy) @ R @ translation(-cx, -cy)

def shear(shx, shy, center):
    cx, cy = center
    Sh = np.array([[1,shx,0],[shy,1,0],[0,0,1]])
    return translation(cx, cy) @ Sh @ translation(-cx, -cy)

# =====================================================
# PAGE NAVIGATION
# =====================================================
pages = [TEXT[language]["home"], TEXT[language]["tools"], TEXT[language]["team"]]
page = st.sidebar.radio("Pages", pages)

# =====================================================
# HOME PAGE
# =====================================================
if page == TEXT[language]["home"]:
    st.title(TEXT[language]["home_title"])
    st.write(TEXT[language]["home_desc"])

    st.subheader(TEXT[language]["what"])
    st.markdown(f"""
    - {TEXT[language]["what_1"]}
    - {TEXT[language]["what_2"]}
    - {TEXT[language]["what_3"]}
    - {TEXT[language]["what_4"]}
    """)

    st.subheader(TEXT[language]["notes"])
    st.markdown(f"""
    - {TEXT[language]["notes_1"]}
    - {TEXT[language]["notes_2"]}
    """)

# =====================================================
# IMAGE PROCESSING TOOLS
# =====================================================
elif page == TEXT[language]["tools"]:

    st.title(TEXT[language]["tools"])
    st.write(TEXT[language]["upload_desc"])

    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if uploaded is None:
        st.stop()

    img = np.array(Image.open(uploaded).convert("RGB"))
    h, w = img.shape[:2]
    center = (w/2, h/2)

    st.image(img, caption=TEXT[language]["original"] if "original" in TEXT[language] else "", use_column_width=True)

    st.sidebar.header(TEXT[language]["transform"])
    tchoice = st.sidebar.selectbox(
        TEXT[language]["transform"],
        [
            TEXT[language]["translation"],
            TEXT[language]["scaling"],
            TEXT[language]["rotation"],
            TEXT[language]["shearing"],
            TEXT[language]["reflection"],
            TEXT[language]["none"]
        ]
    )
    H = np.eye(3)

    if tchoice == TEXT[language]["translation"]:
        tx = st.sidebar.number_input("tx", 0.0)
        ty = st.sidebar.number_input("ty", 0.0)
        H = translation(tx, ty)
    elif tchoice == TEXT[language]["scaling"]:
        sx = st.sidebar.number_input("sx", 1.0)
        sy = st.sidebar.number_input("sy", 1.0)
        H = scaling(sx, sy, center)
    elif tchoice == TEXT[language]["rotation"]:
        ang = st.sidebar.slider("angle", -180, 180, 0)
        H = rotation(ang, center)
    elif tchoice == TEXT[language]["shearing"]:
        shx = st.sidebar.number_input("shx", 0.0)
        shy = st.sidebar.number_input("shy", 0.0)
        H = shear(shx, shy, center)
    result = cv2.warpPerspective(img, H, (w, h))

    st.sidebar.header(TEXT[language]["filters"])
    if st.sidebar.checkbox(TEXT[language]["blur"]):
        result = cv2.GaussianBlur(result, (5,5), 0)
    if st.sidebar.checkbox(TEXT[language]["sharpen"]):
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        result = cv2.filter2D(result, -1, kernel)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Original", use_column_width=True)
    col2.image(result, caption="Result", use_column_width=True)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")

    st.download_button(
        TEXT[language]["download"],
        data=buf.getvalue(),
        file_name="result.png",
        mime="image/png"
    )

# =====================================================
# TEAM MEMBERS PAGE
# =====================================================
elif page == TEXT[language]["team"]:
    st.title(TEXT[language]["team"])

    for m in TEAM_MEMBERS:
        col1, col2 = st.columns([1,2])
        with col1:
            try:
                st.image(Image.open(m["image"]), width=150)
            except:
                st.image("https://via.placeholder.com/150")
        with col2:
            st.markdown(f"{TEXT[language]['name']}: {m['name']}")
            st.markdown(f"{TEXT[language]['role']}: {m['role'][language]}")
        st.divider()