# =====================================================
# FINAL PROJECT LINEAR ALGEBRA — GROUP 7
# Matrix Image Lab (Bilingual EN–ID)
# =====================================================
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# PAGE CONFIG
st.set_page_config(
    page_title="Matrix Image Lab",
    page_icon="🧮",
    layout="wide"
)
# LANGUAGE SETTING & TEXT DICTIONARY
if "lang" not in st.session_state:
    st.session_state.lang = "EN"
if "composite_H" not in st.session_state:
    st.session_state.composite_H = np.eye(3, dtype=np.float64)

# Sidebar: Pilihan Bahasa
lang_choice = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Indonesia"])
st.session_state.lang = "EN" if lang_choice == "English" else "ID"

TEXT = {
    "EN": {
        "title": "Matrix Image Lab — 2D Transforms & Filters",
        "pages": "Pages",
        "home": "Home / Introduction",
        "tools": "Image Processing Tools",
        "team": "Team Members",
        "upload": "Upload an image (jpg/png). Large images may be slow.",
        "upload_info": "Upload an image to begin. Meanwhile you can read the left sidebar to prepare parameters.",
        "original": "Original",
        "result": "Transformed / Filtered",
        "download": "Download Result (PNG)",
        "matrix": "Applied 3×3 Homography (H)",
        "team_title": "Team Members",
        "role": "Role",
        "home_what_title": "*What this app does*",
        "home_what_1": "Apply 2D transformations (Translation, Scaling, Rotation, Shearing, Reflection) using 3×3 homogeneous matrices.",
        "home_what_2": "Apply convolution-based image filters (Blur, Sharpen) using custom kernels.",
        "home_what_3": "Optional: simple background removal (HSV threshold or GrabCut).",
        "home_visual_ex": "*Visual examples*",
        "home_demo_caption": "Rot+Scale+Trans (demo)",
        "transform_controls": "Image Transform Controls",
        "transform_mode": "Transformation mode",
        "single_t": "Single transform",
        "composite_b": "Composite builder",
        "t_choice": "Transform",
        "t_list": ["Translation", "Scaling", "Rotation", "Shearing", "Reflection", "None"],
        "t_tx": "tx (px)", "t_ty": "ty (px)", "t_sx": "sx", "t_sy": "sy", 
        "t_ang": "Angle (deg)", "t_shx": "shx", "t_shy": "shy", "t_axis": "Axis",
        "t_axis_v": "vertical", "t_axis_h": "horizontal",
        "c_build_info": "Build composite (left-multiply to apply new transform after existing).",
        "c_add": "Add transform:",
        "c_multiply": "Left-multiply add",
        "c_reset": "Reset composite",
        "c_current": "Current composite matrix:",
        "canvas_title": "Output canvas",
        "keep_size": "Keep original size",
        "out_w": "out width (px)", "out_h": "out height (px)",
        "filter_title": "Filters (convolution)",
        "apply_blur": "Blur",
        "blur_size": "Blur kernel size",
        "apply_sharp": "Sharpen",
        "bkg_title": "Background removal (optional)",
        "do_bkg": "Enable background removal (Background becomes TRANSPARENT)", 
        "bkg_method": "Method",
        "hsv_thresh": "HSV threshold",
        "grabcut": "GrabCut",
        "preview_title": "Preview & Controls",
        "preview_desc": "Left: original image. Right: current pipeline result.",
        "hsv_info": "HSV thresholds (use slider to find *background* color range to remove).",
        "grabcut_info": "Adjust iter count if needed. Initial rectangle covers 90% of image.",
        "grabcut_iter": "GrabCut iterations",
        "bkg_mask_preview": "Mask Preview (White = Area Removed)",
    },
    "ID": {
        "title": "Matrix Image Lab — Transformasi 2D & Filter",
        "pages": "Halaman",
        "home": "Beranda / Pengantar",
        "tools": "Alat Pengolahan Gambar",
        "team": "Anggota Tim",
        "upload": "Unggah gambar (jpg/png). Gambar besar mungkin lambat.",
        "upload_info": "Unggah gambar untuk memulai. Sementara itu Anda dapat membaca sidebar kiri untuk menyiapkan parameter.",
        "original": "Asli",
        "result": "Hasil Transformasi / Filter",
        "download": "Unduh Hasil (PNG)",
        "matrix": "Matriks Homografi 3×3 (H) yang Diterapkan",
        "team_title": "Anggota Tim",
        "role": "Peran",
        "home_what_title": "*Apa yang dilakukan aplikasi ini*",
        "home_what_1": "Menerapkan transformasi 2D (Translasi, Penskalaan, Rotasi, Geser, Refleksi) menggunakan matriks homogen 3×3.",
        "home_what_2": "Menerapkan filter gambar berbasis konvolusi (Blur, Pertajam) menggunakan kernel kustom.",
        "home_what_3": "Opsional: Penghapusan latar belakang sederhana (threshold HSV atau GrabCut).",
        "home_visual_ex": "*Contoh visual*",
        "home_demo_caption": "Rot+Skala+Trans (demo)",
        "transform_controls": "Kontrol Transformasi Gambar",
        "transform_mode": "Mode Transformasi",
        "single_t": "Transformasi Tunggal",
        "composite_b": "Pembangun Komposit",
        "t_choice": "Transformasi",
        "t_list": ["Translasi", "Penskalaan", "Rotasi", "Geser", "Refleksi", "Tidak Ada"],
        "t_tx": "tx (piksel)", "t_ty": "ty (piksel)", "t_sx": "sx", "t_sy": "sy", 
        "t_ang": "Sudut (derajat)", "t_shx": "shx", "t_shy": "shy", "t_axis": "Sumbu",
        "t_axis_v": "vertikal", "t_axis_h": "horizontal",
        "c_build_info": "Bangun komposit (kalikan kiri untuk menerapkan transformasi baru setelah yang ada).",
        "c_add": "Tambahkan transformasi:",
        "c_multiply": "Kalikan Kiri Tambahkan",
        "c_reset": "Setel Ulang Komposit",
        "c_current": "Matriks komposit saat ini:",
        "canvas_title": "Kanvas Output",
        "keep_size": "Pertahankan ukuran asli",
        "out_w": "lebar output (piksel)", "out_h": "tinggi output (piksel)",
        "filter_title": "Filter (konvolusi)",
        "apply_blur": "Blur",
        "blur_size": "Ukuran kernel blur",
        "apply_sharp": "Pertajam",
        "bkg_title": "Penghapusan latar belakang (opsional)",
        "do_bkg": "Aktifkan penghapusan latar belakang (Latar Belakang menjadi TRANSPARAN)", 
        "bkg_method": "Metode",
        "hsv_thresh": "Threshold HSV",
        "grabcut": "GrabCut",
        "preview_title": "Pratinjau & Kontrol",
        "preview_desc": "Kiri: gambar asli. Kanan: hasil pipeline saat ini.",
        "hsv_info": "Threshold HSV (gunakan slider untuk menemukan rentang warna *latar belakang* yang akan dihapus).",
        "grabcut_info": "Sesuaikan hitungan iterasi jika diperlukan. Persegi panjang awal mencakup 90% gambar.",
        "grabcut_iter": "Iterasi GrabCut",
        "bkg_mask_preview": "Pratinjau Mask (Putih = Area Dihapus)",
    }
}
T = TEXT[st.session_state.lang]

# MATRIX & IMAGE HELPERS 
# --- Transformasi Homogen ---
def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
def scaling(sx, sy, center=(0,0)):
    cx, cy = center
    T1 = translation(-cx, -cy)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ S @ T1
def rotation(angle_deg, center=(0,0)):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    cx, cy = center
    T1 = translation(-cx, -cy)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ R @ T1
def shear(shx, shy, center=(0,0)):
    cx, cy = center
    T1 = translation(-cx, -cy)
    Sh = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float64)
    T2 = translation(cx, cy)
    return T2 @ Sh @ T1
def reflection(axis, axis_pos=None, img_shape=None):
    if img_shape is not None and axis_pos is None:
        h, w = img_shape[:2]
        axis_pos = w/2 if axis=='vertical' else h/2
    if axis == 'vertical':
        T1 = translation(-axis_pos, 0)
        R = np.array([[-1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
        T2 = translation(axis_pos, 0)
        return T2 @ R @ T1
    else: 
        T1 = translation(0, -axis_pos)
        R = np.array([[1,0,0], [0,-1,0], [0,0,1]], dtype=np.float64)
        T2 = translation(0, axis_pos)
        return T2 @ R @ T1

# --- Image IO & Homography Application ---
def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file).convert("RGB"))
def pil_from_cv(img_cv):
    if img_cv.shape[2] == 4:
        return Image.fromarray(np.clip(img_cv,0,255).astype(np.uint8), mode='RGBA')
    return Image.fromarray(np.clip(img_cv,0,255).astype(np.uint8))
def apply_homography_to_image(img_rgb, H, dst_size=None, border_mode=cv2.BORDER_CONSTANT):
    h, w = img_rgb.shape[:2]
    dst_w, dst_h = dst_size if dst_size is not None else (w, h)
    H_cv = H.astype(np.float64)
    transformed = cv2.warpPerspective(img_rgb, H_cv, (dst_w, dst_h), 
                                      flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(255,255,255))
    return transformed
    
# --- Filter Helpers ---
def convolve(img, kernel):
    result = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        result[:,:,c] = cv2.filter2D(img[:,:,c].astype(np.float32), -1, kernel)
    return np.clip(result, 0, 255)

def blur_kernel(size=3):
    k = np.ones((size,size), dtype=np.float32)
    return k / k.sum()
def sharpen_kernel():
    return np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

# --- Background Removal Helpers (DIUBAH KE RGBA/Transparan) ---
def remove_background_hsv(img_rgb, lower, upper):
    """Menghapus piksel di area HSV yang ditentukan (Background) dan menggantinya dengan Transparan."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_bkg = cv2.inRange(hsv, lower, upper)
    
    alpha = np.full((img_rgb.shape[0], img_rgb.shape[1], 1), 255, dtype=np.uint8)
    out_rgba = np.concatenate((img_rgb, alpha), axis=2)
    out_rgba[mask_bkg.astype(bool), 3] = 0
    return mask_bkg, out_rgba.astype(np.uint8)
    
def grabcut_bkg_removal(img_rgb, rect=None, iter_count=5):
    """Menghapus latar belakang menggunakan GrabCut, mempertahankan Foreground, dan mengubah Background menjadi Transparan."""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    if rect is None:
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype('uint8') 
    alpha = mask_fg * 255 
    alpha = alpha[:, :, np.newaxis]
    out_rgba = np.concatenate((img_rgb, alpha), axis=2) 
    bkg_mask_preview = (1-mask_fg)*255
    return bkg_mask_preview.astype(np.uint8), out_rgba.astype(np.uint8)

# Common small helper to display two images
def show_side_by_side(orig, transformed):
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_from_cv(orig), caption=T["original"], use_column_width=True)
    with col2:
        st.image(pil_from_cv(transformed), caption=T["result"], use_column_width=True)

# NAVIGATION

pages_key = ["home", "tools", "team"]
page = st.sidebar.radio(
    T["pages"],
    pages_key,
    format_func=lambda x: T[x]
)

# HOME
if page == "home":
    st.title(T["title"])
    st.markdown(T["home_what_title"])
    st.markdown(f"* {T['home_what_1']}")
    st.markdown(f"* {T['home_what_2']}")
    st.markdown(f"* {T['home_what_3']}")
    st.markdown(T["home_visual_ex"])

    demo_img = np.zeros((220,220,3), dtype=np.uint8)
    demo_img[60:160,40:180] = [200,60,60]
    M_demo = rotation(30, center=(110,110)) @ scaling(0.7, 0.7, center=(110,110)) @ translation(20,-10)
    demo_trans = apply_homography_to_image(demo_img, M_demo, dst_size=(220,220))
    colA, colB = st.columns(2)
    with colA:
        st.image(pil_from_cv(demo_img), caption=T["original"] + " (demo)", use_column_width=True)
    with colB:
        st.image(pil_from_cv(demo_trans), caption=T["home_demo_caption"], use_column_width=True)

# TOOLS
elif page == "tools":
    st.title(T["tools"])
    st.markdown(T["preview_desc"])

    uploaded = st.file_uploader(T["upload"], type=["jpg","jpeg","png"])
    
    if uploaded is None:
        st.info(T["upload_info"])
        st.stop()
    
    img = load_image(uploaded) 
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header(T["transform_controls"])
    transform_mode = st.sidebar.radio(T["transform_mode"], [T["single_t"], T["composite_b"]])
    
    H = np.eye(3, dtype=np.float64)
    h, w = img.shape[:2]
    center = (w/2, h/2)
    
    # Transform Logic
    if transform_mode == T["single_t"]:
        tchoice = st.sidebar.selectbox(T["t_choice"], T["t_list"])
        if tchoice == T["t_list"][0]: # Translasi
            tx = st.sidebar.number_input(T["t_tx"], value=0.0, format="%.1f", key="t_tx")
            ty = st.sidebar.number_input(T["t_ty"], value=0.0, format="%.1f", key="t_ty")
            H = translation(tx, ty)
        elif tchoice == T["t_list"][1]: # Penskalaan
            sx = st.sidebar.number_input(T["t_sx"], value=1.0, format="%.3f", key="t_sx")
            sy = st.sidebar.number_input(T["t_sy"], value=1.0, format="%.3f", key="t_sy")
            H = scaling(sx, sy, center=center)
        elif tchoice == T["t_list"][2]: # Rotasi
            ang = st.sidebar.slider(T["t_ang"], -180.0, 180.0, 0.0, key="t_ang")
            H = rotation(ang, center=center)
        elif tchoice == T["t_list"][3]: # Geser
            shx = st.sidebar.number_input(T["t_shx"], value=0.0, format="%.3f", key="t_shx")
            shy = st.sidebar.number_input("shy", value=0.0, format="%.3f", key="t_shy")
            H = shear(shx, shy, center=center)
        elif tchoice == T["t_list"][4]: # Refleksi
            axis_opts = [T["t_axis_v"], T["t_axis_h"]]
            axis = st.sidebar.radio(T["t_axis"], axis_opts, key="t_ref_axis")
            H = reflection(axis='vertical' if axis==T["t_axis_v"] else 'horizontal', img_shape=img.shape)
        else:
            H = np.eye(3)
    else:
        st.sidebar.markdown(T["c_build_info"])
        ch_list = [T["t_list"][i] for i in range(5)]
        ch = st.sidebar.selectbox(T["c_add"], ch_list)
        M_new = np.eye(3)
        if ch == T["t_list"][0]: # Translasi
            tx = st.sidebar.number_input(T["t_tx"], value=0.0, key="c_tx")
            ty = st.sidebar.number_input(T["t_ty"], value=0.0, key="c_ty")
            M_new = translation(tx,ty)
        elif ch == T["t_list"][1]: # Penskalaan
            sx = st.sidebar.number_input(T["t_sx"], value=1.0, key="c_sx")
            sy = st.sidebar.number_input(T["t_sy"], value=1.0, key="c_sy")
            M_new = scaling(sx, sy, center=center)
        elif ch == T["t_list"][2]: # Rotasi
            ang = st.sidebar.slider(T["t_ang"], -180.0, 180.0, 0.0, key="c_ang")
            M_new = rotation(ang, center=center)
        elif ch == T["t_list"][3]: # Geser
            shx = st.sidebar.number_input(T["t_shx"], value=0.0, key="c_shx")
            shy = st.sidebar.number_input(T["t_shy"], value=0.0, key="c_shy")
            M_new = shear(shx, shy, center=center)
        else: # Refleksi
            axis_opts = [T["t_axis_v"], T["t_axis_h"]]
            axis = st.sidebar.radio(T["t_axis"], axis_opts, key="c_ref_axis")
            M_new = reflection(axis='vertical' if axis==T["t_axis_v"] else 'horizontal', img_shape=img.shape)
        if st.sidebar.button(T["c_multiply"]):
            st.session_state.composite_H = M_new @ st.session_state.composite_H
        if st.sidebar.button(T["c_reset"]):
            st.session_state.composite_H = np.eye(3, dtype=np.float64)
        st.sidebar.markdown(T["c_current"])
        st.sidebar.dataframe(st.session_state.composite_H, use_container_width=True)
        H = st.session_state.composite_H

    # Target Canvas Size
    st.sidebar.header(T["canvas_title"])
    keep_size = st.sidebar.checkbox(T["keep_size"], value=True)
    if keep_size:
        dst_size = (img.shape[1], img.shape[0])
    else:
        out_w = st.sidebar.number_input(T["out_w"], value=img.shape[1], step=1)
        out_h = st.sidebar.number_input(T["out_h"], value=img.shape[0], step=1)
        dst_size = (int(out_w), int(out_h))
        
    # Filters
    st.sidebar.header(T["filter_title"])
    apply_blur = st.sidebar.checkbox(T["apply_blur"], value=False)
    blur_size = st.sidebar.selectbox(T["blur_size"], [3,5,7], index=0) if apply_blur else 3
    apply_sharp = st.sidebar.checkbox(T["apply_sharp"], value=False)
    
    # Background removal
    st.sidebar.header(T["bkg_title"])
    do_bkg = st.sidebar.checkbox(T["do_bkg"], value=False)
    bkg_method = st.sidebar.selectbox(T["bkg_method"], [T["hsv_thresh"], T["grabcut"]]) if do_bkg else None

    # --- IMAGE PROCESSING PIPELINE ---
    
    transformed_img = apply_homography_to_image(img, H, dst_size=dst_size, border_mode=cv2.BORDER_CONSTANT) 

    bkg_mask = None
    bkg_removed = transformed_img.astype(np.uint8) # Default: RGB
    
    if do_bkg:
        if bkg_method == T["hsv_thresh"]:
            st.sidebar.markdown(T["hsv_info"])
            h_min = st.sidebar.slider("H min", 0, 179, 0)
            h_max = st.sidebar.slider("H max", 0, 179, 179)
            s_min = st.sidebar.slider("S min", 0, 255, 0)
            s_max = st.sidebar.slider("S max", 0, 255, 255)
            v_min = st.sidebar.slider("V min", 0, 255, 0)
            v_max = st.sidebar.slider("V max", 0, 255, 255)
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask, bkg_removed = remove_background_hsv(transformed_img.astype(np.uint8), lower, upper)
            bkg_mask = mask
        else: 
            st.sidebar.markdown(T["grabcut_info"])
            iter_count = st.sidebar.slider(T["grabcut_iter"], 1, 10, 5)
            rect = (int(0.05*dst_size[0]), int(0.05*dst_size[1]),
                    int(0.9*dst_size[0]), int(0.9*dst_size[1]))
            mask, bkg_removed = grabcut_bkg_removal(transformed_img.astype(np.uint8), rect=rect, iter_count=iter_count)
            bkg_mask = mask

    if bkg_removed.shape[2] == 4: 
        rgb_part = bkg_removed[:,:,:3].astype(np.float32)
        alpha_part = bkg_removed[:,:,3:].astype(np.float32)
        if apply_blur:
            k = blur_kernel(blur_size)
            rgb_part = convolve(rgb_part, k)
        if apply_sharp:
            s = sharpen_kernel()
            rgb_part = convolve(rgb_part, s)
        final_img = np.concatenate((rgb_part.astype(np.uint8), alpha_part.astype(np.uint8)), axis=2)
    else:
        final_img = bkg_removed.copy().astype(np.float32)
        if apply_blur:
            k = blur_kernel(blur_size)
            final_img = convolve(final_img, k)
        if apply_sharp:
            s = sharpen_kernel()
            final_img = convolve(final_img, s)
        final_img = final_img.astype(np.uint8)
        
    # --- OUTPUT DISPLAY ---
    st.subheader(T["preview_title"])
    show_side_by_side(img, final_img)
    
    st.subheader(T["matrix"].split('(')[0].strip())
    st.markdown(T["matrix"])
    st.dataframe(H, use_container_width=True)

    # Download Button 
    result_pil = pil_from_cv(final_img)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG") 
    byte_im = buf.getvalue()
    st.download_button(T["download"], data=byte_im, file_name="result_transparent.png", mime="image/png")

    if bkg_mask is not None:
        st.markdown("---")
        st.subheader(T["bkg_mask_preview"])
        st.image(pil_from_cv(bkg_mask), use_column_width=False, width=400)

# TEAM MEMBERS PAGE
elif page == "team":
    st.title(T["team_title"])
    team_data = [
        {"Nama": "Dyastiti Eka Marlinda", T['role']: "Project Manager, System Architect & Back-End Engineer", "image": "assets/dyas.jpeg"},
        {"Nama": "Lovyta Amelia", T['role']: "UI/UX specialist & Front-End Developer", "image": "assets/amel.jpeg"},
        {"Nama": "Mutiara Rahemi Putri", T['role']: "Algorithm, Documentation & Presentation Lead", "image": "assets/muti.jpeg"},
        {"Nama": "Putri Wulan Sari", T['role']: "Image Processing, Testing & Debugging", "image": "assets/putri.jpeg"}
    ]
    cols = st.columns(2)
    
    for i, member in enumerate(team_data):
        with cols[i % 2]: 
            try:
                img_pil = Image.open(member["image"])
                st.markdown(f"### {member['Nama']}")
                st.image(img_pil, caption=f"{T['role']}:** {member[T['role']]}", width=200)
                st.markdown("---") 
            except FileNotFoundError:
                st.error(f"File gambar untuk {member['Nama']} tidak ditemukan di {member['image']}. Silakan cek foldernya.")
                st.markdown(f"### {member['Nama']}")
                st.markdown(f"{T['role']}:** {member[T['role']]}")
                st.markdown("---")
