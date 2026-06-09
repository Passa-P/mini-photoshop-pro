import sys
import os
import urllib.request
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QGroupBox, QScrollArea, QFrame, QComboBox,
    QInputDialog, QMessageBox, QProgressDialog, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# =====================================================================
# CONSTANTS
# =====================================================================
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

PROTOTXT_URL  = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/923b3128f25262b5010cef67e4fb9e4b6728ae7b/voc/MobileNetSSD_deploy.prototxt"
CAFFEMODEL_URL = "https://huggingface.co/spaces/Ibtehaj10/cheating-detection/resolve/main/MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH  = "MobileNetSSD_deploy.prototxt"
MODEL_PATH     = "MobileNetSSD_deploy.caffemodel"

GOLD     = (55, 175, 212)    # BGR — warna gold tema
GOLD_HEX = "#D4AF37"
BG_HEX   = "#FAFAFA"
WHITE    = "#FFFFFF"

STYLESHEET = f"""
    QMainWindow  {{ background-color: {BG_HEX}; }}
    QScrollArea  {{ border: none; background-color: {WHITE}; }}
    QWidget#sideContent {{ background-color: {BG_HEX}; }}
    QGroupBox {{
        color: #222222; border: 2px solid #E0D0B0; border-radius: 8px;
        margin-top: 16px; font-weight: bold; font-size: 12px;
        padding: 10px; background-color: {WHITE};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        padding: 2px 8px; background-color: {GOLD_HEX};
        color: white; border-radius: 4px;
    }}
    QPushButton {{
        background-color: {WHITE}; color: #444444;
        border: 1px solid {GOLD_HEX}; border-radius: 5px;
        padding: 7px; font-size: 11px; font-weight: 500;
    }}
    QPushButton:hover  {{ background-color: #F9F5EB; color: #C5A028; border: 1px solid #C5A028; }}
    QPushButton:pressed {{ background-color: #E6D5B3; }}
    QPushButton:disabled {{ background-color: #F0F0F0; color: #AAAAAA; border-color: #DDDDDD; }}
    QLabel {{ color: #555555; font-size: 11px; font-weight: 500; }}
    QComboBox {{
        background-color: {WHITE}; color: #444444;
        border: 1px solid {GOLD_HEX}; border-radius: 4px;
        padding: 5px; font-size: 11px;
    }}
    QComboBox::drop-down {{ border: none; }}
    QSlider::groove:horizontal {{
        border: 1px solid #E0E0E0; height: 6px;
        background: #F0F0F0; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {GOLD_HEX}; border: 1px solid #C5A028;
        width: 14px; margin: -4px 0; border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{ background: #C5A028; }}
"""

# =====================================================================
# PART 1: BACKGROUND THREAD — CNN DETECTION
# =====================================================================
class DetectionThread(QThread):
    """Menjalankan inferensi CNN di background agar UI tidak freeze."""
    finished  = pyqtSignal(object, int, str)   # (result_img, count, target)
    error     = pyqtSignal(str)
    progress  = pyqtSignal(str)

    def __init__(self, img, target_object):
        super().__init__()
        self.img           = img.copy()
        self.target_object = target_object

    def run(self):
        try:
            # 1. Download model jika belum ada
            if not os.path.exists(PROTOTXT_PATH):
                self.progress.emit("Mengunduh arsitektur model (.prototxt)...")
                urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)

            if not os.path.exists(MODEL_PATH):
                self.progress.emit("Mengunduh bobot model (.caffemodel) ~23MB, harap tunggu...")
                urllib.request.urlretrieve(CAFFEMODEL_URL, MODEL_PATH)

            # 2. Load model
            self.progress.emit("Memuat model CNN ke memori...")
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

            # 3. Inferensi
            self.progress.emit("Menjalankan deteksi objek...")
            img = self.img.copy()
            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)),
                0.007843, (300, 300), 127.5
            )
            net.setInput(blob)
            detections = net.forward()

            detected_count = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.45:
                    class_id   = int(detections[0, 0, i, 1])
                    if class_id >= len(CLASSES):
                        continue
                    label_name = CLASSES[class_id]

                    if label_name == self.target_object or self.target_object == "__all__":
                        detected_count += 1
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        startX = max(0, startX); startY = max(0, startY)
                        endX   = min(w - 1, endX); endY  = min(h - 1, endY)

                        # Gambar bounding box berwarna emas (BGR)
                        cv2.rectangle(img, (startX, startY), (endX, endY), GOLD, 2)

                        # Label dengan background gelap agar terbaca
                        text  = f"{label_name.upper()}: {confidence * 100:.1f}%"
                        (tw, th), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        label_y = max(startY - 6, th + 4)
                        cv2.rectangle(
                            img,
                            (startX, label_y - th - baseline - 4),
                            (startX + tw + 6, label_y + baseline - 2),
                            GOLD, -1
                        )
                        cv2.putText(
                            img, text,
                            (startX + 3, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA
                        )

            self.finished.emit(img, detected_count, self.target_object)

        except Exception as e:
            self.error.emit(str(e))


# =====================================================================
# PART 2: CUSTOM LABEL — ROI DRAG CROP
# =====================================================================
class CropableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.begin         = QPoint()
        self.end           = QPoint()
        self.is_drawing    = False
        self.crop_callback = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap():
            self.begin = self.end = event.pos()
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.update()
            if self.crop_callback and self.begin != self.end:
                self.crop_callback(self.begin, self.end)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_drawing:
            painter = QPainter(self)
            pen = QPen(Qt.darkYellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.begin, self.end).normalized())


# =====================================================================
# PART 3: MAIN APPLICATION
# =====================================================================
class MiniPhotoshopPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Photoshop Pro — Luxurious Edition")
        self.setGeometry(50, 50, 1500, 970)
        self.setStyleSheet(STYLESHEET)

        self.img_orig = None
        self.img_proc = None
        self._det_thread = None   # referensi thread deteksi

        self.initUI()

    # ------------------------------------------------------------------
    # UI LAYOUT
    # ------------------------------------------------------------------
    def initUI(self):
        main_widget = QWidget()
        root_layout = QHBoxLayout(main_widget)
        root_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.setSpacing(15)

        # ── SIDEBAR ──────────────────────────────────────────────────
        sidebar = QScrollArea()
        sidebar.setFixedWidth(345)
        sidebar.setWidgetResizable(True)

        side_content = QWidget()
        side_content.setObjectName("sideContent")
        side_layout = QVBoxLayout(side_content)
        side_layout.setContentsMargins(5, 5, 5, 5)
        side_layout.setSpacing(10)

        side_layout.addWidget(self._build_group_file())
        side_layout.addWidget(self._build_group_cnn())
        # TAMBAHAN BAGIAN A
        side_layout.addWidget(self._build_group_classifier())
        side_layout.addWidget(self._build_group_enhancement())
        side_layout.addWidget(self._build_group_color())
        side_layout.addWidget(self._build_group_segmentation())
        side_layout.addWidget(self._build_group_geometry())
        side_layout.addStretch()

        sidebar.setWidget(side_content)

        # ── DISPLAY AREA ─────────────────────────────────────────────
        display_layout = QVBoxLayout()
        display_layout.setSpacing(12)

        img_panel = QHBoxLayout()
        img_panel.setSpacing(12)

        self.lbl_orig = QLabel("Original Canvas")
        self.lbl_proc = CropableLabel("Processed Canvas  (drag to crop)")
        self.lbl_proc.crop_callback = self.apply_crop

        canvas_style = (
            "border: 1px solid #E0D0B0; border-radius: 6px;"
            "background-color: #FFFFFF; color: #999999; font-weight: bold;"
        )
        for lbl in (self.lbl_orig, self.lbl_proc):
            lbl.setFrameShape(QFrame.StyledPanel)
            lbl.setStyleSheet(canvas_style)
            lbl.setMinimumSize(480, 480)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        img_panel.addWidget(self.lbl_orig)
        img_panel.addWidget(self.lbl_proc)

        # Status bar kecil
        self.lbl_status = QLabel("Siap. Buka gambar untuk memulai.")
        self.lbl_status.setStyleSheet(
            "color: #888888; font-size: 10px; padding: 2px 4px;"
        )

        # Histogram
        self.fig, self.ax = plt.subplots(figsize=(7, 2.2), dpi=95)
        self.fig.patch.set_facecolor(WHITE)
        self.canvas_hist = FigureCanvas(self.fig)
        self.canvas_hist.setStyleSheet(
            "border: 1px solid #E0D0B0; border-radius: 6px; background-color: #FFFFFF;"
        )
        self.canvas_hist.setFixedHeight(190)

        display_layout.addLayout(img_panel, stretch=1)
        display_layout.addWidget(self.lbl_status)
        display_layout.addWidget(self.canvas_hist)

        root_layout.addWidget(sidebar)
        root_layout.addLayout(display_layout)
        self.setCentralWidget(main_widget)

    # ── GROUP BUILDERS ────────────────────────────────────────────────
    def _build_group_file(self):
        gp = QGroupBox("File & Compression")
        ly = QVBoxLayout()

        btn_load  = QPushButton("📂  Open Image")
        btn_save  = QPushButton("💾  Save Image (JPEG)")
        btn_reset = QPushButton("🔄  Reset to Original")

        for b in (btn_load, btn_save):
            b.setStyleSheet("background-color: #F9F5EB; font-weight: bold;")

        btn_load.clicked.connect(self.load_image)
        btn_save.clicked.connect(self.save_image)
        btn_reset.clicked.connect(self.reset_image)

        ly.addWidget(btn_load)
        ly.addWidget(btn_save)
        ly.addWidget(btn_reset)
        gp.setLayout(ly)
        return gp

    def _build_group_cnn(self):
        gp = QGroupBox("🤖  CNN Object Detection")
        ly = QVBoxLayout()

        ly.addWidget(QLabel("Target Object:"))
        self.cb_cnn_target = QComboBox()
        self.cb_cnn_target.addItems([c for c in CLASSES if c != "background"])

        ly.addWidget(self.cb_cnn_target)

        # Threshold slider
        ly.addWidget(QLabel("Min. Confidence Threshold (%)"))
        self.sld_confidence = QSlider(Qt.Horizontal)
        self.sld_confidence.setRange(10, 95)
        self.sld_confidence.setValue(45)
        self.lbl_conf_val = QLabel("45%")
        self.sld_confidence.valueChanged.connect(
            lambda v: self.lbl_conf_val.setText(f"{v}%")
        )
        h_conf = QHBoxLayout()
        h_conf.addWidget(self.sld_confidence)
        h_conf.addWidget(self.lbl_conf_val)
        ly.addLayout(h_conf)

        # Deteksi semua objek atau target saja
        self.cb_detect_mode = QComboBox()
        self.cb_detect_mode.addItems(["Deteksi target saja", "Deteksi SEMUA objek"])
        ly.addWidget(self.cb_detect_mode)

        self.btn_detect = QPushButton("🔍  Execute CNN Detection")
        self.btn_detect.setStyleSheet(
            "background-color: #D4AF37; color: white; font-weight: bold; padding: 9px;"
        )
        self.btn_detect.clicked.connect(self.apply_cnn_detection)
        ly.addWidget(self.btn_detect)

        # Tombol cek model
        btn_check = QPushButton("⚙️  Cek / Download Model")
        btn_check.clicked.connect(self.check_model_files)
        ly.addWidget(btn_check)

        gp.setLayout(ly)
        return gp

    # TAMBAHAN BAGIAN B
    def _build_group_classifier(self):
        gp = QGroupBox("🧠  CNN Image Classification (Custom)")
        ly = QVBoxLayout()
        
        ly.addWidget(QLabel("Top-3 prediksi model custom:"))
        self.lbl_clf_result = QLabel("Belum ada hasil.\nBuka gambar lalu klik Classify.")
        self.lbl_clf_result.setWordWrap(True)
        self.lbl_clf_result.setStyleSheet(
            "background-color: #F9F5EB; padding: 10px;"
            "border-radius: 4px; font-size: 11px; color: #333333;"
            "border: 1px solid #E0D0B0;"
        )
        self.lbl_clf_result.setMinimumHeight(80)
        ly.addWidget(self.lbl_clf_result)
        
        self.btn_classify = QPushButton("🔎  Classify Image (Custom CNN)")
        self.btn_classify.setStyleSheet(
            "background-color: #4A90D9; color: white;"
            "font-weight: bold; padding: 9px;"
        )
        self.btn_classify.clicked.connect(self.apply_classification)
        ly.addWidget(self.btn_classify)
        
        gp.setLayout(ly)
        return gp

    def _build_group_enhancement(self):
        gp = QGroupBox("Enhancement & Filtering")
        ly = QVBoxLayout()

        self.sld_bright = QSlider(Qt.Horizontal)
        self.sld_bright.setRange(-100, 100); self.sld_bright.setValue(0)
        self.sld_bright.valueChanged.connect(self.process_enhancement)

        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setRange(50, 300); self.sld_contrast.setValue(100)
        self.sld_contrast.valueChanged.connect(self.process_enhancement)

        btn_eq  = QPushButton("Histogram Equalization")
        btn_sh  = QPushButton("Sharpening (Laplacian)")
        btn_bl  = QPushButton("Gaussian Blur (Smoothing)")
        btn_med = QPushButton("Median Filter (Noise Removal)")

        btn_eq.clicked.connect(self.apply_equalization)
        btn_sh.clicked.connect(self.apply_sharpening)
        btn_bl.clicked.connect(self.apply_blur)
        btn_med.clicked.connect(self.apply_median)

        ly.addWidget(QLabel("Brightness"));  ly.addWidget(self.sld_bright)
        ly.addWidget(QLabel("Contrast"));    ly.addWidget(self.sld_contrast)
        ly.addWidget(btn_eq); ly.addWidget(btn_sh)
        ly.addWidget(btn_bl); ly.addWidget(btn_med)
        gp.setLayout(ly)
        return gp

    def _build_group_color(self):
        gp = QGroupBox("Color Processing")
        ly = QVBoxLayout()

        btn_gray = QPushButton("Convert to Grayscale")
        btn_gray.clicked.connect(self.apply_grayscale)

        h_ch = QHBoxLayout()
        for label, idx in [("Red", 2), ("Green", 1), ("Blue", 0)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, i=idx: self.apply_channel_split(i))
            h_ch.addWidget(b)

        self.sld_hue = QSlider(Qt.Horizontal)
        self.sld_hue.setRange(-180, 180); self.sld_hue.setValue(0)
        self.sld_hue.valueChanged.connect(self.process_color_adjustment)

        self.sld_sat = QSlider(Qt.Horizontal)
        self.sld_sat.setRange(-100, 100); self.sld_sat.setValue(0)
        self.sld_sat.valueChanged.connect(self.process_color_adjustment)

        ly.addWidget(btn_gray)
        ly.addLayout(h_ch)
        ly.addWidget(QLabel("Hue Shift"));   ly.addWidget(self.sld_hue)
        ly.addWidget(QLabel("Saturation")); ly.addWidget(self.sld_sat)
        gp.setLayout(ly)
        return gp

    def _build_group_segmentation(self):
        gp = QGroupBox("Segmentation & Edge Tools")
        ly = QVBoxLayout()

        btn_thresh = QPushButton("Otsu Threshold (Binary)")
        btn_thresh.clicked.connect(self.apply_threshold)

        self.cb_edge = QComboBox()
        self.cb_edge.addItems([
            "Canny", "Sobel", "Prewitt", "Robert",
            "Laplacian", "Laplacian of Gaussian (LoG)"
        ])
        btn_edge = QPushButton("Run Edge Detection")
        btn_edge.clicked.connect(self.apply_edge_detection)

        btn_dil = QPushButton("Morphology Dilation")
        btn_ero = QPushButton("Morphology Erosion")
        btn_dil.clicked.connect(lambda: self.apply_morphology('dilate'))
        btn_ero.clicked.connect(lambda: self.apply_morphology('erode'))

        ly.addWidget(btn_thresh)
        ly.addWidget(QLabel("Edge Operator:"))
        ly.addWidget(self.cb_edge)
        ly.addWidget(btn_edge)
        ly.addWidget(btn_dil)
        ly.addWidget(btn_ero)
        gp.setLayout(ly)
        return gp

    def _build_group_geometry(self):
        gp = QGroupBox("Geometric Matrix Transform")
        ly = QVBoxLayout()

        self.cb_interp = QComboBox()
        self.cb_interp.addItems(["Bilinear (Smooth)", "Nearest Neighbor (Sharp)"])

        self.sld_rotate = QSlider(Qt.Horizontal)
        self.sld_rotate.setRange(0, 360); self.sld_rotate.setValue(0)
        self.sld_rotate.valueChanged.connect(self.apply_geometry_transforms)

        btn_fh = QPushButton("Flip Horizontal")
        btn_fv = QPushButton("Flip Vertical")
        btn_fh.clicked.connect(lambda: self.apply_flip(1))
        btn_fv.clicked.connect(lambda: self.apply_flip(0))

        self.sld_resize = QSlider(Qt.Horizontal)
        self.sld_resize.setRange(10, 200); self.sld_resize.setValue(100)
        self.sld_resize.valueChanged.connect(self.apply_geometry_transforms)

        self.sld_trans_x = QSlider(Qt.Horizontal)
        self.sld_trans_x.setRange(-150, 150); self.sld_trans_x.setValue(0)
        self.sld_trans_x.valueChanged.connect(self.apply_geometry_transforms)

        self.sld_trans_y = QSlider(Qt.Horizontal)
        self.sld_trans_y.setRange(-150, 150); self.sld_trans_y.setValue(0)
        self.sld_trans_y.valueChanged.connect(self.apply_geometry_transforms)

        ly.addWidget(QLabel("Interpolation:"));       ly.addWidget(self.cb_interp)
        ly.addWidget(QLabel("Rotation Angle (°)"));  ly.addWidget(self.sld_rotate)
        ly.addWidget(btn_fh); ly.addWidget(btn_fv)
        ly.addWidget(QLabel("Scale (%)"));            ly.addWidget(self.sld_resize)
        ly.addWidget(QLabel("Translation X"));        ly.addWidget(self.sld_trans_x)
        ly.addWidget(QLabel("Translation Y"));        ly.addWidget(self.sld_trans_y)
        gp.setLayout(ly)
        return gp

    # =====================================================================
    # PART 4: FILE OPERATIONS
    # =====================================================================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.critical(self, "Error", "Gagal membaca gambar.")
                return
            self.img_orig = img
            self.img_proc = img.copy()
            self.reset_sliders()
            self.update_display()
            fname = os.path.basename(path)
            h, w  = img.shape[:2]
            self.set_status(f"Gambar dimuat: {fname}  |  {w}×{h} px")

    def save_image(self):
        if self.img_proc is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada gambar untuk disimpan.")
            return
        quality, ok = QInputDialog.getInt(
            self, "JPEG Compression",
            "Kualitas gambar (10 = low, 100 = high):", 90, 10, 100, 1
        )
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "hasil_edit.jpg", "JPEG (*.jpg);;PNG (*.png)"
        )
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imwrite(path, self.img_proc, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(path, self.img_proc)
            self.set_status(f"Gambar disimpan: {os.path.basename(path)}")

    def reset_image(self):
        if self.img_orig is not None:
            self.img_proc = self.img_orig.copy()
            self.reset_sliders()
            self.update_display()
            self.set_status("Gambar direset ke original.")

    def reset_sliders(self):
        for sld, val in [
            (self.sld_bright, 0), (self.sld_contrast, 100),
            (self.sld_rotate, 0), (self.sld_resize, 100),
            (self.sld_trans_x, 0), (self.sld_trans_y, 0),
            (self.sld_hue, 0), (self.sld_sat, 0),
        ]:
            sld.blockSignals(True)
            sld.setValue(val)
            sld.blockSignals(False)

    # =====================================================================
    # PART 5: CNN DETECTION (THREADED)
    # =====================================================================
    def check_model_files(self):
        proto_ok = os.path.exists(PROTOTXT_PATH)
        model_ok = os.path.exists(MODEL_PATH)

        msg = (
            f"Status file model CNN:\n\n"
            f"{'✅' if proto_ok else '❌'}  MobileNetSSD_deploy.prototxt"
            f"  {'(ada)' if proto_ok else '(belum ada)'}\n"
            f"{'✅' if model_ok else '❌'}  MobileNetSSD_deploy.caffemodel"
            f"  {'(ada)' if model_ok else '(belum ada ~23MB)'}\n\n"
        )

        if proto_ok and model_ok:
            msg += "Semua file siap. Kamu bisa langsung melakukan deteksi."
            QMessageBox.information(self, "Cek Model", msg)
        else:
            msg += "File belum lengkap. Klik OK untuk mengunduh otomatis."
            reply = QMessageBox.question(
                self, "Unduh Model?", msg,
                QMessageBox.Ok | QMessageBox.Cancel
            )
            if reply == QMessageBox.Ok:
                self._download_models_only()

    def _download_models_only(self):
        """Download model tanpa langsung melakukan deteksi."""
        self.set_status("Mengunduh model CNN...")
        self.btn_detect.setEnabled(False)

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        # Pakai thread tapi dengan gambar dummy — akan gagal di deteksi tapi model terunduh
        thread = DetectionThread(dummy, "person")
        thread.progress.connect(self.set_status)
        thread.finished.connect(lambda *_: self._on_download_done())
        thread.error.connect(lambda e: self._on_detection_error(e))
        self._det_thread = thread
        thread.start()

    def _on_download_done(self):
        self.btn_detect.setEnabled(True)
        self.set_status("Model CNN berhasil diunduh dan siap digunakan.")
        QMessageBox.information(self, "Unduh Selesai",
            "Model CNN berhasil diunduh!\nKamu sekarang bisa melakukan deteksi objek.")

    def apply_cnn_detection(self):
        if self.img_proc is None:
            QMessageBox.warning(self, "Peringatan", "Buka gambar terlebih dahulu!")
            return

        if self._det_thread and self._det_thread.isRunning():
            QMessageBox.information(self, "Mohon Tunggu", "Deteksi sedang berjalan...")
            return

        target     = self.cb_cnn_target.currentText()
        detect_all = self.cb_detect_mode.currentIndex() == 1
        conf_thr   = self.sld_confidence.value() / 100.0

        # Jalankan di thread agar UI tidak freeze
        thread = DetectionThread(self.img_proc, target if not detect_all else "__all__")
        thread._confidence_thr = conf_thr
        thread.progress.connect(self.set_status)
        thread.finished.connect(self._on_detection_finished)
        thread.error.connect(self._on_detection_error)

        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("⏳  Mendeteksi...")
        self.set_status("CNN sedang memproses gambar...")
        self._det_thread = thread
        thread.start()

    def _on_detection_finished(self, result_img, count, target):
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("🔍  Execute CNN Detection")

        if count > 0:
            self.img_proc = result_img
            self.update_display()
            self.set_status(f"Deteksi selesai: {count} objek '{target}' ditemukan.")
            QMessageBox.information(
                self, "Deteksi Berhasil",
                f"✅  CNN menemukan {count} objek '{target}' pada gambar.\n\n"
                f"Hasil bounding box telah digambar di kanvas kanan."
            )
        else:
            obj_desc = "semua kelas" if target == "__all__" else f"'{target}'"
            self.set_status(f"Tidak ada objek {obj_desc} terdeteksi (conf ≥ {int(self.sld_confidence.value())}%).")
            QMessageBox.information(
                self, "Deteksi Selesai",
                f"Tidak ada objek {obj_desc} yang terdeteksi.\n\n"
                f"Saran:\n"
                f"• Turunkan threshold confidence\n"
                f"• Coba target objek lain\n"
                f"• Pastikan objek terlihat jelas di gambar"
            )

    def _on_detection_error(self, error_msg):
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("🔍  Execute CNN Detection")
        self.set_status(f"Error: {error_msg}")
        QMessageBox.critical(
            self, "Error CNN",
            f"Terjadi kesalahan saat deteksi:\n\n{error_msg}\n\n"
            f"Pastikan koneksi internet aktif untuk mengunduh model."
        )

    # TAMBAHAN BAGIAN C
    def apply_classification(self):
        if self.img_proc is None:
            QMessageBox.warning(self, "Peringatan", "Buka gambar terlebih dahulu!")
            return
            
        if not os.path.exists("model_custom.h5"):
            QMessageBox.warning(self, "Model Tidak Ada",
                "File model_custom.h5 tidak ditemukan.\n"
                "Jalankan train_cnn_cifar.py terlebih dahulu!")
            return
            
        try:
            self.btn_classify.setEnabled(False)
            self.btn_classify.setText("⏳  Memproses...")
            self.set_status("Menjalankan klasifikasi CNN custom...")
            QApplication.processEvents()
            
            from cnn_classifier import CNNClassifier
            from PIL import Image as PILImage
            
            # Load model (sekali saja, simpan di instance)
            if not hasattr(self, '_classifier') or self._classifier is None:
                self._classifier = CNNClassifier("model_custom.h5", "class_names.txt")
                
            # Convert cv2 → PIL
            rgb     = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            
            # Prediksi
            results = self._classifier.predict(pil_img, top_k=3)
            
            # Tampilkan hasil
            output = ""
            for i, (label, conf) in enumerate(results, 1):
                bar    = "█" * int(conf / 10)
                output += f"{i}. {label.upper()}: {conf:.1f}%\n    {bar}\n"
                
            self.lbl_clf_result.setText(output.strip())
            self.set_status(
                f"Klasifikasi selesai: {results[0][0].upper()} "
                f"({results[0][1]:.1f}% confidence)"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error Klasifikasi", f"Gagal:\n{str(e)}")
            self.set_status("Error saat klasifikasi.")
        finally:
            self.btn_classify.setEnabled(True)
            self.btn_classify.setText("🔎  Classify Image (Custom CNN)")

    # =====================================================================
    # PART 6: IMAGE PROCESSING ALGORITHMS
    # =====================================================================
    def process_enhancement(self):
        if self.img_orig is None:
            return
        b = self.sld_bright.value()
        c = self.sld_contrast.value() / 100.0
        self.img_proc = cv2.convertScaleAbs(self.img_orig, alpha=c, beta=b)
        self.update_display()

    def apply_equalization(self):
        if self.img_proc is None:
            return
        ycrcb = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2YCrCb)
        ch    = list(cv2.split(ycrcb))
        ch[0] = cv2.equalizeHist(ch[0])
        self.img_proc = cv2.cvtColor(cv2.merge(ch), cv2.COLOR_YCrCb2BGR)
        self.update_display()
        self.set_status("Histogram Equalization diterapkan.")

    def apply_sharpening(self):
        if self.img_proc is None:
            return
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        self.img_proc = cv2.filter2D(self.img_proc, -1, kernel)
        self.update_display()
        self.set_status("Sharpening (Laplacian) diterapkan.")

    def apply_blur(self):
        if self.img_proc is None:
            return
        self.img_proc = cv2.GaussianBlur(self.img_proc, (5, 5), 0)
        self.update_display()
        self.set_status("Gaussian Blur diterapkan.")

    def apply_median(self):
        if self.img_proc is None:
            return
        self.img_proc = cv2.medianBlur(self.img_proc, 5)
        self.update_display()
        self.set_status("Median Filter diterapkan.")

    def apply_grayscale(self):
        if self.img_proc is None:
            return
        gray = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2GRAY)
        self.img_proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.update_display()
        self.set_status("Konversi ke Grayscale.")

    def apply_channel_split(self, channel_idx):
        if self.img_proc is None:
            return
        ch    = list(cv2.split(self.img_proc))
        blank = np.zeros_like(ch[0])
        merged = [blank, blank, blank]
        merged[channel_idx] = ch[channel_idx]
        self.img_proc = cv2.merge(merged)
        self.update_display()
        names = {0: "Blue", 1: "Green", 2: "Red"}
        self.set_status(f"Channel {names[channel_idx]} diisolasi.")

    def process_color_adjustment(self):
        if self.img_orig is None:
            return
        h_shift = self.sld_hue.value()
        s_adj   = self.sld_sat.value()
        hsv     = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_adj, 0, 255)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        self.img_proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.update_display()

    def apply_threshold(self):
        if self.img_proc is None:
            return
        gray = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.img_proc = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        self.update_display()
        self.set_status("Otsu Thresholding diterapkan.")

    def apply_edge_detection(self):
        if self.img_proc is None:
            return
        gray   = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2GRAY)
        method = self.cb_edge.currentText()
        res    = None

        if method == "Canny":
            res = cv2.Canny(gray, 50, 150)
        elif method == "Sobel":
            gx  = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
            gy  = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
            res = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
        elif method == "Prewitt":
            kx  = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
            ky  = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
            res = cv2.addWeighted(cv2.filter2D(gray, -1, kx), 0.5,
                                  cv2.filter2D(gray, -1, ky), 0.5, 0)
        elif method == "Robert":
            kx  = np.array([[1, 0], [0, -1]], np.float32)
            ky  = np.array([[0, 1], [-1, 0]], np.float32)
            res = cv2.addWeighted(cv2.filter2D(gray, -1, kx), 0.5,
                                  cv2.filter2D(gray, -1, ky), 0.5, 0)
        elif method == "Laplacian":
            res = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        elif method == "Laplacian of Gaussian (LoG)":
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            res     = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_16S, ksize=3))

        if res is not None:
            self.img_proc = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
            self.update_display()
            self.set_status(f"Edge detection ({method}) diterapkan.")

    def apply_morphology(self, op_type):
        if self.img_proc is None:
            return
        gray    = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2GRAY)
        _, th   = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if op_type == 'dilate':
            res = cv2.dilate(th, kernel, iterations=1)
        else:
            res = cv2.erode(th, kernel, iterations=1)
        self.img_proc = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.update_display()
        self.set_status(f"Morphology {op_type} diterapkan.")

    def apply_flip(self, mode):
        if self.img_proc is None:
            return
        self.img_proc = cv2.flip(self.img_proc, mode)
        self.update_display()
        self.set_status("Flip diterapkan.")

    def apply_geometry_transforms(self):
        if self.img_orig is None:
            return
        img   = self.img_orig.copy()
        h, w  = img.shape[:2]
        interp = (cv2.INTER_LINEAR
                  if self.cb_interp.currentIndex() == 0
                  else cv2.INTER_NEAREST)

        scale = self.sld_resize.value() / 100.0
        img   = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=interp)
        h, w  = img.shape[:2]
        cx, cy = w // 2, h // 2

        M_rot = cv2.getRotationMatrix2D((cx, cy), self.sld_rotate.value(), 1.0)
        img   = cv2.warpAffine(img, M_rot, (w, h), flags=interp)

        M_tr  = np.float32([[1, 0, self.sld_trans_x.value()],
                             [0, 1, self.sld_trans_y.value()]])
        self.img_proc = cv2.warpAffine(img, M_tr, (w, h), flags=interp)
        self.update_display()

    def apply_crop(self, start_pos, end_pos):
        if self.img_proc is None:
            return
        lbl_w, lbl_h  = self.lbl_proc.width(), self.lbl_proc.height()
        img_h, img_w  = self.img_proc.shape[:2]
        x1, y1 = max(0, start_pos.x()), max(0, start_pos.y())
        x2, y2 = min(lbl_w, end_pos.x()), min(lbl_h, end_pos.y())
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        rx1 = int(xmin / lbl_w * img_w); rx2 = int(xmax / lbl_w * img_w)
        ry1 = int(ymin / lbl_h * img_h); ry2 = int(ymax / lbl_h * img_h)

        if (rx2 - rx1) > 5 and (ry2 - ry1) > 5:
            self.img_proc = self.img_proc[ry1:ry2, rx1:rx2]
            self.update_display()
            self.set_status(f"Crop: ({rx1},{ry1}) → ({rx2},{ry2})")

    # =====================================================================
    # PART 7: RENDERING & HISTOGRAM
    # =====================================================================
    def update_display(self):
        if self.img_orig is not None:
            self._show_pixmap(self.img_orig, self.lbl_orig)
        if self.img_proc is not None:
            self._show_pixmap(self.img_proc, self.lbl_proc)
            self._update_histogram()

    def _show_pixmap(self, img, label):
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def _update_histogram(self):
        self.ax.clear()
        img = self.img_proc
        if img is None:
            return

        b, g, r = cv2.split(img)
        is_gray = np.array_equal(b, g) and np.array_equal(g, r)

        if is_gray:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            self.ax.plot(hist, color='#8B7355', linewidth=1.5)
            self.ax.fill_between(range(256), hist, color=GOLD_HEX, alpha=0.2)
        else:
            for channel, color, label in zip(
                [b, g, r], ['#4466CC', '#44AA44', '#CC4444'],
                ['Blue', 'Green', 'Red']
            ):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
                self.ax.plot(hist, color=color, linewidth=1.2, alpha=0.85, label=label)
            self.ax.legend(fontsize=7, loc='upper right')

        self.ax.set_title("Live Histogram", color='#222222',
                          fontsize=9, fontweight='bold', pad=4)
        self.ax.tick_params(colors='#555555', labelsize=7)
        self.ax.set_facecolor('#FAFAFA')
        self.ax.set_xlim([0, 256])
        for spine in self.ax.spines.values():
            spine.set_color('#E0D0B0')

        self.canvas_hist.draw()

    def set_status(self, msg: str):
        self.lbl_status.setText(msg)
        QApplication.processEvents()


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MiniPhotoshopPro()
    window.show()
    sys.exit(app.exec_())