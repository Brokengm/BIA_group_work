"""
GlaucoScan (PyQt5) – glaucoma demo GUI.

Run:
  python app_gui.py

Expected files (relative to this script):
  - welcome.jpeg
  - model.py / preprocess.py
  - 384_unet.pth, efficientnet.pth
"""

# ---- Global UI constants ----
APP_TITLE = "GlaucoScan"
FONT_FAMILY = "Microsoft YaHei"


"""
Main application file: UNet optic-disc segmentation + EfficientNet-B3 glaucoma classification + PyQt5 GUI
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import math
from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms

from model import get_unet_model, get_effnet_model, DEVICE
from preprocess import preprocess_fundus_image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox,
    QFrame, QSpacerItem, QSizePolicy, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WELCOME_IMAGE_PATH = os.path.join(BASE_DIR, "welcome.jpeg")
CLASS_NAMES = ['Non-Glaucoma', 'Glaucoma']
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

print(f"[INFO] Using device: {DEVICE}")


def predict_single_cropped_image_array(img_array: np.ndarray) -> dict:
    """
    Runs the classification model on a pre-processed 384x384 image array.
    """
    model = get_effnet_model() # Get the pre-loaded EfficientNet classifier

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    pil_image = Image.fromarray(img_array)
    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1).cpu().squeeze().numpy()
    predicted_index = int(np.argmax(probabilities))

    result = {
        'predicted_class': CLASS_NAMES[predicted_index],
    }
    for i, class_name in enumerate(CLASS_NAMES):
        result[f'prob_{class_name}'] = float(probabilities[i])

    return result


def glaucoma_predict_pipeline(image_path: str) -> dict:
    """
    Main pipeline: UNet segmentation (preprocessing) -> EfficientNet classification.
    This is the primary function called by the GUI logic.
    """
    if not os.path.exists(image_path):
        return {'error': f"file not found: {image_path}"}

    unet_model = get_unet_model()
    cropped_img = preprocess_fundus_image(image_path, unet_model=unet_model, debug=False)

    cls_result = predict_single_cropped_image_array(cropped_img)
    cls_result['filename'] = Path(image_path).name

    return cls_result


class WelcomePage(QWidget):
    start_clicked = pyqtSignal()

    DESIGN_W = 1440
    DESIGN_H = 810

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scale = 1.0

        self.bg_pixmap = QPixmap()
        if os.path.exists(WELCOME_IMAGE_PATH):
            self.bg_pixmap.load(WELCOME_IMAGE_PATH)

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(800, 500)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bg_label.setScaledContents(False)
        root_layout.addWidget(self.bg_label)

        self.overlay_widget = QWidget(self.bg_label)
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.overlay_layout = QVBoxLayout(self.overlay_widget)
        self.overlay_layout.setContentsMargins(24, 24, 24, 24)  # will be scaled in _apply_scale()
        self.overlay_layout.setSpacing(0)

        self.overlay_layout.addStretch(1)

        self.bottom_row = QHBoxLayout()
        self.bottom_row.setContentsMargins(0, 0, 0, 0)
        self.bottom_row.setSpacing(0)

        self.welcome_text = QLabel(self.overlay_widget)
        self.welcome_text.setTextFormat(Qt.RichText)
        self.welcome_text.setWordWrap(True)
        self.welcome_text.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.welcome_text.setStyleSheet("background: transparent;")

        self.start_button = QPushButton("Let's get started", self.overlay_widget)
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setFocusPolicy(Qt.NoFocus)
        self.start_button.clicked.connect(self.start_clicked.emit)

        self.bottom_row.addWidget(self.welcome_text, 0, Qt.AlignLeft | Qt.AlignBottom)
        self.bottom_row.addStretch(1)
        self.bottom_row.addWidget(self.start_button, 0, Qt.AlignRight | Qt.AlignBottom)

        self.overlay_layout.addLayout(self.bottom_row)

        self._apply_scale()
        self._update_overlay_geometry()
        self.update_background()

    def _scale_factor(self) -> float:
        w = max(1, self.width())
        h = max(1, self.height())
        s = min(w / float(self.DESIGN_W), h / float(self.DESIGN_H))
        return max(0.55, min(1.75, s))

    def _apply_scale(self):
        s = self._scale_factor()
        self._scale = s

        pad = int(round(32 * s))
        self.overlay_layout.setContentsMargins(pad, pad, pad, pad)

        fs1 = int(round(44 * s))   # "Welcome to the"
        fs2 = int(round(68 * s))   # "GlaucoScan"
        fs1 = max(18, min(88, fs1))
        fs2 = max(26, min(120, fs2))

        self.welcome_text.setText(
            f"<div style='color:white; font-family:{FONT_FAMILY}; font-weight:700; line-height:1.05; font-size:{fs1}px;'>"
            f"Welcome to the"
            f"</div>"
            f"<div style='color:white; font-family:{FONT_FAMILY}; font-weight:800; line-height:1.05; font-size:{fs2}px;'>"
            f"GlaucoScan"
            f"</div>"
        )

        btn_fs = int(round(22 * s))
        btn_fs = max(12, min(40, btn_fs))
        pv = int(round(10 * s))
        ph = int(round(24 * s))
        radius = int(round(16 * s))
        radius = max(10, min(28, radius))

        self.start_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #4D6BFF;"
            "  color: white;"
            f"  font-size: {btn_fs}px;"
            "  font-weight: 700;"
            f"  padding: {pv}px {ph}px;"
            f"  border-radius: {radius}px;"
            "  border: none;"
            "}"
            "QPushButton:hover { background-color: #3F5CF5; }"
            "QPushButton:pressed { background-color: #314FEA; }"
        )

    def _update_overlay_geometry(self):
        if hasattr(self, "bg_label") and hasattr(self, "overlay_widget"):
            self.overlay_widget.setGeometry(self.bg_label.rect())

    def update_background(self):
        if self.bg_pixmap.isNull():
            self.bg_label.setText("Welcome image not found")
            self.bg_label.setAlignment(Qt.AlignCenter)
            return

        target = self.bg_label.size()
        if target.width() <= 1 or target.height() <= 1:
            return

        dpr = self.devicePixelRatioF()
        scaled = self.bg_pixmap.scaled(
            int(target.width() * dpr),
            int(target.height() * dpr),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self.bg_label.setPixmap(scaled)
        self.bg_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scale()
        self._update_overlay_geometry()
        self.update_background()


class AnalyzePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self.original_pixmap = None
        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(700, 500)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f5f7fb"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.top_label = QLabel(
            "Please drop one or more eye images for inspection\n"
            "(This version currently supports selecting a single image "
            "via the 'Add image' button below.)"
        )
        self.top_label.setAlignment(Qt.AlignCenter)
        self.top_label.setWordWrap(True)
        self.top_label.setStyleSheet("color: #555555;")
        self.top_label.setFont(QFont(FONT_FAMILY, 10))

        self.image_label = QLabel(
            "Drop area\n\nor click the 'Add image' button below"
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet(
            "color: #777777;"
            "background-color: #ffffff;"
            "border-radius: 12px;"
            "border: 1px dashed #b2bec3;"
        )

        image_card_layout = QVBoxLayout()
        image_card_layout.addWidget(self.image_label)

        image_card = QFrame()
        image_card.setLayout(image_card_layout)
        image_card.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 12px; }"
        )

        self.result_title = QLabel("Analysis result")
        self.result_title.setFont(QFont(FONT_FAMILY, 12, QFont.Bold))

        self.result_label = QLabel("No image has been analyzed yet.")
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "color: black;"
            "background-color: #f8f9fc;"
            "border-radius: 8px;"
            "padding: 10px;"
        )

        self.hint_label = QLabel(
            "This software is for course work and educational demonstration only. "
            "It must NOT be used for real medical diagnosis or self-assessment "
            "of glaucoma."
        )
        self.hint_label.setWordWrap(True)
        
        self.hint_label.setStyleSheet("color: black;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.result_title)
        right_layout.addSpacing(6)
        right_layout.addWidget(self.result_label)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.hint_label)
        right_layout.addStretch()

        right_card = QFrame()
        right_card.setLayout(right_layout)
        right_card.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 12px; "
            "border: 1px solid #dde0e7; }"
        )

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(image_card, stretch=3)
        middle_layout.addSpacing(12)
        middle_layout.addWidget(right_card, stretch=2)
        middle_layout.setContentsMargins(12, 12, 12, 12)

        self.load_button = QPushButton("Add image")
        self.load_button.setMinimumHeight(36)
        self.load_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #ffffff;"
            "  border-radius: 8px;"
            "  border: 1px solid #b2bec3;"
            "  padding: 6px 16px;"
            "  color: #2d3436;"
            "}"
            "QPushButton:hover {"
            "  background-color: #ecf0f1;"
            "}"
        )
        self.load_button.clicked.connect(self.load_image)

        self.analyze_button = QPushButton("Run analysis")
        self.analyze_button.setMinimumHeight(36)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #4c6fff;"
            "  border-radius: 8px;"
            "  border: none;"
            "  padding: 6px 18px;"
            "  color: white;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background-color: #3b59d4;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #b3c3ff;"
            "}"
        )
        self.analyze_button.clicked.connect(self.run_analysis)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.load_button)
        bottom_layout.addWidget(self.analyze_button)
        bottom_layout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        bottom_layout.setContentsMargins(12, 0, 12, 12)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)
        root_layout.addWidget(self.top_label)
        root_layout.addLayout(middle_layout)
        root_layout.addLayout(bottom_layout)

        self.update_typography()

    def load_image(self):
        """Opens a file dialog, loads the selected image, and displays it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select eye image",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
            "All files (*.*)"
        )

        if not file_path:
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(
                self, "Error", "File does not exist. Please try again."
            )
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(
                self, "Error",
                "Failed to load this image file.\n"
                "Please check that its format is supported."
            )
            return

        self.current_image_path = file_path
        self.original_pixmap = pixmap
        self.update_image_display() # Update display with the new image

        self.analyze_button.setEnabled(True) # Enable analysis button
        
        self.result_label.setText(
            f"Loaded image: {os.path.basename(file_path)}\n\n"
            "Click the \"Run analysis\" button on the right."
        )
        self.result_label.setStyleSheet(
            "color: black;"
            "background-color: #f8f9fc;"
            "border-radius: 8px;"
            "padding: 10px;"
        )

    def update_image_display(self):
        """Scales and sets the loaded image to fit the image display label."""
        if self.original_pixmap is None:
            return

        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled_pixmap = self.original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setStyleSheet(
            "border-radius: 12px;"
            "border: 1px solid #dde0e7;"
        )

    
    def update_typography(self):
        w = max(1, self.width())
        h = max(1, self.height())
        base = int(math.sqrt(w * h))  # scale with area (more consistent between normal/maximized)

        top_size = max(8,  min(18, int(base * 0.014)))   # header instructions
        title_size = max(10, min(24, int(base * 0.018))) # "Analysis result"
        body_size = max(8,  min(18, int(base * 0.014)))  # result text / hints / drop area
        btn_size = max(8,  min(16, int(base * 0.014)))
        btn_h = max(30, int(base * 0.055))

        self.top_label.setFont(QFont(FONT_FAMILY, top_size))
        self.image_label.setFont(QFont(FONT_FAMILY, body_size))
        self.result_title.setFont(QFont(FONT_FAMILY, title_size, QFont.Bold))
        self.result_label.setFont(QFont(FONT_FAMILY, body_size))
        self.hint_label.setFont(QFont(FONT_FAMILY, body_size))

        self.load_button.setFont(QFont(FONT_FAMILY, btn_size, QFont.Bold))
        self.analyze_button.setFont(QFont(FONT_FAMILY, btn_size, QFont.Bold))
        self.load_button.setMinimumHeight(btn_h)
        self.analyze_button.setMinimumHeight(btn_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()
        self.update_typography()

    def run_analysis(self):
        """
        Executes the full prediction pipeline (UNet + EfficientNet) for the loaded image
        and updates the result panel.
        """
        if self.current_image_path is None:
            QMessageBox.information(
                self, "Info", "Please add an image first."
            )
            return

        self.analyze_button.setEnabled(False)
        self.result_label.setText("Analyzing... please wait.")
        self.result_label.setStyleSheet(
            "color: black; background-color: #f8f9fc; "
            "border-radius: 8px; padding: 10px;"
        )
        QApplication.processEvents() # Force GUI redraw

        try:
            result = glaucoma_predict_pipeline(self.current_image_path)

            if "error" in result:
                self.result_label.setText("Error: " + result["error"])
                self.result_label.setStyleSheet(
                    "color: black; background-color: #f8f9fc; "
                    "border-radius: 8px; padding: 10px;"
                )
                return

            text = (
                f"File name: {result['filename']}\n"
                f"Predicted class: {result['predicted_class']}\n"
                f"Probability (Non-Glaucoma): {result['prob_Non-Glaucoma']:.4f}\n"
                f"Probability (Glaucoma): {result['prob_Glaucoma']:.4f}"
            )

            self.result_label.setText(text)
            self.result_label.setStyleSheet(
                "color: black; background-color: #f8f9fc; "
                "border-radius: 8px; padding: 10px;"
            )

        except Exception as e:
            self.result_label.setText("Error occurred:\n" + str(e))
            self.result_label.setStyleSheet(
                "color: black; background-color: #f8f9fc; "
                "border-radius: 8px; padding: 10px;"
            )
        finally:
            self.analyze_button.setEnabled(True) # Re-enable the button


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 650)

        self.stack = QStackedWidget()
        self.welcome_page = WelcomePage()
        self.analyze_page = AnalyzePage()

        self.stack.addWidget(self.welcome_page)
        self.stack.addWidget(self.analyze_page)

        self.setCentralWidget(self.stack)

        self.welcome_page.start_clicked.connect(
            lambda: self.stack.setCurrentWidget(self.analyze_page)
        )


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont(FONT_FAMILY, 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


# ---- Entry point ----
if __name__ == "__main__":
    main()
