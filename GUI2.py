import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox,
    QFrame, QSpacerItem, QSizePolicy, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, pyqtSignal


WELCOME_IMAGE_PATH = r"C:\Users\yb\Desktop\welcome.jpeg"


class WelcomePage(QWidget):
    start_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_pixmap = QPixmap()
        if os.path.exists(WELCOME_IMAGE_PATH):
            self.bg_pixmap.load(WELCOME_IMAGE_PATH)
        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(800, 500)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.bg_label = QLabel()
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setStyleSheet("background-color: #000000;")
        root_layout.addWidget(self.bg_label)

        overlay_layout = QVBoxLayout(self.bg_label)
        overlay_layout.setContentsMargins(24, 24, 24, 24)
        overlay_layout.setSpacing(10)

        overlay_layout.addStretch()

        bottom_row = QHBoxLayout()

        self.welcome_text = QLabel("Welcome to the\nGlaucoma Detector")
        self.welcome_text.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.welcome_text.setStyleSheet("color: white;")
        bottom_row.addWidget(self.welcome_text)

        bottom_row.addStretch()

        self.start_button = QPushButton("Let’s get started")
        self.start_button.setMinimumHeight(40)
        self.start_button.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.start_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #4c6fff;"
            "  color: white;"
            "  border-radius: 20px;"
            "  padding: 6px 20px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #3b59d4;"
            "}"
        )
        self.start_button.clicked.connect(self.start_clicked.emit)
        bottom_row.addWidget(self.start_button)

        overlay_layout.addLayout(bottom_row)
        self.update_background()

    def update_background(self):
        if not self.bg_pixmap.isNull():
            size = self.size()
            if size.width() > 0 and size.height() > 0:
                scaled = self.bg_pixmap.scaled(
                    size,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )
                self.bg_label.setPixmap(scaled)
                self.bg_label.setStyleSheet("")
        else:
            self.bg_label.setText(
                "Welcome to the Glaucoma Detector\n\n"
                "(Background image not found. "
                "Please set WELCOME_IMAGE_PATH at the top of the code.)"
            )
            self.bg_label.setStyleSheet(
                "color: white; background-color: #2d3436;"
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_background()


class AnalyzePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self.original_pixmap = None
        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(900, 600)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f5f7fb"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        top_label = QLabel(
            "Please drop one or more eye images for inspection\n"
            "(This demo currently supports selecting a single image "
            "via the 'Add image' button below.)"
        )
        top_label.setAlignment(Qt.AlignCenter)
        top_label.setStyleSheet("color: #555555;")
        top_label.setFont(QFont("Microsoft YaHei", 10))

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
        self.result_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))

        self.result_label = QLabel("No image has been analyzed yet.")
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "color: #333333;"
            "background-color: #f8f9fc;"
            "border-radius: 8px;"
            "padding: 10px;"
        )

        self.hint_label = QLabel(
            "⚠ This software is for course work and educational demonstration only. "
            "It must NOT be used for real medical diagnosis or self-assessment "
            "of glaucoma."
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #c0392b;")

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
        root_layout.addWidget(top_label)
        root_layout.addLayout(middle_layout)
        root_layout.addLayout(bottom_layout)

    def load_image(self):
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
        self.update_image_display()

        self.analyze_button.setEnabled(True)

        self.result_label.setText(
            f"Loaded image: {os.path.basename(file_path)}\n\n"
            "Click the \"Run analysis\" button on the right "
            "(currently a demo placeholder)."
        )
        self.result_label.setStyleSheet(
            "color: #333333;"
            "background-color: #f8f9fc;"
            "border-radius: 8px;"
            "padding: 10px;"
        )

    def update_image_display(self):
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

    def run_analysis(self):
        if self.current_image_path is None:
            QMessageBox.information(
                self, "Info", "Please add an image first."
            )
            return

        result_text, risk_level = self.dummy_analysis(self.current_image_path)

        if risk_level == "high":
            color = "#e74c3c"
        elif risk_level == "medium":
            color = "#e67e22"
        else:
            color = "#27ae60"

        self.result_label.setText(result_text)
        self.result_label.setStyleSheet(
            f"background-color: #f8f9fc;"
            f"border-radius: 8px;"
            f"padding: 10px;"
            f"color: {color};"
        )

    def dummy_analysis(self, image_path: str):
        base_name = os.path.basename(image_path)

        result_text = (
            "[Demo result - NOT for medical diagnosis]\n\n"
            f"Image file: {base_name}\n\n"
            "Here you can display the output of your model, for example:\n"
            "• Cup-to-Disc Ratio (CDR)\n"
            "• RNFL thickness and structural indices\n"
            "• Predicted probability of glaucoma, etc.\n\n"
            "This version is only a GUI demonstration and does NOT provide "
            "any real judgment about glaucoma."
        )
        risk_level = "medium"
        return result_text, risk_level


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Glaucoma Detector Demo")
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
