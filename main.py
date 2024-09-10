import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QLabel,
    QInputDialog,
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from lossless.compress_huffman import huffman_encoding, decode_data
from PIL import Image
import numpy as np
from lossy.JPEG_alg import JPEG_ALGORITHM
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CustomInputDialog(QInputDialog):
    """Window to Select Compression Scale"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Choose a Compression Scale")  # Set dialog title
        self.setFixedSize(475, 250)  # Set width and height
        self.setWindowIcon(QIcon("sample_images/duck_icon.png"))  # Provide the path to icon file
        
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Remove the "?" button


class CompressionTool(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowIcon(QIcon("sample_images/duck_icon.png"))  # Provide the path to icon file
        self.setWindowTitle("Compression Tool - DS5 Exam")
        self.setGeometry(100, 100, 1200, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Compression Tool - DS5 Exam")
        title.setFont(QFont("Helvetica", 32, QFont.Bold)) # Change font style and size
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Main Buttons
        self.btn_lossy = QPushButton("Lossy")
        self.btn_lossless_encode = QPushButton("Lossless Encode")
        self.btn_combination = QPushButton("Lossy and Lossless Combination")
        self.btn_lossless_dencode = QPushButton("Decode")

        layout.addWidget(self.btn_lossy)
        layout.addWidget(self.btn_lossless_encode)
        layout.addWidget(self.btn_combination)
        layout.addWidget(self.btn_lossless_dencode)

        # Connect buttons to their actions
        self.btn_lossless_encode.clicked.connect(self.lossless_functionality)
        self.btn_lossless_dencode.clicked.connect(self.lossless_decode_functionality)
        self.btn_lossy.clicked.connect(self.lossy_functionality)
        self.btn_combination.clicked.connect(self.show_combination_options)

    def show_combination_options(self):
        """Function for Lossy then Lossless functionality. Saves a pickle file to desired destination"""
        file_filter = "Images (*.jpg *.png);;All files (*)"
        expected_ext = ('.jpg', '.png', '.PNG', '.JPG')

        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", file_filter)

        if file_path:
            if isinstance(expected_ext, tuple) and not file_path.endswith(expected_ext):
                self.show_error("File format should be JPG or PNG.")
            else:
                dialog = CustomInputDialog()
                dialog.setIntValue(10)
                dialog.setIntRange(0, 100)
                scale_box = dialog.exec_()
                compression_scale = dialog.intValue() if scale_box else 100

                if scale_box:
                    save_path = "./lossy/reconstructed_image.jpg"
                    logging.info("Compressing Image")
                    reconstructed_image, original_file_size, compressed_file_size = JPEG_ALGORITHM(file_path, compression_scale, save_path)
                    image_np = np.array(reconstructed_image)
                    image_np = image_np.astype(np.float64)
                    logging.info("Encoding Image")
                    encoded_data, tree, data_shape = huffman_encoding(image_np)

                    # Prompt user to select a location to save the encoded file
                    save_file_path, _ = QFileDialog.getSaveFileName(self, "Save Encoded File", "", "Pickle Files (*.pkl)")
                    if save_file_path:
                        # Save to the selected file using the path chosen by user
                        with open(save_file_path, 'wb') as file:
                            pickle.dump((encoded_data, tree, data_shape), file)
                        
                        original_size = sys.getsizeof(image_np)
                        compressed_size = sys.getsizeof(encoded_data)

                        compression_ratio = original_size / compressed_size

                        QMessageBox.information(self, "File Size & Compression Ratio",
                                                f"File size: {original_size} bytes ({original_size/1000000} MB)\n"
                                                f"Compressed file size: {compressed_size} bytes ({compressed_size/1000000} MB)\n"
                                                f"Compression Ratio: {compression_ratio}\n")
                        logging.info("File Saved successfully.")

    def lossless_decode_functionality(self):
        """Function Huffmann decoding and decodes a pickle file. Returns the image"""
        file_filter = "Images (*.txt *.pkl);;All files (*)"
        expected_ext = ('.pkl', '.TXT', '.rtf', '.RTF', '.txt', '.PKL')

        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", file_filter)

        if file_path:
            if isinstance(expected_ext, tuple) and not file_path.endswith(expected_ext):
                self.show_error("File format should be Pickle (.pkl).")
            else:
                try:
                    # Load from the selected pickle file using the path chosen by user
                    with open(file_path, 'rb') as file:
                        encoded_data, tree, data_shape = pickle.load(file)
                    logging.info("Decoding Image")
                    decoded_data = decode_data(encoded_data, tree, data_shape)
                    decoded_data_arr = decoded_data.astype('uint8')
                    img = Image.fromarray(decoded_data_arr)
                    logging.info("Showing Image")
                    img.show()
                    logging.info("Image Shown successfully.")
                except Exception as e:
                    self.show_error(f"Failed to decode the file. Error: {e}")


    def lossy_functionality(self):
        """Function to compress images with lossy JPEG with desired compression scale. Returns compressed image"""
        file_filter = "Images (*.jpg *.png);;All files (*)"
        expected_ext = ('.jpg', '.png', '.PNG', '.JPG')

        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", file_filter)

        if file_path:
            if isinstance(expected_ext, tuple) and not file_path.endswith(expected_ext):
                self.show_error("File format should be JPG or PNG.")
            else:
                dialog = CustomInputDialog()
                dialog.setIntValue(25)
                dialog.setIntRange(0, 100)
                scale_box = dialog.exec_()
                compression_scale = dialog.intValue() if scale_box else 100

                if scale_box:
                    save_path = "./lossy/reconstructed_image.jpg"
                    logging.info("Compressing Image")
                    reconstructed_image, original_file_size, compressed_file_size = JPEG_ALGORITHM(file_path, compression_scale, save_path)
                    compression_ratio = original_file_size / compressed_file_size
                    QMessageBox.information(self, "File Size & Compression Ratio",
                                            f"File size: {original_file_size/1000000} MB\n"
                                            f"Compressed file size: {compressed_file_size/1000000} MB\n"
                                            f"Compression Ratio: {compression_ratio}\n")
                    logging.info("Showing Image")
                    reconstructed_image.show()
                    logging.info("Compressed Image Shown successfully.")

    def lossless_functionality(self):
        """Function for lossless compression using Huffman Coding. Returns to save a pickle file at a chosen destination."""
        file_filter = "Images (*.jpg *.png);;All files (*)"
        expected_ext = ('.jpg', '.png', '.PNG', '.JPG')

        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", file_filter)

        if file_path:
            if isinstance(expected_ext, tuple) and not file_path.endswith(expected_ext):
                self.show_error("File format should be JPG or PNG.")
            else:
                image = Image.open(file_path)
                image_np = np.array(image)
                image_np = image_np.astype(np.float64)
                logging.info("Encoding Image")
                encoded_data, tree, data_shape = huffman_encoding(image_np)

                save_file_path, _ = QFileDialog.getSaveFileName(self, "Save Encoded File", "", "Pickle Files (*.pkl)")
                if save_file_path:
                    # Save to the selected file using the path chosen by the user
                    with open(save_file_path, 'wb') as file:
                        pickle.dump((encoded_data, tree, data_shape), file)

                    original_size = sys.getsizeof(image_np)
                    compressed_size = sys.getsizeof(encoded_data)

                    compression_ratio = original_size / compressed_size

                    QMessageBox.information(self, "File Size & Compression Ratio",
                                            f"File size: {original_size} bytes ({original_size/1000000} MB)\n"
                                            f"Compressed file size: {compressed_size} bytes ({compressed_size/1000000} MB)\n"
                                            f"Compression Ratio: {compression_ratio}\n")
                    logging.info("File saved successfully.")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompressionTool()
    window.show()
    sys.exit(app.exec_())
