# Image Compression Tool - DS5 Exam

This project is an image compression tool that provides four main functionalities: Lossy Compression, Lossless Encoding, Lossless Decoding, and a combination of Lossy and Lossless Compression. The tool is built using Python and PyQt5 for the graphical user interface. 

## Features

1. **Lossy Compression**
2. **Lossless Encoding**
3. **Lossless Decoding**
4. **Lossy and Lossless Combination**

## Installation & Instructions

Create a New Virtual Environment:

Windows:
```bash
python -m venv venv
```

macOS/Linux:
```bash
python3 -m venv venv
```

Activate the Virtual Environment:

Windows:
```bash
.\venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```
Or simply:
```bash
pip install PyQt5 Pillow numpy scipy
```

## Usage

Run the tool using the following command:
```bash
python main.py
```

### Interface Overview

The main interface consists of four buttons, each corresponding to one of the core functionalities of the tool.

### 1. Lossy Compression

- **Button**: `Lossy`
- **Function**: Compress an image using the JPEG lossy algorithm.
- **Procedure**:
  1. Click the "Lossy" button.
  2. Select an image file (JPG or PNG).
  3. Choose a compression scale (0-100) where 0 is least compressed, and 100 is most compressed.
  4. The compressed image is opened, and the original and compressed file sizes, along with the compression ratio, are displayed.
  5. To save the compressed image, simply save it in the used image viewer/ organizer.

### 2. Lossless Encoding

- **Button**: `Lossless Encode`
- **Function**: Encode an image using Huffman encoding (lossless compression).
- **Procedure**:
  1. Click the "Lossless Encode" button.
  2. Select an image file (JPG or PNG).
  3. The image is encoded and is saved in a chosen directory.

### 3. Lossy & Lossless Combination

- **Button**: `Lossy and Lossless Combination`
- **Function**: First apply JPEG lossy compression, then encode the result using Huffman encoding.
- **Procedure**:
  1. Click the "Lossy and Lossless Combination" button.
  2. Select an image file (JPG or PNG).
  3. Choose a compression scale (0-100) where 0 is least compressed, and 100 is most compressed.
  4. The image is first compressed using a lossy algorithm, then encoded using Huffman encoding. The encoded is saved in a chosen directory.


### 4. Lossless Decoding

- **Button**: `Decode`
- **Function**: Decode an image that was previously encoded.
  - This works for images which have been compressed using the lossless functionality, as well as those which have been compressed using the lossy then lossless combination.
- **Procedure**:
  1. Click the "Decode" button.
  2. Select the encoded pickle file.
  3. The decoded image is reconstructed and displayed.
  4. To save the compressed image, simply save it in the used image viewer or organizer.
## Detailed Code Explanation

### Main Code Structure

- **Imports**: The necessary libraries are imported, including PyQt5 for the GUI, Pillow for image processing, Numpy for array manipulations, and custom modules for encoding and decoding.
- **Logging**: Configured to help with debugging and tracking the application's operations.
- **CustomInputDialog**: A subclass of `QInputDialog` for setting the compression scale.
- **CompressionTool**: The main class inheriting from `QWidget`, which sets up the GUI and defines the functionalities for each button.

### Functions

- **show_combination_options**: Handles the lossy and lossless combination option.
- **lossless_decode_functionality**: Handles the lossless decoding functionality.
- **lossy_functionality**: Handles the lossy compression functionality.
- **lossless_functionality**: Handles the lossless encoding functionality.
- **show_error**: Displays error messages.

### Main Execution

- The application is initialized, and the main window (`CompressionTool`) is displayed.
- **Keep in mind**: Larger images can take some time to compress. Patience may be required.

## Logging

The application uses logging to provide detailed information about its execution. Logs include timestamps, log levels, and messages. This helps in debugging and understanding the application's flow.

## Error Handling

The application includes error handling for incorrect file formats and displays appropriate error messages using message boxes.

## Acknowledgments

This tool was developed for the Programming & Algorithms DS5 Exam project. Special thanks to Eliott.

---

Happy compressing!