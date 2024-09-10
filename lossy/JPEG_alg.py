import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import os


def JPEG_ALGORITHM(image_path, scale: int, save_path: str):
    # Adjust quantization tables based on scale
    def adjust_quant_table(base_table, scale):
        scale = max(0, min(scale, 1000))
        scale_factor = 1 + (scale / 10)
        return np.clip(base_table * scale_factor, 1, 255)

    base_lum_quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    base_chrom_quant_table = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    # Create adjusted quantization tables
    lum_quant_table = adjust_quant_table(base_lum_quant_table, scale)
    chrom_quant_table = adjust_quant_table(base_chrom_quant_table, scale)

    image = Image.open(image_path)
    image_np = np.array(image)

    # If the image has an alpha channel, remove it
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]


    # RGB to YCbCr conversion
    def rgb_to_ycbcr(image):
        # Prepare the transformation matrix
        transform_matrix = np.array([[ 0.299,  0.587,  0.114],
                                    [-0.168736, -0.331264,  0.5],
                                    [ 0.5, -0.418688, -0.081312]])
        
        # Add the offset for Y, Cb, and Cr channels
        offset = np.array([0, 128, 128])

        # Apply the transformation
        ycbcr_image = image.dot(transform_matrix.T) + offset
        
        # Ensure the values are in the valid range [0, 255]
        ycbcr_image = np.clip(ycbcr_image, 0, 255).astype(np.uint8)
        
        return ycbcr_image

    # Convert the image
    ycbcr_image_np = rgb_to_ycbcr(image_np)

    # Extract the Y, Cb, and Cr channels
    y_channel = ycbcr_image_np[:, :, 0]
    cb_channel = ycbcr_image_np[:, :, 1]
    cr_channel = ycbcr_image_np[:, :, 2]

    # Downsample the chrominance channels (Cb and Cr)
    def downsample_channel(channel):
        # Ensure dimensions are even
        if channel.shape[0] % 2 != 0:
            channel = channel[:-1, :]
        if channel.shape[1] % 2 != 0:
            channel = channel[:, :-1]
        # Average each 2x2 block
        channel_downsampled = channel.reshape((channel.shape[0] // 2, 2, channel.shape[1] // 2, 2)).mean(axis=(1, 3))
        return channel_downsampled

    # Downsample all channels
    y_downsampled = downsample_channel(y_channel)
    cb_downsampled = downsample_channel(cb_channel)
    cr_downsampled = downsample_channel(cr_channel)

    # Ensure dimensions are multiples of 8 for 8x8 block processing
    def pad_to_multiple_of_8(channel):
        pad_h = (8 - channel.shape[0] % 8) % 8
        pad_w = (8 - channel.shape[1] % 8) % 8
        return np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    y_padded = pad_to_multiple_of_8(y_downsampled)
    cb_padded = pad_to_multiple_of_8(cb_downsampled)
    cr_padded = pad_to_multiple_of_8(cr_downsampled)

    # Divide each channel into 8x8 blocks
    def divide_into_blocks(channel):
        h, w = channel.shape
        blocks = channel.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        return blocks

    y_blocks = divide_into_blocks(y_padded)
    cb_blocks = divide_into_blocks(cb_padded)
    cr_blocks = divide_into_blocks(cr_padded)

    # Center pixel values

    y_centered = y_blocks - 128
    cb_centered = cb_blocks - 128
    cr_centered = cr_blocks - 128

    # Apply Forward DCT
    def apply_dct(blocks):
        return np.array([dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho') for block in blocks])

    y_dct = apply_dct(y_centered)
    cb_dct = apply_dct(cb_centered)
    cr_dct = apply_dct(cr_centered)

    # Apply quantization
    def quantize_block(dct_block, quant_table):
        return np.round(dct_block / quant_table).astype(np.int32)

    y_quantized = np.array([quantize_block(block, lum_quant_table) for block in y_dct])
    cb_quantized = np.array([quantize_block(block, chrom_quant_table) for block in cb_dct])
    cr_quantized = np.array([quantize_block(block, chrom_quant_table) for block in cr_dct])

    ###############################################################################################

    # Dequantization
    def dequantize_block(quantized_block, quant_table):
        return quantized_block * quant_table

    y_dequantized = np.array([dequantize_block(block, lum_quant_table) for block in y_quantized])
    cb_dequantized = np.array([dequantize_block(block, chrom_quant_table) for block in cb_quantized])
    cr_dequantized = np.array([dequantize_block(block, chrom_quant_table) for block in cr_quantized])

    # Inverse DCT
    def apply_idct(blocks):
        return np.array([idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho') for block in blocks])


    y_idct = apply_idct(y_dequantized)
    cb_idct = apply_idct(cb_dequantized)
    cr_idct = apply_idct(cr_dequantized)

    # Center the Pixel Values
    def center_pixel_values_back(blocks):
        return blocks + 128

    y_reconstructed = center_pixel_values_back(y_idct)
    cb_reconstructed = center_pixel_values_back(cb_idct)
    cr_reconstructed = center_pixel_values_back(cr_idct)

    # Reassemble the full channel images
    def assemble_from_blocks(blocks, shape):
        full_image = np.zeros(shape, dtype=np.float32)
        block_size = blocks.shape[1]
        
        for i in range(blocks.shape[0]):
            row = (i // (shape[1] // block_size)) * block_size
            col = (i % (shape[1] // block_size)) * block_size
            full_image[row:row + block_size, col:col + block_size] = blocks[i]
        
        return np.clip(full_image, 0, 255).astype(np.uint8)

    y_full = assemble_from_blocks(y_reconstructed, y_padded.shape)
    cb_full = assemble_from_blocks(cb_reconstructed, cb_padded.shape)
    cr_full = assemble_from_blocks(cr_reconstructed, cr_padded.shape)

    # Upsample the chrominance channels to match Y channel
    def upsample_channel(channel, target_shape):
        upsampled = channel.repeat(1, axis=0).repeat(1, axis=1)
        return upsampled[:target_shape[0], :target_shape[1]]

    cb_upsampled = upsample_channel(cb_full, y_full.shape)
    cr_upsampled = upsample_channel(cr_full, y_full.shape)

    # Convert YCbCr back to RGB
    def ycbcr_to_rgb(ycbcr):
        transform_matrix = np.array([[1, 0, 1.402],
                                    [1, -0.344136, -0.714136],
                                    [1, 1.772, 0]])
        ycbcr = ycbcr.astype(np.float32) - [0, 128, 128]  # Remove offset
        rgb_image = ycbcr.dot(transform_matrix.T)
        return np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Stack the channels together
    ycbcr_reconstructed = np.stack((y_full, cb_upsampled, cr_upsampled), axis=-1)
    rgb_reconstructed = ycbcr_to_rgb(ycbcr_reconstructed)

    # Save and display the image
    reconstructed_image = Image.fromarray(rgb_reconstructed)
    reconstructed_image.save(save_path)
    
    compressed_file_size = os.path.getsize(save_path)
    original_file_size = os.path.getsize(image_path)

    return reconstructed_image, original_file_size, compressed_file_size
