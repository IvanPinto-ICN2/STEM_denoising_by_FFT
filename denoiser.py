import numpy as np
import cv2


def fft(image, log = True):
    fft_image = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    if log:
        fft_image = np.log(fft_image)

    return fft_image

def fft_denoiser(fft, model):
    prediction, coordinates = model.predict(fft)
    pred_r = prediction[0, :, :, 0]

    return pred_r

# Denoising trought ffts
def fft_denoising(image, dl_model, all_process = True):
    # Grayscale image
    if len(image.shape) == 3:
        image = image[:,:,0]

    # Original fft
    fft_original = np.fft.fftshift(np.fft.fft2(image))
    print(fft_original.shape)

    # Log fft to apply the denoising
    log_fft = fft(image, log = True)
    print(log_fft.shape)

    # Denoising prediction
    fft_denoised = fft_denoiser(log_fft, dl_model)
    print(fft_denoised.shape)

    # Denoise the image (IFFT)
    img_denoised = np.abs(np.fft.ifft2(fft_original * fft_denoised))
    print(img_denoised.shape)

    if all_process:
        process = {'original_image' : image, 
                  'original_fft' : log_fft, 
                  'denoised_fft': fft_denoised,
                  'denoised_image': img_denoised}
        return  process
    else:
        return img_denoised
