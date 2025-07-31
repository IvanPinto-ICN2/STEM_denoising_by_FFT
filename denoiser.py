import numpy as np


def fft(image, log = True):
    fft_image = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    if log:
        fft_image = np.log(fft_image)

    return fft_image

def fft_denoiser(fft, model):
    prediction, coordinates = model.predict(fft)
    pred_r = prediction[0]

    return pred_r

# Denoising trought ffts
def fft_denoising(image, dl_model, all_process = True):

    # Original fft
    fft_original = np.fft.fftshift(np.fft.fft2(image))

    # Log fft to apply the denoising
    log_fft = fft(image, log = True)

    # Denoising prediction
    fft_denoised = fft_denoiser(log_fft, dl_model, len(image))

    # Denoise the image (IFFT)
    img_denoised = np.abs(np.fft.ifft2(fft_original * fft_denoised))

    if all_process:
        process = {'original_image' : image, 
                  'original_fft' : log_fft, 
                  'denoised_fft': fft_denoised,
                  'denoised_image': img_denoised}
        return  process
    else:
        return img_denoised
