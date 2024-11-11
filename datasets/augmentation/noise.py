import noisereduce as nr


def reduce_noise(y, sr):
    """
    Applies noise reduction to the audio data.
    """
    # Reduce noise in the audio signal
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    return reduced_noise

def apply_augmentation(y, sr):
    """
    Applies noise reduction and returns the denoised audio data with a suffix.
    """
    y_denoised = reduce_noise(y, sr)
    return [(y_denoised, "noise_reduced")]
