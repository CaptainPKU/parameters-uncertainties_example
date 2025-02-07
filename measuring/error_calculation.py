import numpy as np

def calculate_total_error(wave_use, flux_use, m_fit, weights, window_size=30):
    """
    Calculate the total error for the given wave and flux data.

    Parameters:
    wave_use (list): List of wavelength values.
    flux_use (list): List of flux values.
    m_fit (function): Fitted model function.
    weights (list): List of weights.
    window_size (int): Size of the running window. Default is 30.

    Returns:
    np.ndarray: Array of total errors.
    """
    # Filter wave_use and adjust flux_use accordingly
    wave_use = [i for i in wave_use if 4200 <= i <= 7000]
    flux_use = flux_use[:len(wave_use)]

    # Adjust weights accordingly
    weights = weights[:len(wave_use)]

    # Calculate residuals
    residuals = flux_use - m_fit(wave_use)
    residuals[weights == 0] = 0

    # Calculate standard deviation for each running window
    n = len(wave_use) // window_size + 1  # Number of windows, each containing window_size points
    running_std_list = []
    for i in range(n):
        window = residuals[i * window_size:(i + 1) * window_size]
        if len(window) > 0:  # Ensure the window has data points
            running_std_list.append(np.std(window))
    RMS_run = np.array(running_std_list)

    # Assign the standard deviation of each window to the new array RMS_run_sigma
    RMS_run_sigma = np.zeros_like(wave_use)
    for i in range(n):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(RMS_run_sigma))
        RMS_run_sigma[start_idx:end_idx] = RMS_run[i]

    # Calculate the observation error of the flux: poisson error
    poisson_error = 0.05 * flux_use

    # Calculate total error F_err
    F_err = np.sqrt(RMS_run_sigma**2 + poisson_error**2)

    return F_err