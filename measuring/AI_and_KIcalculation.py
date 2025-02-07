

import numpy as np
import sagan

def calculate_AI_KI_and_vpeak(wave_use, m_fit):
    """
    Calculate the broad line parameters for Ha and Hb.

    Parameters:
    wave_use (list): List of wavelength values.
    m_fit (function): Fitted model function.

    Returns:
    dict: Dictionary containing the broad line parameters for Ha and Hb.
    """
    ls_km = 299792.458  # Speed of light in km/s
    wave_dict = sagan.utils.line_wave_dict

    # Initialize m_multi
    m_multi = np.ones_like(wave_use)
    for m in m_fit:
        if hasattr(m, 'name') and m.name == 'multi':
            m_multi = m(wave_use)
            break  # Exit loop after finding 'multi'

    # Correct the wavelength so that the narrow lines have zero velocity
    dv_sys = m_fit['[O III]'].dv_c.value
    wave_corr = np.array(wave_use) * (1 - dv_sys / ls_km)

    # Calculate the velocity of broad H-alpha and H-beta profiles
    velc_Ha = (wave_corr - wave_dict['Halpha']) / wave_dict['Halpha'] * ls_km
    velc_Hb = (wave_corr - wave_dict['Hbeta']) / wave_dict['Hbeta'] * ls_km

    # Calculate the flux of broad H-alpha and H-beta profiles
    flux_Ha = m_fit[r'H$\alpha$'](wave_use) * m_multi
    flux_Hb = m_fit[r'H$\beta$'](wave_use) * m_multi

    # Calculate peak values
    peak_Ha = np.max(flux_Ha)
    peak_Hb = np.max(flux_Hb)

    # Calculate different heights
    height_1_4_Ha = peak_Ha / 4
    height_1_2_Ha = peak_Ha / 2
    height_3_4_Ha = 3 * peak_Ha / 4

    height_1_4_Hb = peak_Hb / 4
    height_1_2_Hb = peak_Hb / 2
    height_3_4_Hb = 3 * peak_Hb / 4

    # Find the velocities corresponding to these heights
    def find_velocity(velc, flux, height):
        indices = np.where(flux >= height)[0]
        if len(indices) < 2:
            return None, None  # Return None if no two edge points are found
        return velc[indices[0]], velc[indices[-1]]

    # Calculate velocities
    vB_1_4_Ha, vR_1_4_Ha = find_velocity(velc_Ha, flux_Ha, height_1_4_Ha)
    vB_1_2_Ha, vR_1_2_Ha = find_velocity(velc_Ha, flux_Ha, height_1_2_Ha)
    vB_3_4_Ha, vR_3_4_Ha = find_velocity(velc_Ha, flux_Ha, height_3_4_Ha)

    vB_1_4_Hb, vR_1_4_Hb = find_velocity(velc_Hb, flux_Hb, height_1_4_Hb)
    vB_1_2_Hb, vR_1_2_Hb = find_velocity(velc_Hb, flux_Hb, height_1_2_Hb)
    vB_3_4_Hb, vR_3_4_Hb = find_velocity(velc_Hb, flux_Hb, height_3_4_Hb)

    # Calculate v_peak
    v_peak_Ha = dv_sys
    v_peak_Hb = dv_sys

    # Calculate A.I. and K.I.
    def calculate_AI(vR_1_2, vB_1_2, v_peak):
        return (vR_1_2 + vB_1_2 - 2 * v_peak) / (vR_1_2 - vB_1_2)

    def calculate_KI(vR_3_4, vB_3_4, vR_1_4, vB_1_4):
        return (vR_3_4 - vB_3_4) / (vR_1_4 - vB_1_4)

    AI_Ha = calculate_AI(vR_1_2_Ha, vB_1_2_Ha, v_peak_Ha)
    KI_Ha = calculate_KI(vR_3_4_Ha, vB_3_4_Ha, vR_1_4_Ha, vB_1_4_Ha)

    AI_Hb = calculate_AI(vR_1_2_Hb, vB_1_2_Hb, v_peak_Hb)
    KI_Hb = calculate_KI(vR_3_4_Hb, vB_3_4_Hb, vR_1_4_Hb, vB_1_4_Hb)

    return {
        'Halpha': {
            'v_peak': v_peak_Ha,
            'A.I.': AI_Ha,
            'K.I.': KI_Ha
        },
        'Hbeta': {
            'v_peak': v_peak_Hb,
            'A.I.': AI_Hb,
            'K.I.': KI_Hb
        }
    }