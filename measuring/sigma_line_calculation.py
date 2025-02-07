import sagan
import numpy as np

def calculate_broad_line_sigma(m_fit, wave_use):
    ls_km = 299792.458  # 光速，单位为 km/s
    wave_dict = sagan.utils.line_wave_dict

    # 初始化 m_multi
    m_multi = np.ones_like(wave_use)
    for m in m_fit:
        if hasattr(m, 'name') and m.name == 'multi':
            m_multi = m(wave_use)
            break  # 找到 'multi' 后退出循环

    # Correct the wavelength so that the narrow lines have zero velocity
    dv_sys = m_fit['[O III]'].dv_c.value
    wave_corr = np.array(wave_use) * (1 - dv_sys / ls_km)

    # Calculate the velocity of broad H-alpha and H-beta profiles
    velc_Ha = (wave_corr - wave_dict['Halpha']) / wave_dict['Halpha'] * ls_km
    velc_Hb = (wave_corr - wave_dict['Hbeta']) / wave_dict['Hbeta'] * ls_km

    # Calculate the flux of broad H-alpha and H-beta profiles
    flux_Ha = m_fit[r'H$\alpha$'](wave_use) * m_multi
    flux_Hb = m_fit[r'H$\beta$'](wave_use) * m_multi

    # Calculate the velocity-weighted flux of broad H-alpha and H-beta profiles
    vel_flux_Ha = np.trapz(flux_Ha * (velc_Ha)**2, x=velc_Ha)
    vel_flux_Hb = np.trapz(flux_Hb * (velc_Hb)**2, x=velc_Hb)

    # Calculate the velocity-weighted width of broad H-alpha and H-beta profiles
    width_Ha = np.sqrt(vel_flux_Ha / np.trapz(flux_Ha, x=velc_Ha))
    width_Hb = np.sqrt(vel_flux_Hb / np.trapz(flux_Hb, x=velc_Hb))

    return {'sigma_line_Ha':width_Ha,
             'sigma_line_Hb':width_Hb}
