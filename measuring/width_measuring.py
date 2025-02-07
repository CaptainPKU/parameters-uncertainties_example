import numpy as np
import matplotlib.pyplot as plt
import sagan
def calculate_broad_line_widths(wave_use, m_fit):
    '''
    计算宽线的宽度
    wave_use: 使用的波长
    m_fit: 拟合的模型
    return: 宽线的宽度
    '''
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

    # 计算峰值
    peak_Ha = np.max(flux_Ha)
    peak_Hb = np.max(flux_Hb)

    # 计算不同高度的值
    height_1_4_Ha = peak_Ha / 4
    height_1_2_Ha = peak_Ha / 2
    height_3_4_Ha = 3 * peak_Ha / 4

    height_1_4_Hb = peak_Hb / 4
    height_1_2_Hb = peak_Hb / 2
    height_3_4_Hb = 3 * peak_Hb / 4

    # 找到这些高度对应的速度
    def find_width(velc, flux, height):
        indices = np.where(flux >= height)[0]
        if len(indices) < 2:
            return None  # 如果没有找到两个边缘点，返回 None
        return velc[indices[-1]] - velc[indices[0]]

    # 计算宽度
    fwqm_1_4_Ha = find_width(velc_Ha, flux_Ha, height_1_4_Ha)
    fwhm_Ha = find_width(velc_Ha, flux_Ha, height_1_2_Ha)
    fwqm_3_4_Ha = find_width(velc_Ha, flux_Ha, height_3_4_Ha)

    fwqm_1_4_Hb = find_width(velc_Hb, flux_Hb, height_1_4_Hb)
    fwhm_Hb = find_width(velc_Hb, flux_Hb, height_1_2_Hb)
    fwqm_3_4_Hb = find_width(velc_Hb, flux_Hb, height_3_4_Hb)

    return {
        'Halpha': {
            'FWQM_1_4': fwqm_1_4_Ha,
            'FWHM': fwhm_Ha,
            'FWQM_3_4': fwqm_3_4_Ha
        },
        'Hbeta': {
            'FWQM_1_4': fwqm_1_4_Hb,
            'FWHM': fwhm_Hb,
            'FWQM_3_4': fwqm_3_4_Hb
        }
    }
