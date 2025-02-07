import numpy as np
import sagan

def calculate_equivalent_widths_and_integrated_fluxes(wave_use, m_fit):
    wave_dict = sagan.utils.line_wave_dict

    # 初始化 m_multi
    m_multi = np.ones_like(wave_use)
    for m in m_fit:
        if hasattr(m, 'name') and m.name == 'multi':
            m_multi = m(wave_use)
            break  # 找到 'multi' 后退出循环

    # Calculate the flux density of broad Ha, Hb, narrow Ha, Hb, [O III], Fe II and Power Law
    flux_b_Ha = m_fit[r'H$\alpha$'](wave_use) * m_multi
    flux_b_Hb = m_fit[r'H$\beta$'](wave_use) * m_multi
    flux_n_Ha = m_fit['narrow Ha'](wave_use) * m_multi
    flux_n_Hb = m_fit['narrow Hb'](wave_use) * m_multi
    flux_n_OIII = m_fit['[O III]'](wave_use) * m_multi
    flux_FeII = m_fit['Fe II'](wave_use) * m_multi
    flux_PL = m_fit['Power Law'](wave_use) * m_multi

    # Calculate the EW of emission lines: broad Ha, Hb, narrow Ha, Hb, [O III], Fe II
    # During the calculation of FeII EW, the wavelength range is 4434-4684
    EW_b_Ha = np.trapz(flux_b_Ha / flux_PL, wave_use)
    EW_b_Hb = np.trapz(flux_b_Hb / flux_PL, wave_use)
    EW_n_Ha = np.trapz(flux_n_Ha / flux_PL, wave_use)
    EW_n_Hb = np.trapz(flux_n_Hb / flux_PL, wave_use)
    EW_n_OIII = np.trapz(flux_n_OIII / flux_PL, wave_use)

    # 找到波长数组中对应的索引
    array_wave_use = np.array(wave_use)
    wave_FeII_index = np.where((array_wave_use >= 4434) & (array_wave_use <= 4684))[0]

    # 选择积分区间内的波长和通量
    wave_use_subset = array_wave_use[wave_FeII_index]
    flux_FeII_subset = flux_FeII[wave_FeII_index]
    flux_PL_subset = flux_PL[wave_FeII_index]

    # 计算等效宽度
    EW_FeII = np.trapz(flux_FeII_subset / flux_PL_subset, wave_use_subset)

    # Calculate the integrated flux of: broad Ha, Hb, narrow Ha, Hb, [O III], Fe II
    flux_int_b_Ha = np.trapz(flux_b_Ha, wave_use)
    flux_int_b_Hb = np.trapz(flux_b_Hb, wave_use)
    flux_int_n_Ha = np.trapz(flux_n_Ha, wave_use)
    flux_int_n_Hb = np.trapz(flux_n_Hb, wave_use)
    flux_int_n_OIII = np.trapz(flux_n_OIII, wave_use)
    flux_int_FeII = np.trapz(flux_FeII, wave_use)

    results = {
        'EW': {
            'b_Ha': EW_b_Ha,
            'b_Hb': EW_b_Hb,
            'n_Ha': EW_n_Ha,
            'n_Hb': EW_n_Hb,
            'n_OIII': EW_n_OIII,
            'FeII': EW_FeII
        },
        'integrated_flux': {
            'b_Ha': flux_int_b_Ha,
            'b_Hb': flux_int_b_Hb,
            'n_Ha': flux_int_n_Ha,
            'n_Hb': flux_int_n_Hb,
            'n_OIII': flux_int_n_OIII,
            'FeII': flux_int_FeII
        }
    }

    return results
