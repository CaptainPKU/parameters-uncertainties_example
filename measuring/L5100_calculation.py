import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

def calculate_L5100(zred, m_fit, wave_use):
    # 计算光度距离，并转化为厘米单位
    luminosity_distance = cosmo.luminosity_distance(zred)
    luminosity_distance_cm = luminosity_distance.to(u.cm)
    
    # 目标波长
    wave_target = 5100
    
    # 找到最接近目标波长的索引
    closest_index = np.abs(np.array(wave_use) - wave_target).argmin()
    
    # 获取 5100 Å 处的函数值
    flux_at_5100 = (m_fit(wave_use))[closest_index]
    
    # 计算 L_5100
    L_5100 = 4 * np.pi * (luminosity_distance_cm.value ** 2) * flux_at_5100
    
    return L_5100