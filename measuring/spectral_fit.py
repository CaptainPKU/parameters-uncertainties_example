import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from astropy.modeling import models, fitting
import sys
sys.path.append("../..")
import sagan
import matplotlib as mpl


def fit_pg2349(wave_use, flux_use):
    wave_use=np.array(wave_use)
    wave_dict = sagan.utils.line_wave_dict
    label_dict = sagan.utils.line_label_dict

    poly = models.Polynomial1D(degree=2, name='multi')
    pl = models.PowerLaw1D(amplitude=0.0955266, x_0=5500, alpha=1.73025, fixed={'x_0': True, 'amplitude': True, 'alpha': True}, name='Power Law')
    iron = sagan.IronTemplate(amplitude=0.0339, stddev=4109.34, z=0, fixed={'amplitude': True, 'stddev': True}, name='Fe II')

    bounds = {'sigma_w0': (100, 4000)}
    b_ha = sagan.Line_MultiGauss(n_components=2, amp_c=0.229, dv_c=-96, sigma_c=2070, 
                                 wavec=wave_dict['Halpha'], name=label_dict['Halpha'],
                                 amp_w0=0.289, dv_w0=-140, sigma_w0=5556)
    b_hg = sagan.Line_MultiGauss(n_components=1, amp_c=0.0427, dv_c=462, sigma_c=2234, 
                                 wavec=wave_dict['Hgamma'], name=label_dict['Hgamma'])
    b_hb = sagan.Line_MultiGauss(n_components=2, amp_c=0.044, dv_c=-23.66, sigma_c=1567, 
                                 wavec=wave_dict['Hbeta'], name=label_dict['Hbeta'],
                                 amp_w0=1.34, dv_w0=-132, sigma_w0=3000, bounds=bounds)
    
    bounds = {'sigma_c': (100, 4000), 'dv_c': (-500, 500)}
    b_he1 = sagan.Line_MultiGauss(n_components=1, amp_c=0.013, dv_c=-190, sigma_c=2539, wavec=5875.624, name='He I 5876')
    line_s2 = sagan.Line_MultiGauss_doublet(n_components=1, amp_c0=0.038, amp_c1=0.032,
                                            wavec0=wave_dict['SII_6718'], wavec1=wave_dict['SII_6733'], name='[S II]')
    line_n2 = sagan.Line_MultiGauss_doublet(n_components=3, amp_c0=0.15, amp_c1=0.05, dv_c=107, sigma_c=220,
                                            wavec0=wave_dict['NII_6583'], wavec1=wave_dict['NII_6548'], name='[N II]')
    bounds = {'sigma_c': (100, 4000), 'dv_c': (-500, 500), 'dv_w0': (-500, 500), 'dv_w1': (-500, 500),
              'sigma_w0': (100, 4000), 'sigma_w1': (100, 4000)}
    line_o3 = sagan.Line_MultiGauss_doublet(n_components=3, amp_c0=0.2945, amp_c1=0.098, dv_c=-1.26, sigma_c=140, 
                                            wavec0=wave_dict['OIII_5007'], wavec1=wave_dict['OIII_4959'], name='[O III]',
                                            amp_w0=0.13, dv_w0=9, sigma_w0=208,
                                            amp_w1=0.05, dv_w1=-9, sigma_w1=2208, bounds=bounds)
    bounds = {'sigma_c': (100, 4000), 'dv_c': (-500, 500)}
    b_he2 = sagan.Line_MultiGauss(n_components=1, amp_c=0.01156, dv_c=500, sigma_c=2580, 
                                  wavec=wave_dict['HeII_4686'], name=label_dict['HeII_4686'], 
                                  bounds=bounds)
    # Tie the line ratio of [O III] 5007/4959 to 2.98
    def tie_o3(model):
        return model['[O III]'].amp_c0 / 2.98
    line_o3.amp_c1.tied = tie_o3

    def tie_n2(model):
        return model['[N II]'].amp_c0 / 2.96
    line_n2.amp_c1.tied = tie_n2

    # Tie functions
    def tie_narrow_sigma_c(model):
        return model['[O III]'].sigma_c

    def tie_narrow_dv_c(model):
        return model['[O III]'].dv_c

    def tie_narrow_sigma_w0(model):
        return model['[O III]'].sigma_w0

    def tie_narrow_dv_w0(model):
        return model['[O III]'].dv_w0

    def tie_narrow_amp_w0(model):
        return model['[O III]'].amp_w0

    def tie_narrow_sigma_w1(model):
        return model['[O III]'].sigma_w1

    def tie_narrow_dv_w1(model):
        return model['[O III]'].dv_w1

    def tie_narrow_amp_w1(model):
        return model['[O III]'].amp_w1
    line_s2.sigma_c.tied = tie_narrow_sigma_c
    line_s2.dv_c.tied = tie_narrow_dv_c
    line_n2.sigma_c.tied = tie_narrow_sigma_c
    line_n2.dv_c.tied = tie_narrow_dv_c
    line_n2.sigma_w0.tied = tie_narrow_sigma_w0
    line_n2.dv_w0.tied = tie_narrow_dv_w0
    line_n2.amp_w0.tied = tie_narrow_amp_w0
    line_n2.sigma_w1.tied = tie_narrow_sigma_w1
    line_n2.dv_w1.tied = tie_narrow_dv_w1
    line_n2.amp_w1.tied = tie_narrow_amp_w1

    n_hb = sagan.Line_MultiGauss(n_components=3, amp_c=0.086, dv_c=0, sigma_c=200, wavec=wave_dict['Hbeta'], name=f'narrow Hb', amp_w0=0.086, dv_w0=0, sigma_w0=200)
    n_hg = sagan.Line_MultiGauss(n_components=3, amp_c=0.02, dv_c=0, sigma_c=200, wavec=4340, name=f'narrow Hg', amp_w0=0.02, dv_w0=0, sigma_w0=200)
    n_he2 = sagan.Line_MultiGauss(n_components=3, amp_c=0.029, dv_c=0, sigma_c=200, wavec=wave_dict['HeII_4686'], name=f'narrow He2', amp_w0=0.029, dv_w0=0, sigma_w0=200)
    n_o3_4363 = sagan.Line_MultiGauss(n_components=3, amp_c=0.0056, dv_c=0, sigma_c=200, wavec=wave_dict['OIII_4363'], name=f'[O III] 4363', amp_w0=0.0056, dv_w0=0, sigma_w0=200)
    n_ha = sagan.Line_MultiGauss(n_components=3, amp_c=0.16, dv_c=0, sigma_c=200, wavec=wave_dict['Halpha'], name=f'narrow Ha', amp_w0=0.16, dv_w0=0, sigma_w0=200)
    n_o1_6300= sagan.Line_MultiGauss(n_components=3, amp_c=0.016, dv_c=0, sigma_c=200, wavec=6300, name=f'narrow [O I] 6300', amp_w0=0.016, dv_w0=0, sigma_w0=200)
    n_he1_5876 = sagan.Line_MultiGauss(n_components=3, amp_c=0.0094, dv_c=0, sigma_c=200, wavec=5876, name=f'narrow He1 5876', amp_w0=0.0094, dv_w0=0, sigma_w0=200)

    line_he1 =  b_he1 + n_he1_5876
    line_ha  =  b_ha + n_ha
    line_hg  =  b_hg + n_hg
    line_hb  =  b_hb + n_hb
    line_he2 =  b_he2 + n_he2
    m_init = (pl + iron + line_he1 + line_ha + line_s2 + line_n2 + n_o1_6300 + line_hg + n_o3_4363 + line_o3 + line_hb + line_he2) * poly

    for line in [n_ha, n_hb, n_hg, n_he2, n_o3_4363, n_o1_6300, n_he1_5876]:
        line.sigma_c.tied = tie_narrow_sigma_c
        line.dv_c.tied = tie_narrow_dv_c
        line.sigma_w0.tied = tie_narrow_sigma_w0
        line.dv_w0.tied = tie_narrow_dv_w0
        line.amp_w0.tied = tie_narrow_amp_w0
        line.sigma_w1.tied = tie_narrow_sigma_w1
        line.dv_w1.tied = tie_narrow_dv_w1
        line.amp_w1.tied = tie_narrow_amp_w1

    fitter = fitting.LevMarLSQFitter()

    weights = np.ones_like(flux_use)
    fltr1 = (wave_use > 5295) & (wave_use < 5307)
    fltr2 = (wave_use > 6388) & (wave_use < 6398)
    fltr3 = (wave_use > 6470) & (wave_use < 6490)
    fltr11 = (wave_use > 4260) & (wave_use < 4430)
    fltr12 = (wave_use > 4600) & (wave_use < 5120)
    fltr13 = (wave_use > 5550) & (wave_use < 6050)
    fltr14 = (wave_use > 6200) & (wave_use < 6890)
    fltr15 = (wave_use > 7010) & (wave_use < 7500)

    weights[fltr1] = 0.0
    weights[fltr2] = 0.0
    weights[fltr3] = 0.0
    # weights[fltr11] = 0.0
    # weights[fltr12] = 0.0
    # weights[fltr13] = 0.0
    # weights[fltr14] = 0.0
    # weights[fltr15] = 0.0

    m_fit = fitter(m_init, wave_use, flux_use, weights=weights, maxiter=10000)  # Important to set a large maxiter!

    #ax, axr = sagan.plot.plot_fit(wave_use, flux_use, m_fit, weight=weights)
    #ax.set_ylabel(r'$F_\lambda\:(\mathrm{erg\,s^{-1}\,cm^{-2}\,\lambda^{-1}})$', fontsize=24)
    #plt.show()

    #for m in m_fit:
    #    print(m.__repr__())

    return m_fit




def fit_pg2349_only_OIIIcore(wave_use, flux_use):
    #------------------------------------------------------------具体拟合，修改各成分
    wave_use=np.array(wave_use)
    wave_dict = sagan.utils.line_wave_dict
    label_dict = sagan.utils.line_label_dict
    poly = models.Polynomial1D(degree=2, name='multi')  # The name must be `multi`; no more than 5th order
    pl = models.PowerLaw1D(amplitude=0.05508865, x_0=5500, alpha=1.62918538, fixed={'x_0': True,'amplitude':True,'alpha':True},name='Power Law')
    iron = sagan.IronTemplate(amplitude=0.01968473, stddev=4102.913, z=0, name='Fe II',fixed={'amplitude':True,'stddev':True})

    bounds = {'sigma_w0': (100, 4000)}
    b_ha = sagan.Line_MultiGauss(n_components=2, amp_c=0.229, dv_c=-96, sigma_c=2070, 
                                wavec=wave_dict['Halpha'], name=label_dict['Halpha'],
                                amp_w0=0.289, dv_w0=-140, sigma_w0=5556)
    b_hg = sagan.Line_MultiGauss(n_components=1, amp_c=0.0427, dv_c=462, sigma_c=2234, 
                                wavec=wave_dict['Hgamma'], name=label_dict['Hgamma'])
    b_hb = sagan.Line_MultiGauss(n_components=2, amp_c=0.044, dv_c=-23.66, sigma_c=1567, 
                                wavec=wave_dict['Hbeta'], name=label_dict['Hbeta'],
                                amp_w0=1.34, dv_w0=-132, sigma_w0=3000, bounds=bounds)
    bounds = {'sigma_c': (100, 4000), 'dv_c': (-500, 500)}
    b_he1 = sagan.Line_MultiGauss(n_components=1, amp_c=0.013, dv_c=-190, sigma_c=2539, wavec=5875.624, name='He I 5876')
    line_s2 = sagan.Line_MultiGauss_doublet(n_components=1, amp_c0=0.038, amp_c1=0.032,
                                            wavec0=wave_dict['SII_6718'], wavec1=wave_dict['SII_6733'], name='[S II]')
    line_n2 = sagan.Line_MultiGauss_doublet(n_components=1, amp_c0=0.15, amp_c1=0.05, dv_c=107, sigma_c=220,
                                            wavec0=wave_dict['NII_6583'], wavec1=wave_dict['NII_6548'], name='[N II]')
    ##line_o3 = sagan.Line_MultiGauss_doublet(n_components=1, amp_c0=0.32, amp_c1=0.1, dv_c=-10, sigma_c=350, 
    ##                                        wavec0=wave_dict['OIII_5007'], wavec1=wave_dict['OIII_4959'], name='[O III]')
    line_o3 = sagan.Line_MultiGauss_doublet(n_components=2, amp_c0=0.2945, amp_c1=0.098, dv_c=-1.26, sigma_c=240, 
                                            wavec0=wave_dict['OIII_5007'], wavec1=wave_dict['OIII_4959'], name='[O III]',
                                            amp_w0=0.13, dv_w0=91.7, sigma_w0=1738)
    b_he2 = sagan.Line_MultiGauss(n_components=1, amp_c=0.01156, dv_c=500, sigma_c=2580, 
                                wavec=wave_dict['HeII_4686'], name=label_dict['HeII_4686'], 
                                bounds=bounds)

    n_ha = sagan.Line_Gaussian(amplitude=0.13, wavec=wave_dict['Halpha'], name=f'narrow Ha')
    n_o1_6300 = sagan.Line_Gaussian(amplitude=0.013, wavec=6300, name=f'narrow [O I] 6300')
    n_he1_5876 = sagan.Line_Gaussian(amplitude=0.00785, wavec=5876, name=f'narrow He1 5876')
    n_hg = sagan.Line_Gaussian(amplitude=0.015, wavec=wave_dict['Hgamma'], name=f'narrow Hg')
    n_o3_4363 = sagan.Line_Gaussian(amplitude=0.0077, wavec=wave_dict['OIII_4363'], name=f'[O III] 4363')
    n_hb = sagan.Line_Gaussian(amplitude=0.017, wavec=wave_dict['Hbeta'], name=f'narrow Hb')
    n_he2 = sagan.Line_Gaussian(amplitude=0.0086, wavec=wave_dict['HeII_4686'], name=f'narrow He2')

    line_he1 =  b_he1 + n_he1_5876
    line_ha  =  b_ha + n_ha
    line_hg  =  b_hg + n_hg
    line_hb  =  b_hb + n_hb
    line_he2 =  b_he2 + n_he2

    m_init = (pl+iron+line_he1+ line_ha+ line_s2 +line_n2+n_o1_6300 + line_hg+ n_o3_4363+line_o3+line_hb+line_he2 )*poly
    #+line_he1+ line_ha+ line_s2 +line_n2+n_o1_6300 + line_hg+ n_o3_4363+line_o3+line_hb+line_he2

    def tie_o3(model):
        return model['[O III]'].amp_c0 / 2.98
    line_o3.amp_c1.tied = tie_o3

    def tie_n2(model):
        return model['[N II]'].amp_c0 / 2.96
    line_n2.amp_c1.tied = tie_n2

    # Tie the line ratio of [O III] 5007/4959 to 2.98
    def tie_o3(model):
        return model['[O III]'].amp_c0 / 2.98
    line_o3.amp_c1.tied = tie_o3

    #def tie_n2(model):
    #    return model['[N II]'].amp_c0 / 2.96
    #line_n2.amp_c1.tied = tie_n2

    # Tie
    def tie_narrow_sigma(model):
        return model['[O III]'].sigma_c

    def tie_narrow_dv(model):
        return model['[O III]'].dv_c

    line_s2.sigma_c.tied = tie_narrow_sigma
    line_s2.dv_c.tied = tie_narrow_dv
    #line_n2.sigma_c.tied = tie_narrow_sigma
    #line_n2.dv_c.tied = tie_narrow_dv

    for line in [n_ha, n_hb, n_hg, n_he2, n_o3_4363,n_o1_6300,n_he1_5876]:
        line.sigma.tied = tie_narrow_sigma
        line.dv.tied = tie_narrow_dv

    fitter = fitting.LevMarLSQFitter()

    weights = np.ones_like(flux_use)
    fltr1 = (wave_use > 5295) & (wave_use < 5307)
    fltr2= (wave_use > 6388) & (wave_use < 6398)
    fltr3=(wave_use > 6470) & (wave_use < 6490)
    fltr11=(wave_use > 4260) & (wave_use < 4430)
    fltr12=(wave_use > 4600) & (wave_use < 5120)
    fltr13=(wave_use > 5550) & (wave_use < 6050)
    fltr14=(wave_use > 6200) & (wave_use < 6890)
    fltr15=(wave_use > 7010) & (wave_use < 7500)

    weights[fltr1] = 0.0
    weights[fltr2] = 0.0
    weights[fltr3] = 0.0
    #weights[fltr12] = 0.0
    #weights[fltr13] = 0.0
    #weights[fltr14] = 0.0
    #weights[fltr15] = 0.0

    m_fit = fitter(m_init, wave_use, flux_use, weights=weights, maxiter=10000)  # Important to set a large maxiter!

 #   ax, axr = sagan.plot.plot_fit(wave_use, flux_use, m_fit, weight=weights)
 #   ax.set_ylabel(r'$F_\lambda\:(\mathrm{erg\,s^{-1}\,cm^{-2}\,\lambda^{-1}})$', fontsize=24)
 #   plt.show()

 #   for m in m_fit:
 #       print(m.__repr__())
 
    return m_fit