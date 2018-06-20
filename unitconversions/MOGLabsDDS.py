#####################################################################
#                                                                   #
# MOGLabsXRF421.py                                                  #
#                                                                   #
# Copyright 2017, Monash University                                 #
#                                                                   #
# This file is part of the labscript suite (see                     #
# http://labscriptsuite.org) and is licensed under the Simplified   #
# BSD License. See the license.txt file in the root of the project  #
# for the full license.                                             #
#                                                                   #
#####################################################################

from __future__ import division
from UnitConversionBase import *
import pandas as pd 
import numpy as np 
import scipy.interpolate as interpolate
import os

# Constants
FREQ_BITS = 32
AMPL_BITS = 14
PHASE_BITS = 16
FREQ_MIN = 20.e6
FREQ_MAX = 400.e6 - 0.093

class MOGLabsDDSFreqConversion(UnitConversion):
    # This must be defined outside of init, and must match the default
    # hardware unit specified within the BLACS tab
    base_unit = 'Hz'
    derived_units = ['kHz', 'MHz', 'int']

    def __init__(self, calibration_parameters={}):
        self.parameters = calibration_parameters
        self.freq_min = FREQ_MIN
        self.freq_max = FREQ_MAX
        self.df = FREQ_MAX / (0.4 * (2**(FREQ_BITS) - 1))
        UnitConversion.__init__(self, self.parameters)

    def MHz_to_base(self, MHz):
        return MHz*1.0e6

    def MHz_from_base(self, Hz):
        return Hz/1.0e6

    def kHz_to_base(self, kHz):
        return kHz*1.0e3

    def kHz_from_base(self, Hz):
        return Hz/1.0e3

    def int_to_base(self, i):
        if i > round(FREQ_MAX/self.df):
            raise ValueError('int_to_base was called with too high an integer for this device')
        if i < round(FREQ_MIN/self.df):
            return FREQ_MIN
            # raise ValuError('int_to_base was called with too high an integer for this device')
        return i * self.df

    def int_from_base(self, Hz):
        return round(Hz/self.df)

class MOGLabsDDSAmpConversion(UnitConversion):
    # This must be defined outside of init, and must match the default
    # hardware unit specified within the BLACS tab
    base_unit = 'int'
    derived_units = ['dBm', 'mW', 'Vpp', 'frac']

    def __init__(self, calibration_parameters={'calibration_file': 'MOGLabsDDS.csv', 'channel': 1}):
        self.parameters = calibration_parameters
        self.channel = self.parameters['channel']
        cur_dir = os.path.split(os.path.realpath(__file__))[0]
        # df = pd.read_csv(os.path.join(cur_dir, self.parameters['calibration_file']))
        df_both = pd.read_csv(os.path.join(cur_dir, 'MOGLabsDDS.csv'))
        df = df_both[['CH{} amp'.format(self.channel), 'CH{} dBm'.format(self.channel)]].dropna()
        df.columns = [x.split()[-1] for x in df.columns]
        df['i'] = np.log2(df.amp+1)
        df['mW'] = 10**(df['dBm']/10.)
        self.df = df
        self.dBm_max = df.dBm.max()
        self.dBm_min = df.dBm.min()
        self.amp_min = df.amp.min()
        self.amp_max = df.amp.max()
        self.i_min = df.i.min()
        self.i_max = df.i.max()

        # Create interpolating spline to smooth the data
        self.to_dB = interpolate.interp1d(df.i, df.dBm, kind='linear')
        self.from_dB = interpolate.interp1d(df.dBm, df.i, kind='linear')

        self.Vpp_max = self.Vpp_from_base(self.amp_max)
        self.mW_max = self.mW_from_base(self.amp_max)
        UnitConversion.__init__(self, self.parameters)

    def dBm_to_base(self, dB):
        if dB < self.dBm_min:
            return self.amp_min
        elif dB > self.dBm_max:
            raise ValueError('dBm_to_base was called with too high a dBm for this device.')
        else:
            i = self.from_dB(dB)
            return np.int16(2**i-1)

    def dBm_from_base(self, amp):
        i = np.log2(amp+1)
        if i < self.i_min:
            return self.dBm_min
        elif i > self.i_max:
            raise ValueError('dBm_from_base was called with too high an amplitude ({}) / integer ({}) for this device.'.format(amp, i))
        else:
            return self.to_dB(i)

    def mW_to_base(self, mW):
        dBm = 10*np.log10(mW)
        return self.dBm_to_base(dBm)

    def mW_from_base(self, amp):
        dBm = self.dBm_from_base(amp)
        return 10**(dBm/10.)

    def Vpp_to_base(self, Vpp):
        Vrms = Vpp/(2.*np.sqrt(2.))
        P = Vrms**2/50.
        dBm = 10.*np.log10(1.e3*P)
        return self.dBm_to_base(dBm)

    def Vpp_from_base(self, amp):
        dBm = self.dBm_from_base(amp)
        P = 1.e-3*10.**(dBm/10.)
        Vrms = np.sqrt(50.*P)
        return 2.*np.sqrt(2.)*Vrms

    def frac_to_base(self, f):
        return self.Vpp_to_base(f*self.Vpp_max)

    def frac_from_base(self, amp):
        return self.Vpp_from_base(amp)/self.Vpp_max

    # Optional convenience functions
    def dBm_to_frac(self, dBm):
        return self.frac_from_base(self.dBm_to_base(dBm))


if __name__ == '__main__':
    conv = MOGLabsDDSAmpConversion({'channel': 2})
    # x = 15669
    x = 1807
    i = np.log2(x)
    dBi = conv.to_dB(i)
    print(conv.from_dB(dBi) == i)

    # Plot the discretised amplitudes in Vpp
    import matplotlib.pyplot as plt
    x_list = np.arange(0, conv.amp_max, 2**6)
    Vpp_list = np.array([conv.Vpp_from_base(x) for x in x_list])
    plt.plot(x_list, Vpp_list, ls='-', marker='o', drawstyle='steps-post')
    plt.xlabel('amplitude (integer)')
    plt.ylabel('amplitude (Vpp)')
    plt.show()

    # Plot the discretised amplitudes in dBm
    import matplotlib.pyplot as plt
    x_list = np.arange(0, conv.amp_max, 2**6)
    dBm_list = np.array([conv.dBm_from_base(x) for x in x_list])
    plt.plot(x_list, dBm_list, ls='-', marker='o', drawstyle='steps-post')
    plt.xlabel('amplitude (integer)')
    plt.ylabel('amplitude (dBm)')
    plt.show()

    # Plot the discretised power in mW
    import matplotlib.pyplot as plt
    x_list = np.arange(0, conv.amp_max, 2**6)
    mW_list = np.array([conv.mW_from_base(x) for x in x_list])
    plt.plot(x_list, mW_list, ls='-', marker='o', drawstyle='steps-post')
    plt.xlabel('amplitude (integer)')
    plt.ylabel('amplitude (mW)')
    plt.show()
