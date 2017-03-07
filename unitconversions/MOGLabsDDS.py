#####################################################################
#                                                                   #
# MOGLabsXRF421.py                                                  #
#                                                                   #
# Copyright 2013, Monash University                                 #
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

# Constants
FREQ_BITS = 32
AMPL_BITS = 14
PHASE_BITS = 16
FREQ_MIN = 20.0e6
FREQ_MAX = 400e6

class MOGLabsDDSFreqConversion(UnitConversion):
    # This must be defined outside of init, and must match the default
    # hardware unit specified within the BLACS tab
    base_unit = 'Hz'
    derived_units = ['kHz', 'MHz']

    def __init__(self, calibration_parameters=None):
        self.parameters = calibration_parameters
        UnitConversion.__init__(self, self.parameters)

    def MHz_to_base(self, MHz):
        return MHz*1.0e6

    def MHz_from_base(self, Hz):
        return Hz/1.0e6

    def kHz_to_base(self, kHz):
        return kHz*1.0e3

    def kHz_from_base(self, Hz):
        return Hz/1.0e3

class MOGLabsDDSAmpConversion(UnitConversion):
    # This must be defined outside of init, and must match the default
    # hardware unit specified within the BLACS tab
    base_unit = 'int'
    derived_units = ['dBm', 'Vpp', 'frac']

    def __init__(self, calibration_parameters={'calibration_file': 'C:\\labscript_suite\\labscript_utils\\unitconversions\\MOGLabsDDS.csv'}):
        self.parameters = calibration_parameters
        df = pd.read_csv('C:\\labscript_suite\\labscript_utils\\unitconversions\\MOGLabsDDS.csv')
        # df = pd.read_csv(self.parameters['calibration_file'])
        df.i = np.log2(df.amp+1)
        self.df = df
        self.dBm_max = df.dBm.max()
        self.dBm_min = df.dBm.min()
        self.x_min = df.amp.min()
        self.x_max = df.amp.max()
        self.i_min = df.i.min()
        self.i_max = df.i.max()

        # Create interpolating spline to smooth the data
        self.to_dB = interpolate.interp1d(df.i, df.dBm, kind='linear')
        self.from_dB = interpolate.interp1d(df.dBm, df.i, kind='linear')
        self.Vpp_max = self.Vpp_from_base(2**AMPL_BITS-1)
        UnitConversion.__init__(self, self.parameters)

    def dBm_to_base(self, dB):
        if dB < self.dBm_min:
            return self.x_min
        elif dB > self.dBm_max:
            raise ValueError('dBm_to_base was called with too high a dBm for this device.')
        else:
            i = self.from_dB(dB)
            return np.int16(np.round(2**i-1))

    def dBm_from_base(self, x):
        i = np.log2(x+1)
        if i < self.i_min:
            return self.dBm_min
        elif i > self.i_max:
            raise ValueError('dBm_from_base was called with too high an integer for this device.')
        else:
            return self.to_dB(i)

    def Vpp_to_base(self, Vpp):
        Vrms = Vpp/(2.*np.sqrt(2.))
        P = Vrms**2/50.
        dBm = 10.*np.log10(1.e3*P)
        return self.dBm_to_base(dBm)

    def Vpp_from_base(self, x):
        dBm = self.dBm_from_base(x)
        P = 1.e-3*10.**(dBm/10.)
        Vrms = np.sqrt(50.*P)
        return 2.*np.sqrt(2.)*Vrms

    def frac_to_base(self, f):
        return self.Vpp_to_base(f*self.Vpp_max)

    def frac_from_base(self, x):
        return self.Vpp_from_base(x)/self.Vpp_max

    # Optional convenience functions
    def dBm_to_frac(self, dBm):
        return self.frac_from_base(self.dBm_to_base(dBm))


if __name__ == '__main__':
    conv = MOGLabsDDSAmpConversion()
    # x = 15669
    x = 1807
    i = np.log2(x)
    dBi = conv.to_dB(i)
    print(conv.from_dB(dBi) == i)

    # Plot the first hundred discretised amplitudes in Vpp
    import matplotlib.pyplot as plt
    x_list = np.arange(100)
    Vpp_list = np.array([conv.Vpp_from_base(x) for x in x_list])
    plt.plot(x_list, Vpp_list, ls='-', marker='o', drawstyle='steps-post')
    plt.xlabel('amplitude (integer)')
    plt.ylabel('amplitude (Vpp)')
    plt.show()

    # Plot the first hundred discretised amplitudes in dBm
    import matplotlib.pyplot as plt
    x_list = np.arange(100)
    dBm_list = np.array([conv.dBm_from_base(x) for x in x_list])
    plt.plot(x_list, Vpp_list, ls='-', marker='o', drawstyle='steps-post')
    plt.xlabel('amplitude (integer)')
    plt.ylabel('amplitude (dBm)')
    plt.show()