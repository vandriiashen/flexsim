import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

class MaterialHandler(object):
    def __init__(self, energy_bins, mat_config):
        self.energy_bins = energy_bins
        # Attenuation coefficient is in 1/cm, but detector config has distances in mm
        self.units_correction = 0.1
        self.extract_data_from_config(mat_config)
        
    def extract_data_from_config(self, mat_config):
        print(mat_config)
        self.mat_count = mat_config['material_count']
        self.name = ['Background']
        self.lac = [0.]
        self.curve_fname = ['-']
        for i in range(1, self.mat_count+1):
            self.name.append(mat_config['name_{}'.format(i)])
            self.lac.append(mat_config['lac_{}'.format(i)])
            self.curve_fname.append(mat_config['curve_{}'.format(i)])
    
    def get_curve(self, fname):
        nist_data = np.loadtxt(fname)
        f_interp = interp1d(nist_data[:,0], nist_data[:,1], kind="linear")
        curve = np.zeros((self.energy_bins))
        curve[2:] = f_interp(np.arange(2,self.energy_bins))
        curve[0] = curve[2]
        curve[1] = curve[2]
        return curve
    
    def get_material_curve(self, num):
        '''
        if num == 1:
            # this is added to make absorption more similar to real data
            arbitrary_density_coefficient = 0.8
            return self.get_curve("../Materials/playdoh.txt") * self.units_correction * arbitrary_density_coefficient
        if num == 2:
            return self.get_curve("../Materials/silicate.txt") * self.units_correction
        '''
        attenuation_curve = self.get_curve(self.curve_fname[num])
        res = attenuation_curve * self.units_correction
        return res
        
    def spectrum_generate(self, voltage):
        coeff = np.loadtxt('../Materials/tasmip_coeff.txt')
        spectrum = np.zeros((self.energy_bins))
        for i in range(self.energy_bins):
            if i < voltage:
                spectrum[i] = coeff[i][0] + coeff[i][1] * voltage + coeff[i][2] * voltage**2 + coeff[i][3] * voltage**3
                if spectrum[i] < 0:
                    spectrum[i] = 0
        spectrum /= np.sum(spectrum)
        return spectrum
    
    def get_monochromatic_intensity(self, num, voltage):
        lac = self.lac[num]
        return lac
        
    def show_material_curve(self):
        x_range = np.arange(0,self.energy_bins)
        curve1 = self.get_material_curve(1)
        curve2 = self.get_material_curve(2)
        
        plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.tick_params(labelsize = 24)
        
        plt.loglog(x_range, curve1, label="Approximate play-doh attenuation")
        plt.loglog(x_range, curve2, label="Silicate")
        
        plt.grid()
        plt.legend(prop={'size': 24})
        plt.xlabel("Energy, keV", size = 36)
        plt.ylabel("LAC, 1/cm", size = 36)
        plt.tight_layout()
        plt.show()
        
    def show_monochromatic(self):
        # from 30 to 90 kV
        x_range = np.arange(30, 90)
        
        curve1 = [self.get_monochromatic_intensity(1, x) for x in x_range]
        curve2 = [self.get_monochromatic_intensity(2, x) for x in x_range]
        
        plt.figure(figsize=(12,8))
        ax = plt.axes()
        ax.tick_params(labelsize = 24)
        
        plt.plot(x_range, curve1, label="Approximate play-doh attenuation")
        plt.plot(x_range, curve2, label="Silicate")
        
        plt.grid()
        plt.legend(prop={'size': 24})
        plt.xlabel("Voltage, kV", size = 36)
        plt.ylabel("LAC, 1/cm", size = 36)
        plt.tight_layout()
        plt.show()
