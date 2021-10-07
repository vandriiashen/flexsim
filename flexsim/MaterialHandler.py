import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

class MaterialHandler(object):
    '''Class providing material properties.
    
    :param energy_bins: Number of energy bins (step of 1 kV) used for polychromatic simulation
    :type config: :class:`int`
    :param mat_config: Dictionary of material properties that contains a number of materials and 3 parameters for every material: name, linear attenuation coefficient (for monochromatic simulation) and a path to attenuation curve file (for polychromatic simulation)
    :type config: :class:`dict`
    
    '''
    def __init__(self, energy_bins, mat_config):
        '''Constructor method.
        '''
        self.energy_bins = energy_bins
        # Attenuation coefficient is in 1/cm, but detector config has distances in mm
        self.units_correction = 0.1
        self.extract_data_from_config(mat_config)
        
    def extract_data_from_config(self, mat_config):
        '''Builds lists of material parameters based on the config.
        The expected structure of the config:
        material_count = N
        ...
        name_i = <material name>
        lac_i = <LAC value for a certain voltage>        
        curve_i = <Path to a file containing attenuation curve>
        ...
        
        :param mat_config: Dictionary of material properties
        :type config: :class:`dict`
        
        '''
        self.mat_count = mat_config['material_count']
        self.name = ['Background']
        self.lac = [0.]
        self.curve_fname = ['-']
        for i in range(1, self.mat_count+1):
            self.name.append(mat_config['name_{}'.format(i)])
            self.lac.append(mat_config['lac_{}'.format(i)])
            self.curve_fname.append(mat_config['curve_{}'.format(i)])
    
    def get_curve(self, fname):
        '''Rebins attenuation curve to the number of bins specified in the class instance.
        
        :param fname: Path to the file containing attenuation curve
        :type fname: :class:`string`
        :return: Rebinned attenuation curve
        :rtype: :class:`list`
        
        '''
        nist_data = np.loadtxt(fname)
        f_interp = interp1d(nist_data[:,0], nist_data[:,1], kind="linear")
        curve = np.zeros((self.energy_bins))
        curve[2:] = f_interp(np.arange(2,self.energy_bins))
        curve[0] = curve[2]
        curve[1] = curve[2]
        return curve
    
    def get_material_curve(self, num):
        '''Returns attenuation curve corresponding to a certain material.
        
        :param num: Material number
        :type num: :class:`int`
        :return: Attenuation curve
        :rtype: :class:`list`
        
        '''
        attenuation_curve = self.get_curve(self.curve_fname[num])
        res = attenuation_curve * self.units_correction
        return res
        
    def spectrum_generate(self, voltage):
        '''Generates a spectrum corresponding to a certain tube voltage.
        
        :param voltage: Tube voltage
        :type voltage: :class:`float`
        :return: Tube spectrum
        :rtype: :class:`list`
        
        '''
        coeff = np.loadtxt('../Materials/tasmip_coeff.txt')
        spectrum = np.zeros((self.energy_bins))
        for i in range(self.energy_bins):
            if i < voltage:
                spectrum[i] = coeff[i][0] + coeff[i][1] * voltage + coeff[i][2] * voltage**2 + coeff[i][3] * voltage**3
                if spectrum[i] < 0:
                    spectrum[i] = 0
        spectrum /= np.sum(spectrum)
        return spectrum
    
    def get_monochromatic_intensity(self, num):
        '''Gets material's attenuation coefficient based on the config. Attenuation curve is not used.
        
        :param num: Material number
        :type num: :class:`int`
        :return: LAC value
        :rtype: :class:`float`
        
        '''
        lac = self.lac[num]
        return lac
        
    def show_material_curve(self):
        '''Test function to show attenuation curves .     
        '''
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
