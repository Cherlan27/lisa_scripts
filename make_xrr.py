import numpy
import h5py
import sys
from PIL import Image
import pandas
import os

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

from eigene.abs_overlap_fit_poly import Absorber
from eigene.fio_reader import read
from eigene.p08_detector_read import p08_detector_read

class make_xrr():
    def __init__(self,
                 data_directory = "./raw",
                 use_flatfield = True,
                 use_mask = None,
                 experiment = "timing",
                 detector = "lambda",
                 
                 scan_numbers = list(range(1803,1811+1)),
                 
                 detector_orientation = "vertical",
                 footprint_correct = True,
                 beam_width = 20e-6, 
                 sample_length = 81.4e-3,
                 
                 roi = (14,85,60,22),
                 roi_offset = 30,
                 
                 calculate_abs = True,
                 
                 monitor = "ion2",
                 primary_intensity = "normalized",
                 nom_scan_numbers = list(range(1803,1811+1)),
                 auto_cutoff = [0.015, 0.003],
                 auto_cutoff_nom = [0.0184, 0.0013],
                 scan_number_primary = 1137,
                 
                 qc = 0.025,    
                 roughness = 2.55, 
                 
                 save_results = True,
                 show_plot = True,
                 save_plot = True,
                 
                 out_data_directory = "./processed/",
                 out_user_run = "2021-04",
                 out_experiment = "rbbr",
                 out_typ_experiment = "test_5mol", 
                 ):
        
        self.data_directory = data_directory
        self.use_flatfield = use_flatfield,
        self.use_mask = use_mask,
        self.flatfield = "./Module_2017-004_GaAs_MoFluor_Flatfielddata.tif",
        self.pixel_mask = "./Module_2017-004_GaAs_mask.tif",
        self.experiment = experiment,
        self.detector = detector,
        
        self.scan_numbers = scan_numbers,
        
        self.detector_orientation = detector_orientation,
        self.footprint_correct = footprint_correct,
        self.beam_width = beam_width, 
        self.sample_length = sample_length,
        self.wl = 12.38/18 * 1e-10
        
        self.roi = roi,
        self.roi_offset = roi_offset,
        
        # absorber factors:
        self.calculate_abs = calculate_abs,           
        self.absorber_factors = {1: 12.617,
                            2: 11.0553,
                            3: 11.063,
                            4: 11.048,
                            5: 11.7,
                            6: 12.813},
        
        self.monitor = monitor,
        self.primary_intensity = primary_intensity
        self.nom_scan_numbers = nom_scan_numbers,
        self.auto_cutoff = auto_cutoff,
        self.auto_cutoff_nom = auto_cutoff_nom,
        self.scan_number_primary = scan_number_primary,
        
        self.qc = qc,    
        self.roughness = roughness, 
        
        # output data:
        self.save_results = save_results,
        self.show_plot = show_plot,
        self.save_plot = save_plot,
        
        # results are written to this file:
        self.out_data_directory = out_data_directory,
        self.out_user_run = out_user_run,
        self.out_experiment = out_experiment,
        self.out_typ_experiment = out_typ_experiment, 
      
    def __call__(self, scan_number):
        self.scan_numbers = scan_number
        self.get_scans(self.scan_numbers)
        
    def set_roi(self, roi, roi_offset = 30):
        self.roi = roi
        self.roi_offset = roi_offset
        
        
    def abs_fac(self, abs_val):
        abs_val = int(abs_val + 0.2)
        if abs_val == 0:
            return 1.0
        else:
            return self.absorber_factors[abs_val] * self.abs_fac(abs_val - 1)

    def fresnel(self, qc, qz, roughness=2.5):
        """
        Calculate the Fresnel curve for critical q value qc on qz assuming
        roughness.
        """
        return (numpy.exp(-qz**2 * roughness**2) *
                abs((qz - numpy.sqrt((qz**2 - qc**2) + 0j)) /
                (qz + numpy.sqrt((qz**2 - qc**2)+0j)))**2)

    def footprint_correction(self, q, intensity, b, L, wl = 68.88e-12):
        """ 
        Input:
            q [A^(-1)]: Inverse wavevector q_z
            intensity [a.u.]: reflected intensity
            b [mm]: beam width
            L [mm]: sample length
            wl [A^(-1)]: wavelength of X-ray             
        Output:
            intensity2[a.u.]: Corrected intensity
        """
        q_b = (4*numpy.pi/wl*b/L)*10**(-10)
        print(q_b)
        intensity2 = intensity
        print(q[len(q)-1] > q_b)
        i = 0
        for i in range(0,len(q),1):
            if q[i] < q_b:
                print(i)
                print(q[i])
                intensity2[i] = intensity2[i]/(q[i]/q_b)
                i += 1
        else:
            None
        return intensity2, q_b
    
    def use_flatfield(self):
        flatfield_2 = numpy.ones((516,1556))
        flatfield = numpy.array(Image.open(self.flatfield))
        return flatfield
        
    def use_mask(self ):
        file_mask = h5py.File(self.mask, "r")
        img_mask = numpy.array(file_mask["/entry/instrument/detector/data"])[0]
        file_mask.close()
        mask = numpy.zeros_like(img_mask)
        mask[(img_mask > 1)] = 1
        mask = (mask == 1)
        mask_value = 0
        return mask

    def prep_data_structure(self):
        # == prepare data structures
        self.intensity = numpy.array([])
        self.e_intensity = numpy.array([])
        self.qz = numpy.array([])
        
        # == make data
        self.absorbers = Absorber()
        self.temp_intens = {}
        self.temp_e_intens = {}
        
    def normalisation(self):
        # == normalize
        if self.primary_intensity == "auto":
            primary = self.intensity[(self.qz > self.auto_cutoff[0]) & (self.qz<(self.qc - self.auto_cutoff[1]))].mean()
        elif self.primary_intensity == "scan":
            # data for normalization 
            # load scan
            fio_filename = "{0}/{1}_{2:05}.fio".format(self.data_directory, self.experiment[0], scan_number)
            header, column_names, data, scan = read(fio_filename)
            # load monitor
            s_moni = data[self.monitor]
            s_alpha = data["alpha_pos"]
            s_beta = data["beta_pos"]
            s_qz = ((4 * numpy.pi / self.wl) *
                    numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
            # prepare data structures
            s_intens = []
            s_e_intens = []
            # load detector data
            detector_images = p08_detector_read(self.data_directory, self.experiment, self.scan_number_primery, self.detector)()
            n_images = detector_images.shape[0]
            
            img = detector_images[0]
            s_intens, s_e_intens = self.specular_calculation(img)
            
            norm_intens = numpy.array(s_intens)*Absorber.absorbers(list(self.temp_e_intens.values())[0][0])
            norm_e_intens = numpy.array(s_e_intens)*Absorber.absorbers(list(self.temp_e_intens.values())[0][0])
            primary = norm_intens[0]
        elif settings["primary_intensity"] == "normalized":
            primary_ref = intensity[(qz > settings["auto_cutoff"][0]) & (qz<(settings["qc"] - settings["auto_cutoff"][1]))].mean()
            primary = normalisator(settings["nom_scan_numbers"], settings["qc"], settings["footprint_correct"], settings["beam_width"], settings["sample_length"], settings["auto_cutoff_nom"])
            print(primary_ref/primary)
        else:
            primary = self.primary_intensity
        return primary
        
    def get_scans(self, scan_numbers):
        scan_intens = []
        scan_e_intens = []
        temp_intens = {}
        temp_e_intens = {}
        for scan_number in scan_numbers:
            print(scan_numbers)
            fio_filename = "{0}/{1}_{2:05}.fio".format(self.data_directory, self.experiment[0], scan_number)
            self.header, self.column_names, self.data, self.scan_cmd = read(fio_filename)
            self.s_moni = self.data[self.monitor]
            s_alpha = self.data["alpha_pos"]
            s_beta = self.data["beta_pos"]
            s_qz = ((4 * numpy.pi / self.wl) *
                    numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
            
            detector_images = p08_detector_read(self.data_directory, self.experiment, scan_number, self.detector)()
            n_images = detector_images.shape[0]
            n_points = min(n_images, len(s_alpha))
            images_intens, images_e_intens = self.get_images(detector_images, n_points)
            scan_intens.append(images_intens)
            scan_e_intens.append(images_e_intens)
            temp_int_values, temp_int_e_values = self.prep_absorber_calculation(s_qz, images_intens, images_e_intens)
            temp_intens[self.scan_number] = temp_int_values
            temp_e_intens[self.scan_number] = temp_int_e_values
        return s_qz, temp_intens, temp_e_intens
            
    def get_images(self, detector_images, n_points):
        images_intens = []
        images_e_intens = []
        for n in range(n_points):
            img = detector_images[n]
        
            # flatfield correction
            if self.use_flatfield == True:
                flatfield = self.flatfield()
                img = img / flatfield
            
            if self.use_mask == True:
                mask, mask_value = self.use_mask()
                img[mask] = mask_value
            p_intens, p_e_intens = self.specular_calculation(img, n = n)
            images_intens.append(p_intens)
            images_e_intens.append(p_e_intens)
        return images_intens, images_e_intens
            
    def prep_absorber_calculation(self, s_qz, s_intens, s_e_intens, scan_number):
        if self.calculate_abs == True:
            Absorber.absorbers.add_dataset(["abs"], s_qz, s_intens)
            temp_int_value = int(self.header["abs"]+0.1), s_intens
            temp_int_e_value = int(self.header["abs"]+0.1), s_e_intens
        elif self.calculate_abs == None:
            temp_int_value = self.s_intens * self.abs_fac(self.header["abs"])
            temp_int_e_value = self.s_e_intens * self.abs_fac(self.header["abs"])
        return temp_int_value, temp_int_e_value
            
    def absorber_calculation(self, temp_intens, temp_e_intens):
        if self.calculate_abs == True:
            Absorber.calculate_from_overlaps()
            intensity = numpy.concatenate([Absorber.absorbers(x[0])*x[1] for x in list(self.temp_intens.values())])
            e_intensity = numpy.concatenate([Absorber.absorbers(x[0])*x[1] for x in list(self.temp_e_intens.values())])
        else:
            intensity = list(self.temp_intens.values())
            e_intensity = list(self.temp_e_intens.values())
        return intensity, e_intensity


    def background_calculation(self, img):
        if self.detector_orientation == "horizontal":            
            p_bg0 = img[self.roi[1]:(self.roi[1]+self.roi[3]),
                        (self.roi[0]+self.roi[2]+self.roi_offset):(self.roi[0]+2*self.roi[2]+self.roi_offset)].sum()
            p_bg1 = img[self.roi[1]:(self.roi[1]+self.roi[3]),
                        (self.roi[0]-self.roi[2]-self.roi_offset):(self.roi[0]-self.roi_offset)].sum()            
        elif self.detector_orientation == "vertical":            
            p_bg0 = img[(self.roi[1]+self.roi[3]+self.roi_offset):(self.roi[1]+2*self.roi[3]+self.roi_offset),
                        (self.roi[0]):(self.roi[0]+self.roi[2])].sum()
            p_bg1 = img[(self.roi[1]-self.roi[3]-self.roi_offset):(self.roi[1]-self.roi_offset),
                        (self.roi[0]):(self.roi[0]+self.roi[2])].sum()  
        return p_bg0, p_bg1


    def specular_calculation(self, img, n = 0):
        p_specular = img[self.roi[1]:(self.roi[1]+self.roi[3]),self.roi[0]:(self.roi[0]+self.roi[2])].sum() 
        p_bg0, p_bg1 = self.background_calculation(img)
        p_intens = ((p_specular - (p_bg0 + p_bg1) / 2.0) / self.s_moni[n])
        
        if self.monitor == "Seconds":
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / self.s_moni[n])
        else:    
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / self.s_moni[n] 
                         + abs (0.1 * (p_specular - (p_bg0 + p_bg1) / 2.0) / self.s_moni[n]))
        return p_intens, p_e_intens

    def xrr_calculate(self):
        
        qz, s_intens_temp, s_e_intens = self.get_scans(self.scan_numbers)
        if self.footprint_correct == True:
                temp_intensities = self.s_intens
                temp_intensities, q_b = self.footprint_correction(self.s_qz, temp_intensities, self.beam_width, self.sample_length)
                s_intens = temp_intensities
        intensity, e_intensity = self.absorber_calculation(s_intens_temp, s_e_intens)
        
        # == sort data points
        m = numpy.argsort(qz)
        qz = qz[m]
        primary = self.normalisation()
        
        intensity = intensity[m]
        e_intensity = e_intensity[m]
        intensity_norm = intensity/primary
        e_intensity_norm = e_intensity/primary
        
        
def saving_xrr_data(self):
    # == check if path exist or create path
    if self.save_plot or self.save_results:
        file_path = "{0}/reflectivity/".format(self.out_data_directory)
        try:
            os.makedirs(file_path)
        except OSError:
            if not os.path.isdir(file_path):
                raise

    # == save settings dict to file
        setting_save = True
        out_setting_filename = file_path + "/{out_experiment}_{out_typ_experiment}_settings.txt".format(**settings)
        if os.path.exists(out_setting_filename):
            setting_save = input("Settings .txt output file already exists. Overwrite? [y/n] ") == "y"
        if setting_save == True:
            f = open(out_setting_filename,"w")
            f.write( str(settings) )
            f.close()
      
    # == save data to file
    if settings["save_results"]:
        out_filename = file_path + "/{out_experiment}_{out_typ_experiment}.dat".format(**settings)
        df = pandas.DataFrame()
        df["//qz"] = qz
        df["intensity_normalized"] = intensity / primary
        df["e_intensity_normalized"] = e_intensity / primary
        if os.path.exists(out_filename):
            settings["save_results"] = input("Results .dat output file already exists. Overwrite? [y/n] ") == "y"
        if settings["save_results"]:
            df.to_csv(out_filename, sep="\t", index=False)

def plot_xrr(self):
    fig = plt.figure()
    fig.patch.set_color("white")
    ax = fig.gca()
    ax.set_yscale('log', nonposy='clip')
    ax.errorbar(self.qz, self.intensity/self.primary, yerr=self.e_intensity/self.primary, ls='none',
                marker='o', mec='#cc0000', mfc='white', color='#ee0000',
                mew=1.2)
    ax.errorbar(self.qz, self.fresnel(self.qc, self.qz, self.roughness), ls='--', c='#424242')
    ax.set_xlabel('q$_z$')
    ax.set_ylabel('R')
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.subplots_adjust(bottom=0.12)
    if self.show_plot:
        plt.show()
    if self.save_plot:
        out_plotname = file_path + "/{out_experiment}_{out_typ_experiment}.png".format(**settings)
        if os.path.exists(out_plotname):
            self.save_plot = input("plot output file already exists. Overwrite? [y/n] ") == "y"
        if self.save_plot:
            plt.savefig(out_plotname, dpi=300)
          
def plot_xrr_crit(self):
    fig2 = plt.figure()
    fig2.patch.set_color("white")
    ax2 = fig2.gca()
    plt.xlim([0.01,0.04])
    plt.ylim([0.05,1.4])
    #ax2.set_yscale('log', nonposy='clip')
    ax2.errorbar(self.qz, self.intensity/self.primary, yerr=self.e_intensity/self.primary, ls='none',
                marker='o', mec='#cc0000', mfc='white', color='#ee0000',
                mew=1.2)
    ax2.errorbar(self.qz, self.fresnel(self.qc, self.qz, self.roughness), ls='--', c='#424242')
    ax2.set_xlabel('q$_z$')
    ax2.set_ylabel('R')
    ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    
    plt.show()
    
xrr_scan = make_xrr()
xrr_scan([1803,1811])