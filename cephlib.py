# Disable all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import glob as glob
from pathlib import Path
import pandas as pd
import time
import tarfile
import urllib.request

from astropy.visualization import simple_norm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from mpl_toolkits import axes_grid1

from astropy.wcs import WCS
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import QTable
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip


import numpy as np
from scipy.optimize import curve_fit


from tqdm.notebook import tqdm

from photutils.aperture import SkyCircularAperture, CircularAperture
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import IterativePSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground, MedianBackground, BiweightLocationBackground
from photutils.datasets import make_model_sources_image
from photutils.psf import SourceGrouper

import webbpsf
from webbpsf.utils import to_griddedpsfmodel

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Set environmental variables
os.environ["WEBBPSF_PATH"] = "/home/localuser/Documents/LIU/webbpsf-data"
os.environ["PYSYN_CDBS"] = "/home/localuser/Documents/LIU/grp/redcat/trds"

# WEBBPSF Data
boxlink = 'https://stsci.box.com/shared/static/34o0keicz2iujyilg4uz617va46ks6u9.gz'                                                           
boxfile = '/home/localuser/Documents/LIU/webbpsf-data/webbpsf-data-1.0.0.tar.gz'
synphot_url = 'http://ssb.stsci.edu/trds/tarfiles/synphot5.tar.gz'
synphot_file = '/home/localuser/Documents/LIU/synphot5.tar.gz'

webbpsf_folder = "/home/localuser/Documents/LIU/webbpsf-data"
synphot_folder = '/home/localuser/Documents/LIU/grp'

# Gather webbpsf files
psfExist = os.path.exists(webbpsf_folder)
if not psfExist:
    os.makedirs(webbpsf_folder)
    urllib.request.urlretrieve(boxlink, boxfile)
    gzf = tarfile.open(boxfile)
    gzf.extractall(webbpsf_folder)

# Gather synphot files
synExist = os.path.exists(synphot_folder)
if not synExist:
    os.makedirs(synphot_folder)
    urllib.request.urlretrieve(synphot_url, synphot_file)
    gzf = tarfile.open(synphot_file)
    gzf.extractall('/home/localuser/Documents/LIU/')

# Filter data
filters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W2', 'F150W', 'F162M', 'F164N', 'F182M',
           'F187N', 'F200W', 'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F322W2', 'F323N',
           'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M']

psf_fwhm = [0.987, 1.103, 1.298, 1.553, 1.628, 1.770, 1.801, 1.494, 1.990, 2.060, 2.141, 2.304, 2.341, 1.340,
            1.444, 1.585, 1.547, 1.711, 1.760, 1.830, 1.901, 2.165, 2.179, 2.300, 2.302, 2.459, 2.507, 2.535, 2.574]

zp_modA = [25.7977, 25.9686, 25.8419, 24.8878, 27.0048, 25.6536, 24.6957, 22.3073, 24.8258, 22.1775, 25.3677, 24.3296,
           22.1036, 22.7850, 23.5964, 24.8239, 23.6452, 25.3648, 20.8604, 23.5873, 24.3778, 23.4778, 20.5588,
           23.2749, 22.3584, 23.9731, 21.9502, 20.0428, 19.8869, 21.9002]

zp_modB = [25.7568, 25.9771, 25.8041, 24.8738, 26.9821, 25.6279, 24.6767, 22.2903, 24.8042, 22.1499, 25.3391, 24.2909,
           22.0574, 22.7596, 23.5011, 24.6792, 23.5769, 25.3455, 20.8631, 23.4885, 24.3883, 23.4555, 20.7007,
           23.2763, 22.4677, 24.1562, 22.0422, 20.1430, 20.0173, 22.4086]

dict_utils = {filters[i]: {'psf fwhm': psf_fwhm[i], 'VegaMAG zp modA': zp_modA[i],
                           'VegaMAG zp modB': zp_modB[i]} for i in range(len(filters))}

vega_zp = {"F200W": 757.65e-6, "F150W": 1172.06e-6} #MJy


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def clip_percentile(dt, percentile):
    arr = dt["op_fluxes"] - dt["ip_fluxes"]
    dt["arr"] = arr
    lower = np.percentile(arr, percentile)
    upper = np.percentile(arr, 100 - percentile)
    clipped = dt[(dt["arr"] > lower) & (dt["arr"] < upper)]
    ded = dt[~((dt["arr"] > lower) & (dt["arr"] < upper))]
    return clipped, ded

def clip_sigma(x, y, y_err, sigma):
    dt = pd.DataFrame({"x": x, "y": y, "y_err": y_err})
    dev = dt["y"] - dt["x"]
    upper = np.percentile(arr, 100 - percentile)
    clipped = dt[(dt["arr"] > lower) & (dt["arr"] < upper)]
    ded = dt[~((dt["arr"] > lower) & (dt["arr"] < upper))]
    return clipped, ded


class ceph_phot:

    def __init__(self, fits_file, coord_table, target, n_stars = None, r=0.10):
        r = r * u.arcsec
        self.fits_file = Path(fits_file)
        self.coord_table = Path(coord_table)
        self.n_stars = n_stars
        self.maxiters = 20
        self.fitter_maxiters = 500
        self.fit_shape = [7,7]
        self.target= target
        with fits.open(self.fits_file) as f:
            self.wcs = WCS(f[1].header)
            self.pix_ar = f[1].header["PIXAR_SR"]
            self.data = f[1].data
            self.filt = f[0].header['FILTER']
            self.chan = f[0].header['CHANNEL']
        if self.chan == 'LONG':
            self.det = 'NRCB5'
        elif self.chan == 'SHORT':
            self.det = 'NRCB1'
        self.ref = vega_zp[self.filt]
        self.get_aper(r)
        try:
            self.LogP = self.datatable["LogP"]
        except:
            self.LogP = np.log10(self.datatable["period"])


    def calc_corr(self):
        if self.target == "N5584":
            cat_pos = pd.DataFrame({"ra": [215.6143, 215.59372, 215.59988, 215.60519, 215.60307, 215.61027, 215.6014],
                                    "dec": [-0.38678, -0.373, -0.39316, -0.38368, -0.38336, -0.38824, -0.37362]})
            real_pos = pd.DataFrame({"ra": [215.6143475, 215.5937652, 215.5999219, 215.6052277, 215.6031153, 215.6103173, 215.6014448],
                                    "dec": [-0.3865981, -0.3728233, -0.3929806, -0.3834965, -0.3831804, -0.3880587, -0.3734415]})
            return np.mean(cat_pos["ra"] - real_pos["ra"]), np.mean(cat_pos["dec"] - real_pos["dec"])
        elif self.target == "N4258":
            cat_pos = pd.DataFrame({"ra": [184.711655, 184.690048, 184.6987, 184.696487, 184.730682],
                                    "dec": [47.35149, 47.332642, 47.35606, 47.333092, 47.338249]})
            real_pos = pd.DataFrame({"ra": [184.7117847, 184.6901881, 184.6988428, 184.6966229, 184.7308296],
                                    "dec": [47.3512716, 47.3324212, 47.3558388, 47.3328693, 47.3380255]})
            return np.median(cat_pos["ra"] - real_pos["ra"]), np.median(cat_pos["dec"] - real_pos["dec"])
        elif self.target == "N1365":
            return 0,0  #no correction needed, checked
        else:
            return 0,0
    
    def correct_ICRS(self):
        cra, cdec = self.calc_corr()
        new_pos = self.datatable.copy()
        new_pos["ra"] = new_pos["ra"] - cra
        new_pos["dec"] = new_pos["dec"] - cdec
        self.datatable = new_pos
    
    def get_aper(self, r, p=None, skiprows=0):
        dt = pd.read_csv(self.coord_table, header=0, skiprows=skiprows)
        datatable = dt[dt["Host"] == self.target]
        if self.n_stars:
            datatable = datatable.head(self.n_stars)
        self.n_stars = datatable.shape[0]
        if p:
            p_l = p-1
            p_h = p+1
            datatable = datatable[(10**datatable["LogP"] < p_h) & (10**datatable["LogP"] > p_l)]
        datatable.reset_index(inplace=True)
        self.datatable = datatable
        self.correct_ICRS()
        positions = SkyCoord(ra = self.datatable["ra"] * u.deg, dec = self.datatable["dec"] * u.deg, unit = "deg")
        self.sky_apertures = SkyCircularAperture(positions, r=r)
        self.pix_apertures = self.sky_apertures.to_pixel(self.wcs)

    def get_psf(self, outfile="output/psf_model.fits", remake=False):
        outfile_ac = outfile[:-5] + "_" + self.det.lower() + ".fits"
        if os.path.exists(outfile_ac) and not remake:
            self.psf_model = to_griddedpsfmodel(outfile_ac)
        else:
            nc = webbpsf.NIRCam()
            nc.filter = self.filt
            nc.detector = self.det
            psf_grid = nc.psf_grid(num_psfs=1, oversample = 4, all_detectors=False, use_detsampled_psf=False, save=True, outfile=outfile, verbose=False)
            self.psf_model = psf_grid

        # self.psf_model.data[0] = gaussian_filter(self.psf_model.data[0], 2.5)

    def plot_psf(self, save=False):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        data = self.psf_model.data[0]
        data /= np.max(data)
        # im = ax.imshow(self.psf_model.data[0], cmap="gist_heat", origin='lower', norm=simple_norm(self.psf_model.data, 'sqrt', percent=99.3))
        im = ax.imshow(data, cmap="gist_heat", origin='lower', norm=LogNorm(vmin=1e-5, vmax=1))
        add_colorbar(im)
        ax.set_title(f"PSF model for {self.filt} filter")
        if save:
            plt.savefig(f"output/{self.fits_file.stem}_psf.png", dpi=150)
        plt.show()

    def psf_phot(self, data_patch, *, init_params, th):
        r_pix = self.pix_apertures.r
        bkgrms = MADStdBackgroundRMS()
        mmm_bkg = MMMBackground()
        local_bkg_est = LocalBackground(r_pix+2, r_pix+4, BiweightLocationBackground())
        sigma_psf = dict_utils[self.filt]['psf fwhm']
        fit_shape = np.array(self.fit_shape)
        std = bkgrms(data_patch)
        bkg = mmm_bkg(data_patch)
        daofind = DAOStarFinder(threshold=th * std, fwhm=sigma_psf)
        psf_model = self.psf_model.copy()
        data_bkg = data_patch - bkg
        tic = time.perf_counter()
        phot = IterativePSFPhotometry(psf_model=psf_model, finder=daofind,
                                                fitter=LevMarLSQFitter(),
                                                localbkg_estimator= None,
                                                fitter_maxiters=self.fitter_maxiters,
                                                maxiters= self.maxiters,
                                                fit_shape=self.fit_shape, aperture_radius=r_pix)
        result = phot(data_bkg, init_params = init_params)
        model_image = phot.make_model_image(data_patch.shape, [50,50])
        residual_image = data_patch - model_image        #?????
        tab = result.to_pandas()
        toc = time.perf_counter()
        return tab, residual_image, bkg
    
    def test_crowding_empty(self):
        data_patch = np.zeros((50, 50))
        flux = 10
        ceph_tab = QTable()
        ceph_tab["flux_fit"] = [flux]
        ceph_tab["x_fit"] = [25]
        ceph_tab["y_fit"] = [25]
        b = self.crowding_corr(data_patch, ceph_tab, th=3, n_sample=100, verbose=True)
        print(f"Crowding bias on empty image with Cepheid flux assumed {flux} = {b}")

    def test_crowding_single(self):
        data_patch = np.zeros((50, 50))
        flux = 10
        sources = QTable()
        sources['flux'] = [flux]
        sources['x_0'] = [25]
        sources['y_0'] = [25]
        test_stars = make_model_sources_image([50,50], self.psf_model, sources)
        data_star = data_patch + test_stars
        ceph_tab = QTable()
        ceph_tab["flux_fit"] = [flux]
        ceph_tab["x_fit"] = [25]
        ceph_tab["y_fit"] = [25]
        b = self.crowding_corr(data_star, ceph_tab, th=3, n_sample=100, verbose=True)
        print(f"Crowding bias on image with single Cepheid with flux {flux} = {b}")
    
    def crowding_corr(self, data_patch, ceph_tab, th, n_sample=100, verbose=True, id=None):
        shape = data_patch.shape
        ceph_flux = ceph_tab["flux_fit"][0]
        if n_sample % 3 ==1:
            n_sample += 2
        elif n_sample % 3 == 2:
            n_sample += 1
        x_range = np.concatenate([np.linspace(8, ceph_tab["x_fit"][0] - 5, 1000) , np.linspace(ceph_tab["x_fit"][0] + 5, shape[1]-8, 1000)])
        y_range = np.concatenate([np.linspace(8, ceph_tab["y_fit"][0] - 5, 1000) , np.linspace(ceph_tab["y_fit"][0] + 5, shape[0]-8, 1000)])
        xs = np.random.choice(x_range, n_sample)
        ys = np.random.choice(y_range, n_sample)
        fluxes = np.concatenate([np.ones(int(n_sample/3)) * ceph_flux*0.8, np.ones(int(n_sample/3)) * ceph_flux*0.9, np.ones(int(n_sample/3)) *ceph_flux])
        ip_fluxes = []
        op_fluxes = []
        op_flux_err = []
        for i in range(n_sample):
            sources = QTable()
            sources['flux'] = [fluxes[i]]
            sources['x_0'] = xs[i]
            sources['y_0'] = ys[i]
            test_stars = make_model_sources_image(shape, self.psf_model, sources)
            data_star = data_patch + test_stars
            init_params = QTable()
            init_params['x'] =sources['x_0']
            init_params['y'] = sources['y_0']
            tab, residual_image, bkg = self.psf_phot(data_star, init_params=init_params, th=th)
            if (i in [0, 10, 30, 50]) and verbose:
                self.plot_im_res(data_star, residual_image, bkg, tab, id)
            s_tab = tab[tab["iter_detected"] == 1]
            if (s_tab["flags"][0] == 0) or True:
                ip_fluxes.append(fluxes[i])
                op_fluxes.append(s_tab["flux_fit"][0])
                op_flux_err.append(s_tab["flux_err"][0])

        
        #make linear fit
        tab = pd.DataFrame({"ip_fluxes": ip_fluxes, "op_fluxes": op_fluxes, "op_flux_err": op_flux_err})
        tab.dropna(inplace=True)
        dev = tab["op_fluxes"] - tab["ip_fluxes"]
        sc_tab, ded = clip_percentile(tab, 15)
        # try:
        #     sc_tab, ded = clip_percentile(tab, 15)
        # except:
        #     print("clipping failed")
        #     sc_tab = tab
        #     ded = tab.copy().iloc[0:0]
        sc_dev = sc_tab["op_fluxes"] - sc_tab["ip_fluxes"]
        op_flux_err = sc_tab["op_flux_err"]
        b = np.median(sc_dev)
        b_err = (np.median(((sc_dev - b)/op_flux_err)**2)/np.sum(1/op_flux_err**2))**0.5
        print(f"{dev.mean() = }, {dev.std() = }, {sc_dev.mean() = }, {sc_dev.std() = }, {b = }, {b_err = }") 
        if verbose:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(sc_tab["ip_fluxes"], sc_tab["op_fluxes"], "bo", label = "Data Points used for fit")
            ax.plot(ded["ip_fluxes"], ded["op_fluxes"], "ro", label = "Data Points excluded from fit")
            x = np.linspace(ceph_flux*0.8, ceph_flux, 3)
            y = x + b
            ax.plot(x, y, "-", label = rf"y = x + {b:.3f} $\pm$ {b_err:.3f}")
            ax.plot(x, x, "--", label = "y = x")
            ax.set_xlabel("True flux", fontsize = 20)
            ax.set_ylabel("Measured flux", fontsize = 20)
            ax.legend()
            plt.show()

        return b, b_err
    
    def do_photometry(self, th = 3, do_crowd = True, ap_size = 15, verbose=True):
        self.do_crowd = do_crowd
        xs = self.pix_apertures.positions.transpose()[0]
        ys = self.pix_apertures.positions.transpose()[1]
        ids = self.datatable["ID"]
        self.patch_list = []
        self.bkg_vals = []
        self.psf_table_list = []
        self.res_list = []
        self.crowding_list = []
        self.crowding_err = []
        self.surface_brightness = []
        for id, x, y in tqdm(zip(ids, xs, ys), unit="stars", total=self.n_stars):
            xi = int(x)
            yi = int(y)
            init_params = QTable()
            init_params['x'] = [x-xi + ap_size]
            init_params['y'] = [y-yi + ap_size]
            data_patch = self.data[yi-ap_size:yi+ap_size, xi-ap_size:xi+ap_size]
            psf_table, res_im, bkg_value = self.psf_phot(data_patch, init_params=init_params, th=th)
            ceph_tab = psf_table[psf_table["iter_detected"] == 1]
            if self.do_crowd:
                b, b_err = self.crowding_corr(data_patch, ceph_tab, th, verbose=verbose, id=id)
                self.crowding_list.append(b)
                self.crowding_err.append(b_err)
            self.psf_table_list.append(psf_table)
            self.res_list.append(res_im)
            self.patch_list.append(data_patch)
            self.bkg_vals.append(bkg_value)
            self.surface_brightness.append(np.sum(data_patch))
        self.crowding_list = np.array(self.crowding_list)
        self.crowding_err = np.array(self.crowding_err)


    def calc_mag(self, flux, flux_err):
        y = -2.5*np.log10(flux*self.pix_ar/self.ref)
        y_err = np.abs(2.5/(np.log(10)*flux))*flux_err
        return y, y_err
    def results(self, out_folder="output"):
        out_folder = Path(out_folder)

        self.full_table = self.psf_table_list[0]
        for tab in self.psf_table_list[1:]:
            self.full_table = pd.concat([self.full_table, tab])
        self.full_table.reset_index(inplace=True)
        self.full_table.to_excel(out_folder/(self.fits_file.stem+"_psftable.xlsx"), index = False)

        out_table_name = f"{self.fits_file.stem}_out.xlsx"
        short_table = self.full_table[self.full_table["iter_detected"] == 1][["x_init", "y_init", "flux_init", "x_fit", "y_fit", "flux_fit", "x_err", "y_err", "flux_err", "flags"]]
        short_table["mag"], short_table["mag_err"] = self.calc_mag(short_table["flux_fit"], short_table["flux_err"])
        short_table.reset_index(inplace=True)
        short_table["LogP"] = np.array(self.LogP)
        short_table["ID"] = self.datatable["ID"]
        short_table["ra"] = self.datatable["ra"]
        short_table["dec"] = self.datatable["dec"]
        short_table["ra_pix"] = self.pix_apertures.positions.transpose()[0]
        short_table["dec_pix"] = self.pix_apertures.positions.transpose()[1]

        fitted_positions = np.array([short_table["x_fit"], short_table["y_fit"]]).transpose()
        t_ap = CircularAperture(fitted_positions, r=5)
        t_ap = t_ap.to_sky(self.wcs)
        short_table["ra_fit"] = t_ap.positions.ra.deg
        short_table["dec_fit"] = t_ap.positions.dec.deg
        if self.crowding_list.size != 0:
            short_table["b"] = self.crowding_list
            short_table["b_e"] = self.crowding_err
            y = short_table["flux_fit"]
            b = short_table["b"]
            be = short_table["b_e"]
            dy = short_table["flux_err"]
            x = (y - b)
            dx = (dy**2 + be**2)**0.5
            short_table["flux_corr"] = x
            short_table["flux_corr_err"] = dx
            short_table["mag_corr"], short_table["mag_corr_err"] = self.calc_mag(short_table["flux_corr"], short_table["flux_corr_err"])
            short_table["surface_brightness"] = np.array(self.surface_brightness)
        self.short_table = short_table
        short_table.to_excel(out_folder/ out_table_name, index=False)
        return out_table_name

    #plotting functions
    def plot_im_res(self, data_p, res, bkg_value, psf_tab, id=None):
        psf_tab = psf_tab[psf_tab["iter_detected"] == 1]
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        im0 = ax[0].imshow(data_p, origin='lower', norm=simple_norm(data_p - res, 'sqrt', percent=99))
        cc = Circle([psf_tab["x_init"],psf_tab["y_init"]], 2, color="r", fill=False)
        ccf = Circle([psf_tab["x_fit"],psf_tab["y_fit"]], 2, color="b", fill=False)
        ax[0].add_patch(cc)
        ax[0].add_patch(ccf)
        ax[0].set_title('Data')
        add_colorbar(im0)

        im1 = ax[1].imshow(data_p - res, origin='lower', norm=simple_norm(data_p - res, 'sqrt', percent=99))
        cc = Circle([psf_tab["x_init"],psf_tab["y_init"]], 2, color="r", fill=False)
        ccf = Circle([psf_tab["x_fit"],psf_tab["y_fit"]], 2, color="b", fill=False)
        ax[1].add_patch(cc)
        ax[1].add_patch(ccf)
        ax[1].set_title('Model')
        add_colorbar(im1)

        im2 = ax[2].imshow(res, origin='lower', norm=simple_norm(data_p -res, "sqrt", percent=99))
        cc = Circle([psf_tab["x_init"],psf_tab["y_init"]], 2, color="r", fill=False)
        ccf = Circle([psf_tab["x_fit"],psf_tab["y_fit"]], 2, color="b", fill=False)
        ax[2].add_patch(cc)
        ax[2].add_patch(ccf)
        ax[2].set_title('Residual Image')
        add_colorbar(im2)

        plt.suptitle(f" ID: {id}, Background value: {bkg_value:.3f} \n initial position: red, fitted position: blue")
        plt.show()

    def plot_residuals(self, n=None):
        if n is None:
            n = self.n_stars
        for i in tqdm(range(n), total=n):
            self.plot_im_res(self.patch_list[i], self.res_list[i], self.bkg_vals[i], self.psf_table_list[i], self.datatable["ID"][i])

    def image_plot_stars(self, plot_aper=True):
        plt.figure(29, dpi=500)
        img = plt.imshow(self.data, cmap='viridis', norm = simple_norm(self.data, 'sqrt', percent=99.9), origin="lower")
        cbar = plt.colorbar(img)
        cbar.set_label("Intensity (Arbritary units)")
        if(plot_aper):
            self.pix_apertures.plot(color="red")
        plt.show()
    
    def zoom_images_all(self, ap_size=25):
        cols = 6
        rows = int(np.ceil(self.n_stars/cols))
        # Adjust the space between the subplots
        plt.subplots_adjust(wspace=.1, hspace=.1)
        # Loop through the rows and columns to create individual subplots
        for i in tqdm(range(rows)):
            fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(5*cols, 5), sharex=True, sharey=True)
            axes = axes.flatten()
            for j in range(cols):
                n = i*cols + j
                if n>=self.n_stars:
                    break
                x = self.pix_apertures.positions[n][0]
                y = self.pix_apertures.positions[n][1]
                xi = int(x)
                yi = int(y)
                data_patch = self.data[yi-ap_size:yi+ap_size, xi-ap_size:xi+ap_size]
                xo = x-xi + ap_size
                yo = y-yi + ap_size
                star_origin = [xo,yo]
                
                cc = Circle(star_origin, 2, color="tomato", fill=False, lw=2)
                try:
                    norm = simple_norm(data_patch, 'sqrt', percent=99.9)
                    im = axes[j].imshow(data_patch, origin='lower', norm=norm)
                except:
                    im = axes[j].imshow(data_patch, origin='lower', vmax=1.8)
                
                axes[j].add_patch(cc)
                ra = self.datatable["ra"][n]
                dec = self.datatable["dec"][n]
                logperi = self.LogP[n]
                id = self.datatable["ID"][n]
                axes[j].set_title(f'{ra = :.8f} {dec = :.8f} \n logp = {logperi:.3f} {id =}')
                # plt.colorbar(im, ax = axes[i])
            plt.show()

class ceph_plot:
    def __init__(self, fits_file, table_name, plot_crowd = True, fit_curve = True):
        self.table_name = Path(table_name)
        self.fits_file = Path(fits_file)
        op_dir = Path("output")
        self.table = pd.read_excel(op_dir/self.table_name)
        self.table = self.table[self.table["flags"] == 0]
        self.table.dropna(subset=["mag", "mag_err", "mag_corr", "mag_corr_err", "b", "b_e"], inplace=True)
        self.fits_file = Path(fits_file)
        LogP = self.table["LogP"]
        x = LogP - 1
        y, y_err = self.table["mag"], self.table["mag_err"]
        if plot_crowd:
            y_c, y_err_c = self.table["mag_corr"], self.table["mag_corr_err"]
        n_row = 1
        n_col = 2
        ymin = np.min([y.min(), y_c.min()]) if plot_crowd else y.min()
        ymax = np.max([y.max(), y_c.max()]) if plot_crowd else y.max()
        ylim = [ymin -2, ymax + 2]
        fig, axs = plt.subplots(n_row, n_col, figsize=(12*n_col, 6*n_row))
        _ = self.plot_one(x, y, y_err, ylim, axs[0], "Apparent magnitude", fit_curve)
        if plot_crowd:
            _ = self.plot_one(x, y_c, y_err_c, ylim, axs[1], "Crowding corrected magnitude", fit_curve)
        plt.savefig(op_dir / f"{self.fits_file.stem}_plot_PL.png", dpi = 150)
        plt.show()
        if plot_crowd:
            self.plot_crowding()


    
    def plot_one(self, x, y, y_err, ylim, ax, label, fit_curve = True):
        ax.errorbar(x, y, y_err, fmt="ro", label=label)
        ax.set_xlabel("Log(P/days) - 1", fontsize = 20)
        ax.set_ylabel("mag", fontsize = 20)
        if fit_curve:
            m = -3.2
            def linear(x, c):
                return m*x + c
            opt, cov = curve_fit(linear, x, y, p0 = 25, sigma=y_err, absolute_sigma=False, check_finite=False, nan_policy="omit")
            c = opt[0]
            m = m
            c_err = np.sqrt(np.diag(cov))[0]
            xf = np.array([min(x), max(x)])
            yf = m*xf +c
            yf_up = m*xf +c +3*c_err
            yf_down = m*xf +c -3*c_err
            ax.plot(xf, yf, "c-", label=f" y = {m:.5g}*x + {c:.5g} $\pm$ {c_err:.5g}")
            # ax.plot(xf, yf_up, "c--", label = "$\pm 3\sigma$ lines")
            # ax.plot(xf, yf_down, "c--")
            ax.fill_between(xf, yf_up, yf_down, color="c", alpha=0.2, label = "$\pm 3\sigma$ confidence interval")
            ax.set_ylim(ylim)
        ax.legend()
        ax.invert_yaxis()
        return ax
    
    def plot_crowding(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        b_list = self.table["b"]
        b_err_list = self.table["b_e"]
        ax.errorbar(self.table["flux_fit"], b_list, yerr=b_err_list, fmt="ro", label = "Crowding bias")
        ax.axhline(np.nanmedian(b_list), color="black", lw=1, label = f"Median: {np.nanmedian(b_list):.3f}")
        ax.set_xlabel("Flux", fontsize = 20)
        ax.set_ylabel("Crowding bias", fontsize = 20)
        ax.legend()
        plt.show()
        # plot histogram and median of b
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(b_list, bins=100, color="skyblue", edgecolor="black")
        ax.axvline(np.nanmedian(b_list), color="black", lw=1, label = f"Median: {np.nanmedian(b_list):.3f}")
        ax.set_xlabel("Crowding bias", fontsize = 20)
        ax.set_ylabel("Frequency", fontsize = 20)
        ax.legend()
        plt.show()