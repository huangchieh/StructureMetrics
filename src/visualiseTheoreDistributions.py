#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH, compute_sg_sk_all
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from water import cal_all_hydrogen_bonds

from utils import plot_kde_fill, plot_joint_distribution
from utils import plot_joint_distributions
from utils import plot_joint_distributions_in_row

import seaborn as sns
from scipy.stats import gaussian_kde

if __name__ == '__main__':
    inputFolder = '../data/structures/simulations/'
    processedFolder = '../processed_data/theory_distributions/'
    os.makedirs(processedFolder, exist_ok=True)
    baseOut = '../results/theoretical_distributions/'
    structures = ['Label']
    show = False
    
    for structure in structures:
        # Create the output folders
        figureOut = os.path.join(baseOut, structure)
        os.makedirs(figureOut, exist_ok=True)
        # Processed data output
        npzOut = os.path.join(processedFolder, structure)
        os.makedirs(npzOut, exist_ok=True)

        # Read the samples
        sampleFolder = os.path.join(inputFolder, structure) 
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()

        # Common parameters for the plots, and output folder
        r_max = 3.5
        #mic = True if structure == 'Label' else False
        mic = False
        bins = 120
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)
        alpha = 0.3

        # O-O distances 
        z_thresholds = {'All': None, 'Top': True, 'Bottom': False}
        colors = {'All': 'green', 'Top': 'red', 'Bottom': 'black'}
        linestypes = {'All': '-', 'Top': '-', 'Bottom': '-'}
        fills = {'All': True, 'Top': False, 'Bottom': False}
        for key, value in z_thresholds.items():
            label = 'OO_distances_{}_{}'.format(key, structure)
            legend = '{}'.format(key) 
            npzFile = '{}/OO_distances_{}.npz'.format(npzOut, key) 
            if os.path.exists(npzFile):
                print('Loading OO_distances from file: {}'.format(npzFile))
                OO_distances = np.load(npzFile)['distances']
            else:
                print('Calculating OO_distances ...')
                OO_distances = mean_rdf(samples, 'O', 'O', r_max=r_max, mic=mic, aboveZthres=value, onlyDistances=True)
                np.savez(npzFile, distances=OO_distances, r_max=r_max)
            if key != 'All':
                axs[0].hist(OO_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            #sns.kdeplot(OO_distances, ax=axs[0], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill(ax=axs[0], xmin=0, xmax=r_max, data=OO_distances, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=alpha)
        axs[0].set_xlabel(r'$r_\text{OO}$ (Å)')
        axs[0].set_ylabel(r'$\rho(r)$')
        axs[0].set_ylim(0, 4.8)
        axs[0].set_xlim(2.2, r_max-0.25)
        axs[0].legend(frameon=False, ncol=1)
        axs[0].tick_params(axis='both', direction='in')

        # O-H distances
        r_max = 1.25
        for key, value in z_thresholds.items():
            label = 'OH_distances_{}_{}'.format(key, structure)
            legend = 'OH {} ({})'.format(key, structure) if structure != 'P' else 'OH {} (Reference)'.format(key)
            npzFile = '{}/OH_distances_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading OH_distances from file: {}'.format(npzFile))
                OH_distances = np.load(npzFile)['distances']
            else:
                print('Calculating OH_distances ...')
                OH_distances = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, aboveZthres=value, onlyDistances=True)
                np.savez(npzFile, distances=OH_distances, r_max=r_max)
            if key != 'All':
                axs[1].hist(OH_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            #sns.kdeplot(OH_distances, ax=axs[1], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill(ax=axs[1], data=OH_distances, xmin=0, xmax=r_max, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        axs[1].set_xlabel(r'$r_\text{OH}$ (Å)')
        axs[1].set_ylabel(r'$\rho(r)$')
        #axs[1].set_ylim(0, 120)
        axs[1].set_xlim(0.95, 1.05)
        axs[1].tick_params(axis='both', direction='in')

        # H-O-H angles
        print('Plotting H-O-H ...')
        firstTwo = False
        onlyAngle = True
        r_max = 1.25
        y_lim = 0.4
        for key, value in z_thresholds.items():
            label = "HOH_dist_{}_{}".format(key, structure)
            legend = 'HOH {} ({})'.format(key, structure) if structure != 'P' else 'HOH {} (Reference)'.format(key)
            npzFile = '{}/HOH_distances_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading HOH_distances from file: {}'.format(npzFile))
                angles = np.load(npzFile)['angles']
            else:
                print('Calculating HOH_distances ...')
                angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle, aboveZthres=value)
                np.savez(npzFile, angles=angles)
            if key != 'All':
                axs[2].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            #sns.kdeplot(angles, ax=axs[2], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill(ax=axs[2], data=angles, xmin=0, xmax=180, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        axs[2].set_xlabel(r'$\angle$HOH (degrees)')
        axs[2].set_xlim(98, 112)
        axs[2].set_ylabel(r'$\rho(\theta)$')
        axs[2].tick_params(axis='both', direction='in')

        # ZOH angles 
        r_max = 1.25
        y_lim = 0.04
        bins = 120 
        for key, value in z_thresholds.items():
            label = "Theta_OH_dist_{}_{}".format(key, structure)
            legend = r"$\theta_{{\text{{OH}}}}$ {} ({})".format(key, structure) if structure != 'P' else r"$\theta_{{\text{{OH}}}}$ {} (Reference)".format(key)
            npzFile = '{}/Theta_OH_distances_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading Theta_OH_distances from file: {}'.format(npzFile))
                angles = np.load(npzFile)['angles']
            else:
                print('Calculating Theta_OH_distances ...')
                angles = mean_adf_OH(samples, r_max=r_max, firstTwo=False, mic=False, onlyAngle=True, aboveZthres=value)
                np.savez(npzFile, angles=angles)
            if key != 'All': 
                axs[3].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            #sns.kdeplot(angles, ax=axs[3], linewidth=1, label=legend, bw_adjust=0.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
            plot_kde_fill(ax=axs[3], data=angles, xmin=0, xmax=180, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        axs[3].set_xlabel(r'$\angle$ZOH (degrees)')
        axs[3].set_xlim(0, 180)
        axs[3].set_ylabel(r'$\rho(\theta)$')
        axs[3].tick_params(axis='both', direction='in')

        fig.subplots_adjust(hspace=0.2, wspace=0.2, left=0.1, bottom=0.1, right=0.95, top=0.98)
        plt.savefig('{}/RDF_ADF_{}.pdf'.format(figureOut, structure))
        plt.savefig('{}/RDF_ADF_{}.png'.format(figureOut, structure), dpi=300)
        plt.savefig('{}/RDF_ADF_{}.svg'.format(figureOut, structure))
        if show: plt.show()
        plt.close() 

        ##################
        # Order parameter
        ##################
        r_max = 3.5
        sgs, sks = compute_sg_sk_all(samples, r_max=r_max)
        x_min, y_min = sgs.min(), sks.min()
        x_max, y_max = 1+0.1, 1+0.0008
        # Plot All, Top, and Bottom separately 
        for k, (key, value) in enumerate(z_thresholds.items()):
            npzFile = '{}/OrderParameters_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile): 
                print('Loading OrderParameters from file: {}'.format(npzFile))
                sgs, sks = np.load(npzFile)['sgs'], np.load(npzFile)['sks']
            else:
                print('Calculating OrderParameters ...')
                sgs, sks = compute_sg_sk_all(samples, r_max=r_max, aboveZthres=value)
                np.savez(npzFile, sgs=sgs, sks=sks)
            
            xs, ys = sgs, sks
            x_label, y_label = r'$S_g$', r'$S_k$'
            image_prefix = f"{figureOut}/OrderParameters_{structure}_{key}"
            text = key
            plot_joint_distribution(xs, ys, x_min, x_max, y_min, y_max, x_label,
                                    y_label, image_prefix, text, show)

        # Plot All, Top, and Bottom in one figure
        npz_prefix = f"{npzOut}/OrderParameters"
        x_max, y_max = 1+0.1, 1+0.0008
        npz_x, npz_y = 'sgs', 'sks'
        image_prefix = f"{figureOut}/OrderParameters_{structure}_overlay"
        plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, text, show)

        # Plot distributions side by side
        x_max, y_max = 1, 1
        image_prefix = f"{figureOut}/OrderParameters_{structure}_row"
        plot_joint_distributions_in_row(z_thresholds, npz_prefix, npz_x, npz_y, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, text, show)


        ##################
        # H-bonds
        ##################
        hbonds_ = cal_all_hydrogen_bonds(samples)
        distances_ = np.array([hb[3] for hb in hbonds_])
        angles_ = np.array([hb[4] for hb in hbonds_])
        x_min, y_min = distances_.min(), angles_.min()
        x_max, y_max = 3.5, 180
        x_label, y_label = '$d_{OO}$ (Å)', r'$\angle$DHA (°)'
        # Plot All, Top, and Bottom separately
        for k, (key, value) in enumerate(z_thresholds.items()):
            npzFile = '{}/Hbonds_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile):
                print('Loading Hbonds from file: {}'.format(npzFile))
                hbonds = np.load(npzFile)['hbond']
                #distances = np.load(npzFile)['distance']
                #angles = np.load(npzFile)['angle']
            else:
                print('Calculating Hbonds ...')
                hbonds = cal_all_hydrogen_bonds(samples, aboveZthres=value, zThresholdO=4.85)
                distances_da = [hb[3] for hb in hbonds]
                angles_dha = [hb[4] for hb in hbonds]
                OO_OHO = np.array([distances_da, angles_dha]).T
                np.savez(npzFile, hbond=hbonds, distance=distances_da, 
                                  angle=angles_dha, OO_OHO=OO_OHO)
            xs = np.array([hb[3] for hb in hbonds])
            ys = np.array([hb[4] for hb in hbonds])
            image_prefix = f"{figureOut}/Hbonds_{structure}_{key}"
            text = key
            plot_joint_distribution(xs, ys, x_min, x_max, y_min, y_max, x_label,
                                    y_label, image_prefix, text, show)
        # Plot All, Top, and Bottom in one figure
        npz_prefix = f"{npzOut}/Hbonds"
        #x_max, y_max = 1+0.1, 1+0.0008
        npz_x, npz_y = 'distance', 'angle'
        image_prefix = f"{figureOut}/Hbonds_{structure}_overlay"
        plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, text, show)
        
        # Plot distributions side by side
        #x_max, y_max = 1, 1
        image_prefix = f"{figureOut}/Hbonds_{structure}_row"
        plot_joint_distributions_in_row(z_thresholds, npz_prefix, npz_x, npz_y, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, text, show)
