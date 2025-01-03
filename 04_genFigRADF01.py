#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

if __name__ == '__main__':
    baseOut = 'Figures'
    structures = ['Label']
    
    for structure in structures:
        # Create the output folders
        os.makedirs(os.path.join(baseOut, structure), exist_ok=True)

        # Read the samples
        sampleFolder = os.path.join('BatchOutStructures', structure) if (structure == 'Label' or structure == 'P') else os.path.join('BatchOutStructures', structure, 'Prediction_c')
        samples = read_samples_from_folder(sampleFolder)
        print('Calculating for structure: {}'.format(structure))

        fig, axs = plt.subplots(4, 1, figsize=(6, 6))
        axs = axs.flatten()

        # Common parameters for the plots, and output folder
        r_max = 3.5
        #mic = True if structure == 'Label' else False
        mic = False
        bins = 120
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)

        # --- RDF 
        z_thresholds = {'All': None, 'Top': True, 'Bottom': False}
        colors = {'All': 'green', 'Top': 'red', 'Bottom': 'black'}
        linestypes = {'All': '-', 'Top': ':', 'Bottom': '-.'}
        fills = {'All': True, 'Top': True, 'Bottom': False}
        for key, value in z_thresholds.items():
            OO_distances = mean_rdf(samples, 'O', 'O', r_max=3.5, mic=mic, aboveZthres=value, onlyDistances=True)
            label = 'OO_distances_{}_{}'.format(key, structure)
            legend = 'OO {} ({})'.format(key, structure) if structure != 'P' else 'OO {} (Reference)'.format(key)
            np.savez('{}/OO_distances_{}.npz'.format(outputFolder, key), distances=OO_distances, r_max=r_max)
            if key != 'Bottom':
                axs[0].hist(OO_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            sns.kdeplot(OO_distances, ax=axs[0], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
        axs[0].set_xlabel('r (Å)')
        axs[0].set_ylabel(r'$\rho_\text{OO}(r)$')
        #axs[0].set_ylim(0, 4.8)
        axs[0].set_xlim(0, r_max)
        axs[0].legend(frameon=False, ncol=1)
        axs[0].tick_params(axis='both', direction='in')

        # O-H all water molecules
        r_max = 1.25
        for key, value in z_thresholds.items():
            OH_distances = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, aboveZthres=value, onlyDistances=True)
            label = 'OH_distances_{}_{}'.format(key, structure)
            legend = 'OH {} ({})'.format(key, structure) if structure != 'P' else 'OH {} (Reference)'.format(key)
            #ylim = 150 if structure == 'Label' else 15
            np.savez('{}/OH_distances_{}.npz'.format(outputFolder, key), distances=OH_distances, r_max=r_max)
            if key != 'Bottom':
                axs[1].hist(OH_distances, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            sns.kdeplot(OH_distances, ax=axs[1], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
        axs[1].set_xlabel('r (Å)')
        axs[1].set_ylabel(r'$\rho_\text{OH}(r)$')
        #axs[1].set_ylim(0, 120)
        axs[1].set_xlim(0.9, r_max-0.15)
        axs[1].tick_params(axis='both', direction='in')
        #axs[1].legend()  
        # --- ADF
        print('Calculating ADF ...')
        firstTwo = False
        onlyAngle = True
        r_max = 1.25
        y_lim = 0.4
        for key, value in z_thresholds.items():
            label = "HOH_dist_{}_{}".format(key, structure)
            legend = 'HOH {} ({})'.format(key, structure) if structure != 'P' else 'HOH {} (Reference)'.format(key)
            angles = mean_adf(samples, 'H', 'O', 'H', r_max=r_max, firstTwo=firstTwo, mic=mic, onlyAngle=onlyAngle, aboveZthres=value)
            np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
            #plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder, show=True, figure_size=(3, 3))
            if key != 'Bottom':
                axs[2].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            sns.kdeplot(angles, ax=axs[2], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
        axs[2].set_xlabel(r'$\angle$HOH (degrees)')
        axs[2].set_xlim(90, 120)
        axs[2].set_ylabel(r'$\rho(\theta)$')
        axs[2].tick_params(axis='both', direction='in')
        #axs[2].legend()

        # Theta of OH and z-axis 
        r_max = 1.25
        y_lim = 0.04
        bins = 120 
        for key, value in z_thresholds.items():
            label = "Theta_OH_dist_{}_{}".format(key, structure)
            legend = r"$\theta_{{\text{{OH}}}}$ {} ({})".format(key, structure) if structure != 'P' else r"$\theta_{{\text{{OH}}}}$ {} (Reference)".format(key)
            angles = mean_adf_OH(samples, r_max=r_max, firstTwo=False, mic=False, onlyAngle=True, aboveZthres=value)
            np.savez('{}/{}.npz'.format(outputFolder, label), angles=angles)
            #plot_angle_distribution(angles, label, legend, color=color, bins=bins, y_lim=y_lim, outfolder=outputFolder, show=True, figure_size=(3, 3))
            if key != 'Bottom': 
                axs[3].hist(angles, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
            sns.kdeplot(angles, ax=axs[3], linewidth=1, label=legend, bw_adjust=0.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
        axs[3].set_xlabel(r'$\angle$ZOH (degrees)')
        axs[3].set_xlim(0, 180)
        axs[3].set_ylabel(r'$\rho(\theta)$')
        axs[3].tick_params(axis='both', direction='in')
        #axs[3].legend()

        #plt.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0, left=0.1, bottom=0.1, right=0.95, top=0.98)
        plt.savefig('{}/RDF_ADF_{}.pdf'.format(outputFolder, structure))
        plt.savefig('{}/RDF_ADF_{}.png'.format(outputFolder, structure), dpi=300)
        plt.savefig('{}/RDF_ADF_{}.svg'.format(outputFolder, structure))
        plt.show()




        # Bakup 
        #  # --- RDF 
        # z_thresholds = {'All': None, 'Top': True, 'Bottom': False}
        # colors = {'All': 'green', 'Top': 'red', 'Bottom': 'black'}
        # linestypes = {'All': '-', 'Top': ':', 'Bottom': '-.'}
        # fills = {'All': True, 'Top': True, 'Bottom': False}
        # for key, value in z_thresholds.items():
        #     r, gr_OO = mean_rdf(samples, 'O', 'O', r_max=3.5, mic=mic, bins=bins, aboveZthres=value)
        #     label = 'RDF_OO_{}_{}'.format(key, structure)
        #     legend = 'OO {} ({})'.format(key, structure) if structure != 'P' else 'OO {} (Reference)'.format(key)
        #     ylim = 10 if structure == 'Label' else 1.7
        #     #plot_rdf(r, gr_OO, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder, show=True, figure_size=(3, 3))
        #     np.savez('{}/RDF_OO_{}.npz'.format(outputFolder, key), r=r, gr=gr_OO, show=True)
        #     if key != 'Bottom':
        #         axs[0].step(r[:-1], gr_OO, alpha=0.3, color=colors[key], linewidth=0.5)
        #     #axs[0].plot(r[:-1], gr_OO, label=legend)
        #     axs[0].plot(r[:-1], gaussian_filter1d(gr_OO, sigma=1.5), label=key, color=colors[key], linestyle=linestypes[key], linewidth=1)
        #     if key != 'Bottom':
        #         axs[0].fill_between(r[:-1], 0, gaussian_filter1d(gr_OO, sigma=1.5), color=colors[key], alpha=0.3)
        # axs[0].set_xlabel('r (Å)')
        # axs[0].set_ylabel(r'$g_\text{OO}(r)$')
        # axs[0].set_ylim(0, 8)
        # axs[0].set_xlim(0, 3.5)
        # axs[0].legend(frameon=False, ncol=3)
        # axs[0].tick_params(axis='both', direction='in')

        # # O-H all water molecules
        # r_max = 1.25
        # for key, value in z_thresholds.items():
        #     r, gr_OH = mean_rdf(samples, 'O', 'H', r_max=r_max, mic=mic, bins=bins, aboveZthres=value)
        #     label = 'RDF_OH_{}_{}'.format(key, structure)
        #     legend = 'OH {} ({})'.format(key, structure) if structure != 'P' else 'OH {} (Reference)'.format(key)
        #     ylim = 150 if structure == 'Label' else 15
        #     #plot_rdf(r, gr_OH, label, legend, x_lim=r_max, y_lim=ylim, outfolder=outputFolder, show=True, figure_size=(3, 3))
        #     np.savez('{}/RDF_OH_{}.npz'.format(outputFolder, key), r=r, gr=gr_OH)
        #     #axs[1].plot(r, gr_OH, label=legend)
        #     if key != 'Bottom':
        #         axs[1].step(r[:-1], gr_OH, alpha=0.3, color=colors[key], linewidth=0.5)
        #     axs[1].plot(r[:-1], gaussian_filter1d(gr_OH, sigma=1.5), label=legend, color=colors[key], linestyle=linestypes[key], linewidth=1)
        #     if key != 'Bottom':
        #         axs[1].fill_between(r[:-1], 0, gaussian_filter1d(gr_OH, sigma=1.5), color=colors[key], alpha=0.3)
        # axs[1].set_xlabel('r (Å)')
        # axs[1].set_ylabel(r'$g_\text{OH}(r)$')
        # axs[1].set_ylim(0, 120)
        # axs[1].set_xlim(0, r_max)
        # axs[1].tick_params(axis='both', direction='in')