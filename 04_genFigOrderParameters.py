#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution, plot_kde_fill
from water import cal_d5_all, compute_sg_all, compute_sk_all, compute_lsi_all, compute_sg_sk_all
import seaborn as sns
from scipy.stats import gaussian_kde

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

        # Common parameters for the plots, and output folder
        bins = 50
        color = '#299035'
        outputFolder = os.path.join(baseOut, structure)

        z_thresholds = {'All': None, 'Top': True, 'Bottom': False}
        colors = {'All': 'green', 'Top': 'red', 'Bottom': 'black'}
        linestypes = {'All': '-', 'Top': ':', 'Bottom': '-.'}
        fills = {'All': True, 'Top': True, 'Bottom': False}

        r_max = 3.5
        qs, sks = compute_sg_sk_all(samples, r_max=r_max)
        x_min, y_min = qs.min(), sks.min()
        x_max, y_max = 1, 1
        nbin = 50 
        figsize = (8, 2.5)
        cmap = 'Greens'
        xgrid = np.linspace(x_min, x_max, nbin)
        ygrid = np.linspace(y_min, y_max, nbin)
        X, Y = np.meshgrid(xgrid, ygrid)
        texts = ['All', 'Top', 'Bottom']

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.025}, constrained_layout=True)
        for k, (key, value) in enumerate(z_thresholds.items()):
            qs, sks = compute_sg_sk_all(samples, r_max=r_max, aboveZthres=value)
            xy = np.vstack([qs, sks])
            kde = gaussian_kde(xy)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            contour = axes[k].pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=1200)
            axes[k].scatter(qs, sks, s=0.5, color='black', alpha=0.1)
            axes[k].tick_params(axis='both', direction='in', top=True, right=True)
            axes[k].text(0.2, 0.95, texts[k], color='black', fontsize=10, transform=axes[k].transAxes, verticalalignment='top')
            axes[k].set_xlim(x_min, x_max)
            axes[k].set_ylim(y_min, y_max)
            axes[k].set_xlabel(r'$S_g$')
            if k == 0:
                axes[k].set_ylabel(r'$S_k$')  # Only the first subplot has a y-label

        # Create shared colorbar
        cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label(r'$\rho(S_g, S_k)$')

        # Save and show
        plt.savefig('{}/OrderParameter2D.png'.format(outputFolder), dpi=600)
        plt.savefig('{}/OrderParameter2D.svg'.format(outputFolder))
        plt.show()
        

        # # --- d5 
        # for key, value in z_thresholds.items():
        #     d5s = cal_d5_all(samples, r_max=r_max, aboveZthres=value)
        #     label = 'd5_distances_{}_{}'.format(key, structure)
        #     legend = 'd5 {} ({})'.format(key, structure) if structure != 'P' else 'd5 {} (Reference)'.format(key)
        #     np.savez('{}/d5_distances_{}.npz'.format(outputFolder, key), d5=d5s, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[0].hist(d5s, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     #sns.kdeplot(OO_distances, ax=axs[0], linewidth=1, label=legend, bw_adjust=1.5, color=colors[key], linestyle=linestypes[key], fill=fills[key], alpha=0.3)
        #     plot_kde_fill(ax=axs[0], xmin=0, xmax=6.5, data=d5s, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[0].set_xlabel(r'$d_\text{5}$ (Å)')
        # axs[0].set_ylabel(r'$\rho(r)$')
        # axs[0].set_xlim(2.6, r_max+0.5)
        # axs[0].legend(frameon=False, ncol=1)
        # axs[0].tick_params(axis='both', direction='in')

        # # q value
        # r_max = 3.7
        # for key, value in z_thresholds.items():
        #     qs = compute_sg_all(samples, r_max=r_max, aboveZthres=value)
        #     label = 'q_{}_{}'.format(key, structure)
        #     legend = 'q {} ({})'.format(key, structure) if structure != 'P' else 'q {} (Reference)'.format(key)
        #     np.savez('{}/q_{}.npz'.format(outputFolder, key), q=qs, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[1].hist(qs, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     plot_kde_fill(ax=axs[1], xmin=-2.3, xmax=1, data=qs, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[1].set_xlabel(r'$q$')
        # axs[1].set_ylabel(r'$\rho(q)$')
        # #axs[1].set_xlim(0, r_max)
        # axs[1].legend(frameon=False, ncol=1)
        # axs[1].tick_params(axis='both', direction='in')

        # print('Calculating Sk ...')
        # r_max = 3.7
        # for key, value in z_thresholds.items():
        #     sks = compute_sk_all(samples, r_max=r_max, aboveZthres=value)
        #     label = 'S_k_{}_{}'.format(key, structure)
        #     legend = 'S_k {} ({})'.format(key, structure) if structure != 'P' else 'S_k {} (Reference)'.format(key)
        #     np.savez('{}/sk_{}.npz'.format(outputFolder, key), sk=sks, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[2].hist(sks, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     plot_kde_fill(ax=axs[2], xmin=0.9, xmax=1, data=sks, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[2].set_xlabel(r'$S_k$')
        # axs[2].set_ylabel(r'$\rho(S_k)$')
        # #axs[2].set_xlim(0, r_max)
        # axs[2].legend(frameon=False, ncol=1)
        # axs[2].tick_params(axis='both', direction='in')

        # bins = 50
        # r_max = 4.5
        # for key, value in z_thresholds.items():
        #     lsis = compute_lsi_all(samples, r_max=r_max, aboveZthres=value)
        #     print('Min LSI: ', np.min(lsis))
        #     print('Max LSI: ', np.max(lsis))
        #     label = 'LSI_{}_{}'.format(key, structure)
        #     legend = 'LSI {} ({})'.format(key, structure) if structure != 'P' else 'LSI {} (Reference)'.format(key)
        #     np.savez('{}/lsi_{}.npz'.format(outputFolder, key), lsi=lsis, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[3].hist(lsis, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     plot_kde_fill(ax=axs[3], xmin=0, xmax=0.23, data=lsis, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[3].set_xlabel(r'LSI')
        # axs[3].set_ylabel(r'$\rho(\text{LSI})$')
        # axs[3].set_xlim(0, 0.05)
        # axs[3].legend(frameon=False, ncol=1)
        # axs[3].tick_params(axis='both', direction='in')

        # # q V
        # r_max = 3.7
        # for key, value in z_thresholds.items():
        #     qs, sks = compute_sg_sk_all(samples, r_max=r_max, aboveZthres=value)
        #     label = 'q_{}_{}'.format(key, structure)
        #     legend = 'q {} ({})'.format(key, structure) if structure != 'P' else 'q {} (Reference)'.format(key)
        #     #np.savez('{}/q_{}.npz'.format(outputFolder, key), q=qs, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[4].hist(qs, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     plot_kde_fill(ax=axs[4], xmin=-2.3, xmax=1, data=qs, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[4].set_xlabel(r'$q$')
        # axs[4].set_ylabel(r'$\rho(q)$')
        # #axs[1].set_xlim(0, r_max)
        # axs[4].legend(frameon=False, ncol=1)
        # axs[4].tick_params(axis='both', direction='in')

        # r_max = 3.7
        # for key, value in z_thresholds.items():
        #     qs, sks = compute_sg_sk_all(samples, r_max=r_max, aboveZthres=value)
        #     label = 'S_k_{}_{}'.format(key, structure)
        #     legend = 'S_k {} ({})'.format(key, structure) if structure != 'P' else 'S_k {} (Reference)'.format(key)
        #     np.savez('{}/sk_{}.npz'.format(outputFolder, key), sk=sks, r_max=r_max)
        #     if key != 'Bottom':
        #         axs[5].hist(sks, bins=bins, histtype='step', density=True, linewidth=0.5, color=colors[key], alpha=0.2)
        #     plot_kde_fill(ax=axs[5], xmin=0.9, xmax=1, data=sks, color=colors[key], linestyle=linestypes[key], label=legend, fill=fills[key], alpha_fill=0.3)
        # axs[5].set_xlabel(r'$S_k$')
        # axs[5].set_ylabel(r'$\rho(S_k)$')
        # #axs[2].set_xlim(0, r_max)
        # axs[5].legend(frameon=False, ncol=1)
        # axs[5].tick_params(axis='both', direction='in')

        # #plt.tight_layout()
        # fig.subplots_adjust(hspace=0.4, wspace=0, left=0.1, bottom=0.1, right=0.95, top=0.98)
        # plt.savefig('{}/OrderParameter{}.pdf'.format(outputFolder, structure))
        # plt.savefig('{}/OrderParameter{}.png'.format(outputFolder, structure), dpi=600)
        # plt.savefig('{}/OrderParameter{}.svg'.format(outputFolder, structure))
        # plt.show()
