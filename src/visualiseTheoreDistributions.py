#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from water import read_samples_from_folder 
from water import mean_rdf, mean_adf, mean_distance_distribution, mean_adf_OH, compute_sg_sk_all
from water import plot_rdf, plot_angle_distribution, plot_distance_distribution

from utils import plot_kde_fill

import seaborn as sns
from scipy.stats import gaussian_kde

if __name__ == '__main__':
    inputFolder = '../data/structures/simulations/'
    processedFolder = '../processed_data/theory_distributions/'
    os.makedirs(processedFolder, exist_ok=True)
    baseOut = '../results/theoretical_distributions/'
    structures = ['Label']
    show = True
    
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
        x_max, y_max = 1, 1
        figsize = (8, 2.5)
        cmap = 'Greens'
        # Save the All, Top, and Bottom separately 
        for k, (key, value) in enumerate(z_thresholds.items()):
            npzFile = '{}/OrderParameter_{}.npz'.format(npzOut, key)
            if os.path.exists(npzFile): 
                print('Loading OrderParameter from file: {}'.format(npzFile))
                sgs, sks = np.load(npzFile)['sgs'], np.load(npzFile)['sks']
            else:
                print('Calculating OrderParameter ...')
                sgs, sks = compute_sg_sk_all(samples, r_max=r_max, aboveZthres=value)
                np.savez(npzFile, sgs=sgs, sks=sks)
            num_samples = sgs.size

            sns.set(style="white")
            g = sns.jointplot(x=sgs, y=sks, kind="kde", fill=True, bw_adjust=0.5,
                              height=5,        # Height of the joint plot (in inches)
                              ratio=4)         # Size ratio of marginal plots to joint)

            g.fig.set_size_inches(6, 6)
            g.ax_joint.scatter(sgs, sks, s=5, color="black", alpha=0.3,
                               marker='o', linewidths=0)  # You can tweak size, color, alpha
            
            # Set axis limits
            g.ax_joint.set_xlim(x_min, x_max + 0.1)
            g.ax_joint.set_ylim(y_min, y_max + 0.0008)

            # Force ticks to show
            g.ax_joint.tick_params(left=True, bottom=True, direction='in')
            g.ax_marg_x.tick_params(bottom=True)
            g.ax_marg_y.tick_params(left=True)
            
            g.fig.subplots_adjust(hspace=0.01, wspace=0.01)
            g.set_axis_labels(r'$S_g$', r'$S_k$', labelpad=8)
            g.ax_joint.text(0.15, 0.95, key + " samples {}".format(num_samples), color='black', fontsize=14,
                transform=g.ax_joint.transAxes, verticalalignment='top')
            g.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            g.fig.savefig(f"{figureOut}/OrderParameter_{structure}_{key}.pdf")
            g.fig.savefig(f"{figureOut}/OrderParameter_{structure}_{key}.png", dpi=300)
            g.fig.savefig(f"{figureOut}/OrderParameter_{structure}_{key}.svg")
            if show: plt.show()  
            plt.close()


        # Show the All, Top, and Bottom in one figure
        sgs_list, sks_list = [], []
        for k, (key, value) in enumerate(z_thresholds.items()):
            sgs, sks = np.load(f"{npzOut}/OrderParameter_{key}.npz")['sgs'], np.load(f"{npzOut}/OrderParameter_{key}.npz")['sks']
            sgs_list.append(sgs)
            sks_list.append(sks)

        sns.set(style="white")
        # Create 2x2 layout: [0,0]=marg_x, [1,0]=joint, [1,1]=marg_y
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                            hspace=0.05, wspace=0.05)
        
        ax_joint = fig.add_subplot(grid[1, 0])
        ax_marg_x = fig.add_subplot(grid[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(grid[1, 1], sharey=ax_joint)

        
        # Loop over each group to plot joint and marginal densities
        #for sgs, sks, label, color in zip(sgs_list, sks_list, labels, colors):
        for k, (key, value) in enumerate(z_thresholds.items()):
            sgs, sks = np.load(f"{npzOut}/OrderParameter_{key}.npz")['sgs'], np.load(f"{npzOut}/OrderParameter_{key}.npz")['sks']
            color = colors[key]
            if key == "All":
                # Joint KDE for 'All' only
                sns.kdeplot(x=sgs, y=sks, fill=True, bw_adjust=0.5, ax=ax_joint,
                            cmap=sns.light_palette(color, as_cmap=True))
            
            else:
                # Scatter for Top and Bottom only
                ax_joint.scatter(sgs, sks, s=5, marker = ',' if key =="Bottom"
                                 else 'x', color=color, alpha=0.5,
                                 label=f'{key} samples')
        
            # Marginal KDEs
            sns.kdeplot(x=sgs, ax=ax_marg_x, color=color, fill=False if key !=
                        "All" else True,
                        bw_adjust=0.5, alpha=0.3 if key ==
                        "All" else 1, label=f"{label} samples")
            sns.kdeplot(y=sks, ax=ax_marg_y, color=color, fill=False if key !=
                        "All" else True,
                        bw_adjust=0.5, alpha=0.3 if key ==
                        "All" else 1, label=f"{label} samples") 
        
        # Clean up joint plot
        ax_joint.set_xlim(x_min, x_max + 0.1)
        ax_joint.set_ylim(y_min, y_max + 0.0008)
        ax_joint.set_xlabel(r"$S_g$")
        ax_joint.set_ylabel(r"$S_k$")
        ax_joint.tick_params(direction="in")
        ax_joint.legend(loc='upper left', frameon=False)
        ax_marg_x.legend(loc='lower left', frameon=False)
        
        # Hide axis ticks for marginal plots
        ax_marg_x.axis("off")
        ax_marg_y.axis("off")
       
        
        # Create custom legend handles for marginal lines
        legend_lines = [
            Line2D([0], [0], color=color, lw=2, label=label)
            for label, color in colors.items()
        ]
        
        # Place legend at top center of marginal x-axis
        ax_marg_x.legend(handles=legend_lines, loc='lower center', frameon=False,
                         ncol=1, bbox_to_anchor=(0.15, 0.00), fontsize=10)
        #fig.subplots_adjust(hspace=0.01, wspace=0.01)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        # Save figure
        fig.savefig(f"{figureOut}/OrderParameter_{structure}_overlay_all.pdf", bbox_inches='tight')
        fig.savefig(f"{figureOut}/OrderParameter_{structure}_overlay_all.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{figureOut}/OrderParameter_{structure}_overlay_all.svg", bbox_inches='tight')
        
        if show: plt.show()
        plt.close()

        # Show all side by side
        nbin = 50
        figsize = (8, 2.5)
        cmap = 'Greens'
        xgrid = np.linspace(x_min, x_max, nbin)
        ygrid = np.linspace(y_min, y_max, nbin)
        X, Y = np.meshgrid(xgrid, ygrid)
        texts = ['All', 'Top', 'Bottom']
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.025}, constrained_layout=True)
        for k, (key, value) in enumerate(z_thresholds.items()):
            sgs, sks = np.load(f"{npzOut}/OrderParameter_{key}.npz")['sgs'], np.load(f"{npzOut}/OrderParameter_{key}.npz")['sks']
            xy = np.vstack([sgs, sks])
            kde = gaussian_kde(xy)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            contour = axes[k].pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=1200)
            axes[k].scatter(sgs, sks, s=0.5, color='black', alpha=0.1)
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
        plt.savefig(f'{figureOut}/OrderParameter_{structure}_row.pdf'.format(outputFolder))
        plt.savefig(f'{figureOut}/OrderParameter_{structure}_row.png'.format(outputFolder), dpi=600)
        plt.savefig(f'{figureOut}/OrderParameter_{structure}_row.svg'.format(outputFolder))
        if show: plt.show() 
        
