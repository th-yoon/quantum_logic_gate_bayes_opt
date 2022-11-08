import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib import gridspec
from matplotlib.transforms import Affine2D
import numpy as np
import sys
import csv
import os.path as osp
import re

def gp_posterior(optimizer, x_obs, y_obs, grid):
    """
    Gets Gaussian Process posterior mu and sigma from observed data.

    Parameters:
    optimizer (BayesianOptimization): BayesianOptimization object from bayes_opt package
    x_obs (numpy array): observed data for x-axis
    y_obs (numpy array): observed data for y-axis
    grid (numpy array): 1-d grid for x-axis

    Returns:
    y_mean (numpy array): ndarray of shape (n_samples,) or (n_samples, n_targets)
        Mean of predictive distribution a query points.

    y_std (numpy array): ndarray of shape (n_samples,) or (n_samples, n_targets), optional
        Standard deviation of predictive distribution at query points.
        Only returned when `return_std` is True.

    y_cov (numpy array): ndarray of shape (n_samples, n_samples) or \
            (n_samples, n_samples, n_targets), optional
        Covariance of joint predictive distribution a query points.
        Only returned when `return_cov` is True.
    """
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, utility_function, x, y, name, max_list):
    """
    Plots the approximated function using Gaussian Process surrogate function.
    Plots the utility (acquisition) function over the approximated function.

    Parameters:
    optimizer (BayesianOptimization): BayesianOptimization object from bayes_opt package
    utility_function (UtilityFunction): UtilityFunction object from bayes_opt package
    x (numpy array): numpy array that represents the x-axis values
    y (numpy array): numpy array that represents the y-axis values for their corresponding x-axis values
    name (str): y-axis label
    max_list (list): list of optimizer.max whose target exceeds the specified fidelity threshold

    Returns:
    None, plots pop up
    """
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Acquisition Function after {} steps'.format(steps),
        fontdict={'size':30}
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    x_obs = np.array([[res['params']['t']] for res in optimizer.res])
    y_obs = np.array([res['target'] for res in optimizer.res])
    y_obs *= 100
    mu, sigma = gp_posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', linewidth=1, color='k', label='Prediction')
    axis.plot(
        optimizer.max['params']['t'], optimizer.max['target'] * 100, '*', markersize=10,
        label=u'Current max point', markerfacecolor='blue'
    )
    lo = sys.maxsize if len(max_list) != 0 else 0
    target = 0
    for m in max_list:
        if m['params']['t'] < lo:
            lo = m['params']['t']
            target = m['target']
    
    axis.plot(
        lo, target * 100, 'P', markersize=10,
        label=u'max point w/ min T', markerfacecolor='red'
    )
    axis.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval'
    )
    axis.set_xlabel('x', fontdict={'size':20})
    axis.set_ylabel(name, fontdict={'size':20})
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Acquisition Function', color='purple')
    acq.plot(
        x[np.argmax(utility)], np.max(utility), '*', markersize=10,
        label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k',
        markeredgewidth=1
    )
    acq.set_xlabel('x', fontdict={'size':20})
    acq.set_ylabel('Acquisition', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def save_data(max_dict_list, best_iteration, name, threshold, save_dir, j=None):
    """
    Saves csv data of all the sampled points (x, y, z, c) 
    and best points sampled so far (max).
    org_no_threshold or corr_no_threshold holds the boolean value 
    whether the corresponding evaluation step satisfies the threshold.

    Parameters:
    max_dict_list (list): list of dictionaries holding the sampled points from the optimizer.
    best_iteration (int): evaluation step that shows the current best samples.
    name (str): 'corr' (corrected) or 'org' (original).
    threshold (float): target fidelity threshold in [0, 1].
    save_dir (str): destination directory to save the csv file.

    Returns:
    None, saves the csv file. 
    """
    steps = len(max_dict_list['T'])
    x, y, z, c, org_no_threshold, corr_no_threshold = [], [], [], [], [], []
    for i in range(steps):
        x.append(max_dict_list['Bz_L'][i])
        y.append(max_dict_list['Bz_R'][i])
        z.append(max_dict_list['J'][i])
        c.append(max_dict_list['T'][i][max_dict_list[f'T_{name}_best'][i]])
        org_no_threshold.append(max_dict_list['org_no_threshold'][i])
        corr_no_threshold.append(max_dict_list['corr_no_threshold'][i])
    max = []
    max.append(max_dict_list['Bz_L'][best_iteration])
    max.append(max_dict_list['Bz_R'][best_iteration])
    max.append(max_dict_list['J'][best_iteration])
    max.append(max_dict_list['T'][best_iteration][max_dict_list[f'T_{name}_best'][best_iteration]])
    max.append(max_dict_list[f'F_{name}_best'][best_iteration])
    max.append(max_dict_list['org_no_threshold'][best_iteration])
    max.append(max_dict_list['corr_no_threshold'][best_iteration])

    with open(osp.join(save_dir, f'{name}_{threshold}_{steps}_i={j}.csv'),
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(name)
        writer.writerow([steps])
        writer.writerow(max)
        writer.writerow(x)
        writer.writerow(y)
        writer.writerow(z)
        writer.writerow(c)
        writer.writerow(org_no_threshold)
        writer.writerow(corr_no_threshold)

def read_data(dir, filename):
    """
    Reads the csv file saved from 'save_data' function.
    Gets the grid (x, y, z, c) and the sampled points (max).
    org_no_threshold or corr_no_threshold holds the boolean value 
    whether the corresponding evaluation step satisfies the threshold.

    Parameters:
    dir (str): directory where the csv file is located.
    filename (str): file name of the csv.

    Returns:
    name (str): 'corr' (corrected) or 'org' (original).
    steps (str): the number of evaluation steps.
    max (list): list of 'Bz_L', 'Bz_R', 'J', 'T', 'F_{}_best', 
    'org_no_threshold', 'corr_no_threshold' of the current best sample.
    x, y, z, c (numpy array): grid for Bz_L, Bz_R, J and min_t.
    org_no_threshold, corr_no_threshold (str): string of boolean value
    whether it exceeds the target threshold at the corresponding evaluation step.
    
    """
    with open(osp.join(dir, filename), 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                name = str(line)
            elif i == 1:
                steps = str(line)
            elif i == 2:
                max = line
            elif i == 3:
                x = np.array(line, dtype=float)
            elif i == 4:
                y = np.array(line, dtype=float)
            elif i == 5:
                z = np.array(line, dtype=float)
            elif i == 6:
                c = np.array(line, dtype=float)
            elif i == 7:
                org_no_threshold = np.array(line, dtype=object)
            elif i == 8:
                corr_no_threshold = np.array(line, dtype=object)
    name = re.findall('.+(?=\d*)', name)[0]
    steps = re.findall('\d+', steps)[0]
    erase = ',[]\' '
    org_no_threshold = (org_no_threshold == 'True')
    corr_no_threshold = (corr_no_threshold == 'True')
    for e in range(len(erase)):
        name = name.replace(erase[e], '')
        steps = steps.replace(erase[e], '')


    return name, steps, max, x, y, z, c, org_no_threshold, corr_no_threshold

def surfcont(dir, filename, optimizer, utility):
    """
    Plots four figures: 
        1) 3D scatter plot of sampled points and the 2D contour plot for UCB.
        2) 2D line plot for UCB(Bz_L) over the grid of Bz_L. 
        3) 2D line plot for UCB(Bz_R) over the grid of Bz_R.
        4) 2D line plot for UCB(J) over the grid of J.

    Parameters:
    dir (str): directory where the csv file is located.
    filename (str): file name of the csv.
    optimizer (BayesianOptimization): BayesianOptimization object from bayes_opt package
    utility_function (UtilityFunction): UtilityFunction object from bayes_opt package

    Returns:
    None, four figures are saved at the destination directory.
    """
    bounds = optimizer.space.bounds
    n_samples = 100
    x_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_samples)
    y_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_samples)
    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], n_samples)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_grid, y_grid, z_grid)
    x_surface, y_surface = np.meshgrid(x_grid, y_grid)
    grid = np.vstack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()])
    grid = grid.T # (n_samples)^3 x 3
    gp_mean_prediction, gp_std_prediction = optimizer._gp.predict(grid, return_std=True) 
    gp_mean_prediction = -gp_mean_prediction.reshape(n_samples, n_samples, n_samples) # in nanosecond
    gp_std_prediction = gp_std_prediction.reshape(n_samples, n_samples, n_samples) # in nanosecond
    z_argmin_t = np.argmin(gp_mean_prediction, axis=2)

    z_argmin_values, z_argmin_counts = np.unique(z_argmin_t, return_counts=True)
    z_argmin_ind = np.argmax(z_argmin_counts)

    z_min_t = np.take_along_axis(grid.reshape(n_samples, n_samples, n_samples, 3)[:, :, :, 2], z_argmin_t[None], axis=2)[:, :, 0]
    z_min_t /= 10
    gp_min_t_mean = np.take_along_axis(gp_mean_prediction, z_argmin_t[None], axis=2)[:, :, z_argmin_values[z_argmin_ind]]
    gp_min_t_std = np.take_along_axis(gp_std_prediction, z_argmin_t[None], axis=2)[:, :, z_argmin_values[z_argmin_ind]]

    util = utility.utility(grid, optimizer._gp, optimizer._space.target.max()) # (n_samples)^3
    util = util.reshape(n_samples, n_samples, n_samples)

    name, steps, max, x, y, z, c, org_no_threshold, corr_no_threshold = read_data(dir, filename)
    cmap_surface = plt.cm.get_cmap('rainbow')
    cmap_sampled = plt.cm.get_cmap('tab20b')

    x1 = np.linspace(x.min(), x.max(), len(np.unique(x)))
    y1 = np.linspace(y.min(), y.max(), len(np.unique(y)))
    x2, y2 = np.meshgrid(x1, y1)
    c *= 1e9
    z /= 10
    # t_CNOT
    color_dimension_sampled = c
    util_argmax_indices = np.unravel_index(util.argmax(), util.shape)
    # scale ucb color to [0, 1]
    color_dimension_grid_contour = (util[:, :, util_argmax_indices[2]] - np.min(util)) / (np.max(util) - np.min(util))
    # ucb
    minn_grid_contour, maxx_grid_contour = color_dimension_grid_contour.min(), color_dimension_grid_contour.max()
    # t_CNOT
    minn_sampled, maxx_sampled = color_dimension_sampled.min(), color_dimension_sampled.max()
    norm_grid_contour = matplotlib.colors.Normalize(minn_grid_contour, maxx_grid_contour)
    norm_sampled = matplotlib.colors.Normalize(minn_sampled, maxx_sampled)

    m_grid_contour = plt.cm.ScalarMappable(norm=norm_grid_contour, cmap=cmap_surface)
    m_grid_contour.set_array([])
    fcolors_grid_contour = m_grid_contour.to_rgba(color_dimension_grid_contour)

    m_sampled = plt.cm.ScalarMappable(cmap=cmap_sampled)
    m_sampled.set_array([])
    fcolors_sampled = m_sampled.to_rgba(color_dimension_sampled)

    fig_ax = plt.figure(figsize=(10, 10))
    fig_j = plt.figure(figsize=(3, 4))
    fig_bzl = plt.figure(figsize=(6, 6))
    fig_bzr = plt.figure(figsize=(6, 6))

    ax = fig_ax.add_subplot(111, projection='3d')
    j = fig_j.add_subplot(111)
    bzl = fig_bzl.add_subplot(111)
    bzr = fig_bzr.add_subplot(111)

    text = f'best point values: \
        \nBz_L: {float(max[0]):.2f} GHz \
        \nBz_R: {float(max[1]):.2f} GHz \
        \nJ: {float(max[2]) / 10:.2f} MHz \
        \nmin_t: {float(max[3]) * 1e9:.2f} ns \
        \nF_{name}: {float(max[4]) * 100:.2f} %'
    ax.annotate(text, xy=(0.01, 0.85), xycoords='axes fraction')
    eval_order_text = [str(i+1) for i in range(len(x))]

    cset = ax.contourf(
        x_surface, y_surface, color_dimension_grid_contour, levels=40, zdir='z', 
        offset=np.min(z)-200, norm=norm_grid_contour, cmap=cmap_surface)
    cbar_surface = fig_ax.colorbar(m_grid_contour, shrink=0.6, aspect=50, ax=ax, orientation='horizontal', location='top')
    cbar_surface.ax.get_yaxis().labelpad = 15
    cbar_surface.mappable.set_clim(0, 1)
    cbar_sampled = fig_ax.colorbar(m_sampled, shrink=0.6, aspect=50, ax=ax, orientation='vertical', location='left')
    cbar_sampled.mappable.set_clim(0, 200)
    ax.scatter(x, y, zs=np.min(z)-200, zdir='z', color='black', marker='x', zorder=200)

    print('corr_no_threshold:', corr_no_threshold)

    ax.scatter(
        x[corr_no_threshold==True], 
        y[corr_no_threshold==True], 
        z[corr_no_threshold==True], 
        marker='s', facecolors='none', edgecolors='black', s=100, zorder=100)
    ax.scatter(
        x[corr_no_threshold==False], 
        y[corr_no_threshold==False], 
        z[corr_no_threshold==False], 
        c=color_dimension_sampled[corr_no_threshold==False], 
        cmap=cmap_sampled,
        label='sampled points', 
        vmin=0, vmax=200,
        s=100, zorder=100)
    for i, txt in enumerate(eval_order_text):
        ax.text(x[i], y[i], z[i], '%s' % (txt), size=12, zorder=100)

    ax.scatter(
        float(max[0]), float(max[1]), float(max[2]) / 10, 
        c=float(max[3])*1e9, cmap=cmap_sampled, label='best point', marker='*', zorder=200, s=200)

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    ax.tick_params(axis="x", direction='out')

    ax.set_xlabel('Bz_L (GHz)')
    ax.set_ylabel('Bz_R (GHz)')
    ax.set_zlabel('J (MHz)')
    ax.set_zlim(z.min()-200, z.max())
    ax.set_xlim(10, 1)
    ax.set_ylim(10, 1)
    zticks = ax.zaxis.get_major_ticks()
    for i in range(len(zticks)):
        zticks[i].label1.set_visible(False)
    for i in range(3):
        zticks[-(i + 1)].label1.set_visible(True)

    fidelity = filename.split('_')[1]

    j_sum = util[util_argmax_indices[0], util_argmax_indices[1], :]
    j_sc = j.plot(
        np.linspace(0.3, 100, len(util[0, 0, :])),#, 0])), 
        (j_sum - np.min(j_sum)) / (np.max(j_sum) - np.min(j_sum)) * 100, 
        )
    j.set_ylabel('UCB(J)', fontdict={'size':20})
    j_xmax = np.linspace(0.3, 100, len(util[0, 0, :]))[np.argmax(j_sum)]
    j_ymax = ((j_sum - np.min(j_sum)) / (np.max(j_sum) - np.min(j_sum)) * 100).max()
    j.plot(
        j_xmax, j_ymax, 'P', markersize=10,
        label=u'Next sampling point', markerfacecolor='red'
    )
    bzl_sum = util[:, util_argmax_indices[1], util_argmax_indices[2]]
    bzl_sc = bzl.plot(
        np.linspace(1, 10, len(util[:, 0, 0])),#, 0])), 
        (bzl_sum - np.min(bzl_sum)) / (np.max(bzl_sum) - np.min(bzl_sum)) * 100, 
        )
    bzl.set_ylabel(r'UCB(Bz_L)', fontdict={'size':20})
    bzl_xmax = np.linspace(1, 10, len(util[:, 0, 0]))[np.argmax(bzl_sum)]
    bzl_ymax = ((bzl_sum - np.min(bzl_sum)) / (np.max(bzl_sum) - np.min(bzl_sum)) * 100).max()
    bzl.plot(
        bzl_xmax, bzl_ymax, 'P', markersize=10,
        label=u'Next sampling point', markerfacecolor='red'
    )
    bzr_sum = util[util_argmax_indices[0], :, util_argmax_indices[2]]
    bzr_sc = bzr.plot(
        np.linspace(1, 10, len(util[0, :, 0])),#, 0])), 
        (bzr_sum - np.min(bzr_sum)) / (np.max(bzr_sum) - np.min(bzr_sum)) * 100, 
        )
    bzr.set_ylabel(r'UCB(Bz_R)', fontdict={'size':20})
    bzr_xmax = np.linspace(1, 10, len(util[0, :, 0]))[np.argmax(bzr_sum)]
    bzr_ymax = ((bzr_sum - np.min(bzr_sum)) / (np.max(bzr_sum) - np.min(bzr_sum)) * 100).max()
    bzr.plot(
        bzr_xmax, bzr_ymax, 'P', markersize=10,
        label=u'Next sampling point', markerfacecolor='red'
    )

    fig_ax.savefig(osp.join(dir, f'surface_Bz_L-Bz_R-J-{name}_{fidelity}_{steps}.svg'), format='svg')
    fig_j.savefig(osp.join(dir, f'j_Bz_L-Bz_R-J-{name}_{fidelity}_{steps}.svg'), format='svg')
    fig_bzl.savefig(osp.join(dir, f'bzl_Bz_L-Bz_R-J-{name}_{fidelity}_{steps}.svg'), format='svg')
    fig_bzr.savefig(osp.join(dir, f'bzr_Bz_L-Bz_R-J-{name}_{fidelity}_{steps}.svg'), format='svg')

def min_t_plot(dir, orgs, corrs):
    min_ts_orgs = []
    min_ts_corrs = []
    F_orgs = []
    F_corrs = []
    for i in range(len(orgs)):
        name, steps, max, x, y, z, c = read_data(dir, orgs[i])
        F_orgs.append(orgs[i].split('_')[1])
        min_ts_orgs.append(float(max[3]))
    for i in range(len(corrs)):
        name, steps, max, x, y, z, c = read_data(dir, corrs[i])
        F_corrs.append(corrs[i].split('_')[1])
        min_ts_corrs.append(float(max[3]))

    plt.xlabel('Fidelities')
    plt.ylabel('min_t')
    plt.plot(F_orgs, min_ts_orgs, label='F_org')
    plt.legend(loc='best')
    plt.plot(F_corrs, min_ts_corrs, label='F_corr')
    plt.legend(loc='best')

    plt.title(f'min_t vs. F', fontsize=20)
    plt.savefig(osp.join(dir, f'min_t vs. F.png'))
    plt.show()