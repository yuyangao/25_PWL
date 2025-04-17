import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from scipy.ndimage import gaussian_filter1d
from tools.model_color import *

##  violin
def violin(ax, data, x, y, order, palette, orient='v',
        hue=None, hue_order=None,
        mean_marker_size=6, err_capsize=.11, scatter_size=7):
        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order,
                            orient=orient, palette=palette, 
                            legend=False, alpha=.25, inner=None, density_norm='width',
                            ax=ax)
        plt.setp(v.collections, alpha=.35, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            size=scatter_size,
                            edgecolor='auto', jitter=True, alpha=.45,
                            dodge=False if hue is None else True,
                            legend=False, zorder=2,
                            ax=ax)
        sns.barplot(data=data, 
                            x=x, y=y, order=order, 
                            orient=orient, 
                            hue=hue, hue_order=hue_order,
                            errorbar='se', linewidth=1, 
                            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                            capsize=err_capsize, err_kws={'linewidth': 2.5,'color': [0.2, 0.2, 0.2]},
                            ax=ax)
        groupby = [g_var, hue] if hue is not None else [g_var]
        sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                        x=x, y=y, order=order, 
                        hue=hue, hue_order=hue_order, 
                        palette=[[.2]*3]*len(hue_order) if hue is not None else None,
                        dodge=False if hue is None else True,
                        marker='o', size=mean_marker_size, color=[.2]*3, ax=ax)

## accuracy
def accuracy(all_ac1, all_ac2, group1, group2,
             ylabel, xlabel, fig_size, ):    
    '''
    all_acc: accuracy in dataframe form

    group: group name

    ylabel/xlabel: dependent/independent variable

    save_fig: save fig or not, bool

    output_file: if save, the save path
    '''
    font = {'family': 'Arial', 'weight': 'bold'}
    # Apply the font settings
    plt.rc('font', **font)
    plt.figure(figsize=fig_size, dpi=300)
    data_df1 = pd.DataFrame({'value': all_ac1, 'group': group1})
    data_df2 = pd.DataFrame({'value': all_ac2, 'group': group2})
    data_df = pd.concat([data_df1, data_df2])
    colors = {group1: "indianred",group2: "steelblue"}


    v = sns.violinplot(data=data_df, x='group', y='value', hue='group',
                    palette=colors, width=0.8, alpha=0.55, zorder=1,
                    inner=None, dodge=False)
    plt.setp(v.collections, alpha=.35, edgecolor='none')
    # draw scatterplot
    s = sns.stripplot(data=data_df, x='group', y='value', hue='group',
                    edgecolor='black', palette=colors,
                    alpha=0.8, size=6, zorder=1)
    b = sns.barplot(data=data_df, x='group', y='value',
                    hue='group', palette=colors, dodge=False, zorder=3)
    # 隐藏所有的条形
    for bar in b.patches:
        bar.set_height(0)  # 将条形的高度设为0，使其不可见
        bar.set_edgecolor((0, 0, 0, 0)) 
    # 计算每个组的均值
    group_means = data_df.groupby('group')['value'].mean()
    x_positions = np.array([0, 1])
    # 在errorbar上绘制每个组的均值横线
    for i, group in enumerate(data_df['group'].unique()):
        y_mean = group_means[group]
        plt.plot([x_positions[i] - 0.02, x_positions[i] + 0.02], [y_mean, y_mean], color='black', linewidth=2)

    # 设置坐标轴线条加粗...
    v.spines['bottom'].set_linewidth(1.5)  # 底部坐标轴线条
    v.spines['left'].set_linewidth(1.5)  # 左侧坐标轴线条
    v.tick_params(axis='both', which='major', width=1.5)  # 轴上刻度的粗细
    v.tick_params(axis='both', which='minor', width=1.5)
    # 去除边框线
    v.spines['top'].set_visible(False)
    v.spines['right'].set_visible(False)
    plt.ylabel(ylabel, fontsize=10, fontdict=font)
    plt.xlabel(xlabel, fontsize=10, fontdict=font)
    plt.yticks(fontsize=10, fontname='Arial', weight='bold')
    #plt.legend().remove()
    plt.xticks(fontsize=10)  # 更改 xtick 标签的字体大小
    plt.yticks(fontsize=10)
    plt.tight_layout()

# bar+scatter 
def bar_scat(ax, data, order, dot_size, color):
    '''
    datum's format: dataframe--cols: name(x), value(y), group
    '''
    font = {'family': 'Arial', 'weight': 'bold'}
    plt.rc('font', **font)

    s = sns.stripplot(data=data, x='name', y='value',hue='group', order=order, 
                      edgecolor='black', palette=color, 
                      dodge=True, jitter=False, size=dot_size, zorder=2,
                      ax=ax)    
    # 绘制 barplot
    b = sns.barplot(data=data, x='name', y='value', order=order, 
                    hue='group', palette=color, alpha=0.8, dodge=True,
                    ax=ax)

def basic_format(ax, x_name, y_name):
    font = {'family': 'Arial', 'weight': 'regular'}
    plt.rc('font', **font)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=1)
    ax.tick_params(axis='both', which='minor', width=1)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel(y_name,fontsize=10, fontdict=font)
    plt.xlabel(x_name, fontsize=10, fontdict=font)
    plt.xticks(fontsize=10)  # 更改 xtick 标签的字体大小
    plt.yticks(fontsize=10)


def model_compare_per_participant(ax, pth, models, group, valence, n_data=None, cr='bic'):
    crs_table = [] 
    for m in models:
        fname = f'{pth}/{m}/{m}_{valence}_{group}.pickle'
        with open(fname, 'rb')as handle: fit_info = pickle.load(handle)
        sub_lst = list(fit_info.keys())
        if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
        crs = {'sub_id': [], 'aic': [], 'bic': [], 'model': []}
        for sub_id in sub_lst:
            crs['sub_id'].append(sub_id)
            crs['aic'].append(fit_info[sub_id]['AIC'])
            crs['bic'].append(fit_info[sub_id]['BIC'])
            crs['model'].append(m)
        crs_table.append(pd.DataFrame.from_dict(crs))
    crs_table = pd.concat(crs_table, axis=0, ignore_index=True)
    sel_table = crs_table.pivot_table(
        values=cr,
        index='sub_id',
        columns='model',
        aggfunc=np.mean,
    ).reset_index()
    sel_table[f'min_{cr}'] = sel_table.apply(
        lambda x: np.min([x[m] for m in models]), 
        axis=1)
    sort_table = sel_table.sort_values(by=f'min_{cr}').reset_index()
    sort_table['sub_seq'] = sort_table.index

    for m in models:
        model_fn = eval(m)
        sns.scatterplot(x='sub_seq', y=m, data=sort_table,
                        marker=model_fn.marker,
                        size=model_fn.size,
                        color=model_fn.color,
                        alpha=model_fn.alpha,
                        linewidth=1.1,
                        edgecolor=model_fn.color if model_fn.marker in ['o'] else 'none',
                        facecolor='none'if model_fn.marker in ['o'] else model_fn.color,
                        ax=ax)

    #n_data = (12*8+6*8)*2
    if n_data is not None:
        ax.axhline(y=-np.log(1/2)*n_data*2, xmin=0, xmax=1,
                        color='k', lw=1, ls='--')
    ax.set_xlim([-2, sort_table.shape[0]+5])
    ax.legend(loc='upper left')
    ax.spines['left'].set_position(('axes',-0.02))
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2.5)
    ax.set_xlabel(f'Participant index (sorted by the minimum {cr} score over all models)')
    ax.set_ylabel(cr.upper())