import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Agg' 如果不需要显示窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- 数据 ----------
if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    data = {
        'Model': ['w/o M&N', 'w/o S&N', 'w/o S&M', 'w/o S',
                  'w/o M', 'w/o N', 'w/o CL', 'Full'],
        'MSE': [0.178, 0.204, 0.176, 0.158, 0.165, 0.155, 0.158, 0.140],
        'CI': [0.885, 0.902, 0.893, 0.909, 0.908, 0.905, 0.907, 0.918],
        'r2_m': [0.737, 0.723, 0.758, 0.781, 0.775, 0.784, 0.777, 0.808]
    }
    df = pd.DataFrame(data)

    palette = {'MSE': '#4C72B0', 'CI': '#DD8452', 'r2_m': '#55A868'}

    # 关键：缩小高度
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    fig.subplots_adjust(wspace=0.3, top=0.88)

    metrics = ['MSE', 'CI', 'r2_m']
    titles = ['MSE (↓)', 'CI (↑)', r'$r_m^2$ (↑)']
    ylabels = ['Mean Squared Error', 'Concordance Index', r'$r_m^2$']

    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        bars = ax.bar(df['Model'], df[metric], color=palette[metric],
                      alpha=0.85, edgecolor='black', linewidth=0.6, width=0.65)
        bars[-1].set_color('#D62728')
        bars[-1].set_edgecolor('darkred')
        bars[-1].set_linewidth(1.2)
        for bar, val in zip(bars, df[metric]):
            offset = 0.002 if metric == 'MSE' else 0.002
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='medium')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        if metric == 'MSE':
            ax.set_ylim(0.12, 0.22)
        elif metric == 'CI':
            ax.set_ylim(0.87, 0.94)
        else:
            ax.set_ylim(0.70, 0.82)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

    plt.suptitle('Ablation Study Results on Davis Dataset', fontsize=14, y=0.98)

    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor='#D62728', edgecolor='darkred', label='Full Model')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10)

    plt.tight_layout()
    plt.show()
    # plt.savefig('ablation_results.pdf', dpi=300, bbox_inches='tight')