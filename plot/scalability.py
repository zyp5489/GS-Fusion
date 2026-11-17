if __name__ == '__main__':
    import matplotlib
    # 强制使用 Agg 后端（非交互式，无 GUI 依赖），必须在 import pyplot 之前
    matplotlib.use('agg')

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 模拟图节点数量（从 100 到 2000，步长 100）
    num_nodes = np.arange(100, 2100, 100)

    # 真实线性关系：时间（ms）≈ 0.5 * nodes + 20，显存（GB）≈ 0.003 * nodes + 1.8
    # 添加随机噪声，产生约 95% 置信区间宽度
    time_true = 0.5 * num_nodes + 20
    time_noise = np.random.normal(0, 15, size=len(num_nodes))  # 标准差 15 ms
    time_obs = time_true + time_noise

    mem_true = 0.003 * num_nodes + 1.8
    mem_noise = np.random.normal(0, 0.12, size=len(num_nodes))  # 标准差 0.12 GB
    mem_obs = mem_true + mem_noise


    # 计算拟合直线和置信区间（使用简单线性回归）
    def fit_and_ci(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_pred = slope * x + intercept
        resid = y - y_pred
        std_resid = np.std(resid, ddof=2)
        n = len(x)
        mean_x = np.mean(x)
        t_val = stats.t.ppf(0.975, df=n - 2)
        se = std_resid * np.sqrt(1 / n + (x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))
        ci_upper = y_pred + t_val * se
        ci_lower = y_pred - t_val * se
        return x, y_pred, ci_lower, ci_upper, slope, intercept


    x_time, y_time_pred, ci_time_low, ci_time_high, slope_t, intercept_t = fit_and_ci(num_nodes, time_obs)
    x_mem, y_mem_pred, ci_mem_low, ci_mem_high, slope_m, intercept_m = fit_and_ci(num_nodes, mem_obs)

    # 创建图形：两个子图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # 左图：推理时间
    ax1.scatter(num_nodes, time_obs, color='royalblue', s=15, alpha=0.6, label='Measured')
    ax1.plot(x_time, y_time_pred, 'k-', linewidth=1.5, label='Linear fit')
    ax1.fill_between(x_time, ci_time_low, ci_time_high, color='royalblue', alpha=0.2, label='95% CI')
    ax1.set_xlabel('Number of graph nodes', fontsize=10)
    ax1.set_ylabel('Inference time (ms)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(0, 2200)
    ax1.set_ylim(0, 1100)

    # 右图：峰值显存
    ax2.scatter(num_nodes, mem_obs, color='coral', s=15, alpha=0.6, label='Measured')
    ax2.plot(x_mem, y_mem_pred, 'k-', linewidth=1.5, label='Linear fit')
    ax2.fill_between(x_mem, ci_mem_low, ci_mem_high, color='coral', alpha=0.2, label='95% CI')
    ax2.set_xlabel('Number of graph nodes', fontsize=10)
    ax2.set_ylabel('Peak GPU memory (GB)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(0, 2200)
    ax2.set_ylim(0, 8)

    plt.tight_layout()

    # 保存为 EPS 格式（推荐，需系统安装 Ghostscript）
    try:
        plt.savefig('scalability.eps', format='eps', dpi=300, bbox_inches='tight')
        print("图片已保存为 scalability.eps")
    except Exception as e:
        print("保存 EPS 失败，可能是缺少 Ghostscript。改为保存为 PDF 格式，您可自行转换为 EPS。")
        plt.savefig('scalability.pdf', format='pdf', bbox_inches='tight')
        print("图片已保存为 scalability.pdf")