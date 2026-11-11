"""实验二：绘制创新实验的两张图。数据来自已完成的 q4_experiment.py 输出。
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
fonts = {f.name for f in font_manager.fontManager.ttflist}
for name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP']:
    if name in fonts:
        plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 图1：四种方法的实际叶子访问数（不同 N/H 为离散实验配置）
scales = ['N=4, H=2', 'N=4, H=4', 'N=6, H=2', 'N=6, H=4', 'N=6, H=6']
leaves = {
    'Minimax': [12, 24, 30, 360, 720],
    'Fixed-left': [8, 16, 18, 146, 171],
    'Fixed-right': [11, 16, 14, 77, 147],
    'Dynamic': [6, 12, 10, 74, 111],
}
fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
x = np.arange(len(scales))
for method, counts in leaves.items():
    ax.plot(x, counts, marker='o', linewidth=2, markersize=5, label=method)
ax.set_xticks(x, scales)
ax.set_ylabel('Visited leaves')
ax.set_title('Search effort across treasure scenarios')
ax.set_ylim(bottom=0)
ax.grid(axis='y', alpha=0.25)
ax.legend(frameon=False, ncol=2)
fig.savefig(OUT / 'fig1_search_efficiency.png', bbox_inches='tight')
plt.close(fig)

# 图2：同一地图和搜索深度下，价值配置对叶子访问数的影响
scenarios = ['Original', 'High variance', 'Similar values']
fixed_right = [147, 151, 139]
dynamic = [111, 63, 110]
fig, ax = plt.subplots(figsize=(7.8, 4.5), layout='constrained')
x = np.arange(len(scenarios))
w = 0.35
r = ax.bar(x-w/2, fixed_right, w, label='Fixed-right')
d = ax.bar(x+w/2, dynamic, w, label='Dynamic')
ax.set_xticks(x, scenarios)
ax.set_ylabel('Visited leaves')
ax.set_title('Effect of treasure value distributions (N=6, H=6)')
ax.set_ylim(0, 185)
ax.grid(axis='y', alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False)
ax.bar_label(r, padding=3)
ax.bar_label(d, padding=3)
fig.savefig(OUT / 'fig2_value_distribution.png', bbox_inches='tight')
plt.close(fig)
print('Figures saved to:', OUT)