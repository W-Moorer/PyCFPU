<div align="center">
  <pre style="font-family: 'Courier New', monospace; 
              line-height: 1.2; 
              white-space: pre-wrap;
              display: inline-block;
              padding: 10px;
              border-radius: 4px;
              border: 1px solid #3b0aceff;">
$$$$$$$\             $$$$$$\  $$$$$$$$\ $$$$$$$\  $$\   $$\ 
$$  __$$\           $$  __$$\ $$  _____|$$  __$$\ $$ |  $$ |
$$ |  $$ |$$\   $$\ $$ /  \__|$$ |      $$ |  $$ |$$ |  $$ |
$$$$$$$  |$$ |  $$ |$$ |      $$$$$\    $$$$$$$  |$$ |  $$ |
$$  ____/ $$ |  $$ |$$ |      $$  __|   $$  ____/ $$ |  $$ |
$$ |      $$ |  $$ |$$ |  $$\ $$ |      $$ |      $$ |  $$ |
$$ |      \$$$$$$$ |\$$$$$$  |$$ |      $$ |      \$$$$$$  |
\__|       \____$$ | \______/ \__|      \__|       \______/ 
          $$\   $$ |                                        
          \$$$$$$  |                                        
           \______/                                         
  </pre>
</div>

# PyCFPU

面向有向点云（点 + 法向）进行隐式曲面重建的 Python 实现，算法基于 Curl-Free RBF Partition of Unity（CFPU）。本项目将原始 MATLAB 版本迁移为 Python 版本，并提供示例数据与 PyVista 可视化脚本。

## 来源与引用
- 原始 MATLAB 实现与示例：<https://github.com/gradywright/cfpu>
- 参考文献：[1] K. P. Drake, E. J. Fuselier, and G. B. Wright. Implicit Surface Reconstruction with a Curl-free Radial Basis Function Partition of Unity Method. SIAM J. Sci. Comput. 42, A3018–A3040 (2022). doi:10.1137/20M1386166. 预印本：<https://arxiv.org/abs/2101.05940>
- 本仓库的 Python 代码是对上述 MATLAB 版本的移植与工程化封装，核心算法与符号约定与原文一致。

## 功能概述
- 重建函数：`cfpurecon(points, normals, patches, m, kernel, regularization)`
- 示例数据：`data/demo_nodes.txt`、`data/demo_normals.txt`、`data/demo_patches.txt` 以及多模型的 `demo_*__模型名.txt`
- 可视化脚本：双视图联动渲染零等值面与点云（PyVista）

## 安装与运行
- 依赖：`pip install numpy scipy pyvista`
- 开发模式安装（可选，用于命令行入口）：
  - `pip install -e .`
  - 安装后可使用 `pycfpu-render`（指向 `scripts/render_pyvista.py:main`）
- 直接运行脚本：
  - `python scripts/render_pyvista.py --m 256`
  - 指定模型：`python scripts/render_pyvista.py --model homer --m 256`
  - 指定并行：`python scripts/render_pyvista.py --m 256 --jobs 4`；不传或 `--jobs 0` 为自动并行

## 目录结构
- `cfpurecon.py`：重建主函数
- `util/`：工具函数（`curlfree_poly.py`、`weight.py`、`gcv_cost_function.py`）
- `data/`：示例与转换后的点云数据（`demo_nodes.txt` 等及 `demo_*__模型名.txt`）
- `scripts/render_pyvista.py`：渲染入口（参数解析见 `scripts/render_pyvista.py:7`）

## 快速开始
- 可视化渲染（默认 demo 数据）：
  - `python scripts/render_pyvista.py --m 256`
- 指定模型渲染：
  - `python scripts/render_pyvista.py --model stanford_dragon --m 256`
- 直接调用 API：
  ```python
  import numpy as np
  from pycfpu.cfpurecon import cfpurecon

  points = np.loadtxt('data/demo_nodes.txt')
  normals = np.loadtxt('data/demo_normals.txt')
  patches = np.loadtxt('data/demo_patches.txt')

  m = 400
  kernel = {
      'phi': lambda r: -r,
      'eta': lambda r: -r,
      'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
      'order': 1
  }
  regularization = {
      'exactinterp': 1,
      'nrmlreg': 1,
      'nrmllambda': 1e-4,
      'potreg': 0
  }
  potential, X, Y, Z = cfpurecon(points, normals, patches, m, kernel, regularization)
  ```

## 数据说明
- 直接以文本形式读取：
  - 默认：`data/demo_nodes.txt`、`data/demo_normals.txt`、`data/demo_patches.txt`
  - 指定模型：`data/demo_nodes__模型名.txt`、`data/demo_normals__模型名.txt`、`data/demo_patches__模型名.txt`
- 法向量应为单位向量；如需归一化，使用 `normals /= np.linalg.norm(normals, axis=1, keepdims=True)` 并对零范数做保护。

## 参数与调优
- `m`：背景网格分辨率。数值越大曲面更平滑、细节更清晰，但耗时增加；常用 256–500。
- `kernel`（多调和样条）：
  - 一阶：`phi(r)=-r`、`eta(r)=-r`、`zeta(r)=-1/r`、`order=1`（默认）
  - 二阶：`phi(r)=r^3`、`eta(r)=r^3`、`zeta(r)=3r`、`order=2`
- `regularization`：
  - `exactinterp`：势函数插值修正；`1` 启用。
  - `nrmlreg`/`nrmllambda`：法向拟合正则（岭/GCV/局部）。
  - `potreg`/`potlambda`：势函数残差的标量 RBF 正则。

## PyVista 渲染提示
- 双视图联动：在两个子图中调用 `plotter.link_views()` 以同步交互。
- 节点叠加：使用 `add_points(..., render_points_as_spheres=True)` 避免 Glyph 的 `orient` 警告。

## 并行与性能
- 核心重建函数支持并行：`cfpurecon(..., n_jobs=None)`；当 `n_jobs=None` 时自动使用 `min(patches_count, os.cpu_count())`。
- 渲染脚本参数：`--jobs`，默认 `0` 表示自动；传正整数则使用指定线程数。
- 若本地 BLAS 已启用多线程，过大的 `--jobs` 可能导致过度调度；建议 2–8 间调试。

## 批量保存截图
- 批量遍历 `data` 中所有 demo 模型并保存 PNG 到 `figs/`，默认 DPI=600：
  - `python scripts/save_all_figs.py --m 256 --dpi 600`
- 指定模型与输出目录、并行与尺寸：
  - `python scripts/save_all_figs.py --models homer armadillo --m 256 --dpi 600 --out figs --jobs 0 --width_in 6 --height_in 4`
- 文件命名：`figs/<模型名>_m<m>.png`；默认模型文件名为 `default_m<m>.png`。
- 运行时输出进度：显示 `[i/N] 模型名 ####------ 进度% -> 文件名`，并在开始提示总模型数与并行模式。

## 致谢（数据来源）
- Stanford Bunny、Happy Buddha、Stanford Dragon、Armadillo：<http://graphics.stanford.edu/data/3Dscanrep/>
- Homer、Raptor、Filigree、Pump Carter、Dancing Children、Gargoyle、De Bozbezbozzel：<http://visionair.ge.imati.cnr.it>

## 参考文献
- [1] K. P. Drake, E. J. Fuselier, and G. B. Wright. Implicit Surface Reconstruction with a Curl-free Radial Basis Function Partition of Unity Method. SIAM J. Sci. Comput. 42, A3018–A3040 (2022). doi:10.1137/20M1386166. 预印本：<https://arxiv.org/abs/2101.05940>

## 模型图集（figs）
<table>
  <tr>
    <td><a href="figs/default_m256.png"><img src="figs/default_m256.png" alt="default" width="300"></a></td>
    <td><a href="figs/armadillo_m256.png"><img src="figs/armadillo_m256.png" alt="armadillo" width="300"></a></td>
    <td><a href="figs/bunny_large_m256.png"><img src="figs/bunny_large_m256.png" alt="bunny_large" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/cantius_tooth_m256.png"><img src="figs/cantius_tooth_m256.png" alt="cantius_tooth" width="300"></a></td>
    <td><a href="figs/dancing_children_m256.png"><img src="figs/dancing_children_m256.png" alt="dancing_children" width="300"></a></td>
    <td><a href="figs/filigree_m256.png"><img src="figs/filigree_m256.png" alt="filigree" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/frog_m256.png"><img src="figs/frog_m256.png" alt="frog" width="300"></a></td>
    <td><a href="figs/gargoyle_m256.png"><img src="figs/gargoyle_m256.png" alt="gargoyle" width="300"></a></td>
    <td><a href="figs/happy_buddha_m256.png"><img src="figs/happy_buddha_m256.png" alt="happy_buddha" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/homer_m256.png"><img src="figs/homer_m256.png" alt="homer" width="300"></a></td>
    <td><a href="figs/interlocked_tori_m256.png"><img src="figs/interlocked_tori_m256.png" alt="interlocked_tori" width="300"></a></td>
    <td><a href="figs/mammoth_tooth_m256.png"><img src="figs/mammoth_tooth_m256.png" alt="mammoth_tooth" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/pump_carter_m256.png"><img src="figs/pump_carter_m256.png" alt="pump_carter" width="300"></a></td>
    <td><a href="figs/raptor_head_m256.png"><img src="figs/raptor_head_m256.png" alt="raptor_head" width="300"></a></td>
    <td><a href="figs/stanford_bunny_m256.png"><img src="figs/stanford_bunny_m256.png" alt="stanford_bunny" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/stanford_dragon_m256.png"><img src="figs/stanford_dragon_m256.png" alt="stanford_dragon" width="300"></a></td>
    <td><a href="figs/trefoil_N11616_m256.png"><img src="figs/trefoil_N11616_m256.png" alt="trefoil_N11616" width="300"></a></td>
    <td><a href="figs/trefoil_N18816_m256.png"><img src="figs/trefoil_N18816_m256.png" alt="trefoil_N18816" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/trefoil_N23064_m256.png"><img src="figs/trefoil_N23064_m256.png" alt="trefoil_N23064" width="300"></a></td>
    <td><a href="figs/trefoil_N27744_m256.png"><img src="figs/trefoil_N27744_m256.png" alt="trefoil_N27744" width="300"></a></td>
    <td><a href="figs/trefoil_N32856_m256.png"><img src="figs/trefoil_N32856_m256.png" alt="trefoil_N32856" width="300"></a></td>
  </tr>
  <tr>
    <td><a href="figs/trefoil_N6144_m256.png"><img src="figs/trefoil_N6144_m256.png" alt="trefoil_N6144" width="300"></a></td>
    <td><a href="figs/trefoil_N8664_m256.png"><img src="figs/trefoil_N8664_m256.png" alt="trefoil_N8664" width="300"></a></td>
    <td></td>
  </tr>
</table>
