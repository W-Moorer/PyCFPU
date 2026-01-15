import numpy as np
import pyvista as pv
# Modified by Refactoring Tool: Updated import path from 'pycfpu.cfpurecon' to 'pycfpu.cfpu'
from pycfpu.packages.cfpu import cfpurecon
import time
from datetime import datetime
from pathlib import Path
import os

# ==================== 全局参数配置 ====================
# 用户可在此处修改以下参数来控制重建过程

# 文件路径配置 - 用户可在此指定具体的文件路径
NODES_FILE = "data/input_txt/nonsmooth_geometry/TruncatedRing_nodes.txt"    # node文件路径
NORMALS_FILE = "data/input_txt/nonsmooth_geometry/TruncatedRing_normals.txt"  # normal文件路径
PATCHES_FILE = "data/input_txt/nonsmooth_geometry/TruncatedRing_patches.txt"  # patch文件路径

# 如果上述文件路径为None，则使用默认模型文件
MODEL_NAME = None    # 模型名称，None表示使用默认模型

# 重建参数配置
GRID_M = 128         # 网格分辨率
JOBS = 0             # 并行工作数，0表示自动选择

# 核函数配置
KERNEL_CONFIG = {
    'phi': lambda r: -r,
    'eta': lambda r: -r,
    'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
    'order': 1
}

# 正则化配置
REGULARIZATION_CONFIG = {
    'exactinterp': 1,
    'nrmlreg': 1,
    'nrmllambda': 1e-4,
    'potreg': 0
}

# 目录配置
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
LOGS_DIR = Path(__file__).resolve().parents[1] / 'logs'

# 可视化配置
VISUALIZATION_CONFIG = {
    'isosurface_value': 0.0,
    'mesh_color': 'lightgray',
    'mesh_specular': 0.1,
    'mesh_opacity': 0.85,
    'points_color': 'red',
    'points_size': 10,
    'smooth_shading': True
}

# ==================== 主函数 ====================

def main():
    """
    主函数：执行CFPU重建过程
    """
    # 打印配置信息
    print("=== CFPU重建配置 ===")
    print(f"节点文件: {NODES_FILE or '使用默认模型'}")
    print(f"法向量文件: {NORMALS_FILE or '使用默认模型'}")
    print(f"补丁文件: {PATCHES_FILE or '使用默认模型'}")
    print(f"模型名称: {MODEL_NAME or '默认模型'}")
    print(f"网格分辨率: {GRID_M}")
    print(f"并行工作数: {JOBS}")
    print(f"数据目录: {DATA_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print("=" * 40)
    
    # 加载数据文件
    try:
        if NODES_FILE and NORMALS_FILE and PATCHES_FILE:
            # 使用用户指定的具体文件路径
            points = np.loadtxt(NODES_FILE)
            normals = np.loadtxt(NORMALS_FILE)
            patches = np.loadtxt(PATCHES_FILE)
            print(f"使用自定义文件: {NODES_FILE}, {NORMALS_FILE}, {PATCHES_FILE}")
        elif MODEL_NAME:
            # 使用指定模型名称的默认文件
            points = np.loadtxt(str(DATA_DIR / f'demo_nodes__{MODEL_NAME}.txt'))
            normals = np.loadtxt(str(DATA_DIR / f'demo_normals__{MODEL_NAME}.txt'))
            patches = np.loadtxt(str(DATA_DIR / f'demo_patches__{MODEL_NAME}.txt'))
            print(f"使用模型文件: {MODEL_NAME}")
        else:
            # 使用默认模型文件
            points = np.loadtxt(str(DATA_DIR / 'demo_nodes.txt'))
            normals = np.loadtxt(str(DATA_DIR / 'demo_normals.txt'))
            patches = np.loadtxt(str(DATA_DIR / 'demo_patches.txt'))
            print("使用默认模型文件")
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 - {e}")
        return
    
    # 计算边界
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    
    # 确定并行工作数
    M = patches.shape[0]
    used_workers = JOBS if (JOBS and JOBS > 0) else min(M, os.cpu_count() or 1)
    mode = 'auto' if JOBS is None else 'manual'
    
    # 打印数据信息
    print("=== 数据信息 ===")
    print(f"点云形状: {points.shape}")
    print(f"法向量形状: {normals.shape}")
    print(f"补丁形状: {patches.shape}")
    print(f"边界最小值: {bounds_min}")
    print(f"边界最大值: {bounds_max}")
    print(f"使用线程数: {used_workers} (模式: {mode})")
    
    # 执行CFPU重建
    print("\n=== 开始CFPU重建 ===")
    t0 = time.perf_counter()
    
    try:
        potential, X, Y, Z = cfpurecon(
            points, normals, patches, GRID_M, 
            KERNEL_CONFIG, REGULARIZATION_CONFIG, JOBS
        )
        t1 = time.perf_counter()
        
        print(f"重建完成! 耗时: {t1 - t0:.6f} 秒")
        print(f"网格形状: {X.shape}")
        
    except Exception as e:
        print(f"重建失败: {e}")
        return
    
    # 保存日志
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"recon__{MODEL_NAME or 'default'}__m{GRID_M}.log"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] model={MODEL_NAME or 'default'} m={GRID_M} threads={used_workers} mode={mode}\n")
        f.write(f"points_shape={points.shape} normals_shape={normals.shape} patches_shape={patches.shape}\n")
        f.write(f"bounds_min={bounds_min.tolist()} bounds_max={bounds_max.tolist()}\n")
        f.write(f"cfpurecon_time_sec={t1 - t0:.6f}\n")
    
    print(f"日志已保存: {log_path}")
    
    # 可视化结果
    print("\n=== 开始可视化 ===")
    visualize_results(potential, X, Y, Z, patches)


def visualize_results(potential, X, Y, Z, patches):
    """
    可视化重建结果
    
    参数:
        potential: 势场数据
        X, Y, Z: 网格坐标
        patches: 补丁点云
    """
    try:
        # 创建结构化网格
        sg = pv.StructuredGrid(X, Y, Z)
        sg['potential'] = potential.ravel(order='F')
        
        # 提取等值面
        iso = sg.contour(isosurfaces=[VISUALIZATION_CONFIG['isosurface_value']])
        
        # 创建绘图器
        plotter = pv.Plotter(shape=(1, 2))
        
        # 左侧子图：仅显示等值面
        plotter.subplot(0, 0)
        plotter.add_mesh(
            iso, 
            color=VISUALIZATION_CONFIG['mesh_color'], 
            specular=VISUALIZATION_CONFIG['mesh_specular'], 
            smooth_shading=VISUALIZATION_CONFIG['smooth_shading'], 
            opacity=VISUALIZATION_CONFIG['mesh_opacity']
        )
        plotter.add_axes()
        plotter.add_title("重建等值面")
        
        # 右侧子图：等值面 + 补丁点云
        plotter.subplot(0, 1)
        plotter.add_mesh(
            iso, 
            color=VISUALIZATION_CONFIG['mesh_color'], 
            specular=VISUALIZATION_CONFIG['mesh_specular'], 
            smooth_shading=VISUALIZATION_CONFIG['smooth_shading'], 
            opacity=VISUALIZATION_CONFIG['mesh_opacity']
        )
        plotter.add_points(
            patches, 
            color=VISUALIZATION_CONFIG['points_color'], 
            render_points_as_spheres=True, 
            point_size=VISUALIZATION_CONFIG['points_size']
        )
        plotter.add_axes()
        plotter.add_title("重建等值面 + 补丁点云")
        
        # 链接视图并显示
        plotter.link_views()
        plotter.show()
        
        print("可视化完成!")
        
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == '__main__':
    main()