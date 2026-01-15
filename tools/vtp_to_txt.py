#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK文件转换脚本
将VTK PolyData文件转换为节点和法向量文本文件
"""

import os
import sys
import numpy as np
from pathlib import Path

# 检查必要的依赖库
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("错误: 需要安装vtk库。请运行: pip install vtk")
    sys.exit(1)


def read_vtk_file(vtk_file_path):
    """
    读取VTK文件并提取节点坐标和法向量
    
    参数:
        vtk_file_path: VTK文件路径
        
    返回:
        points: 节点坐标数组 (N, 3)
        normals: 法向量数组 (N, 3)
    """
    # 检查文件是否存在
    if not os.path.exists(vtk_file_path):
        raise FileNotFoundError(f"VTK文件不存在: {vtk_file_path}")
    
    # 读取VTK文件
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    
    # 获取多边形数据
    polydata = reader.GetOutput()
    
    # 检查数据是否有效
    if polydata is None:
        raise ValueError(f"无法读取VTK文件: {vtk_file_path}")
    
    # 提取节点坐标
    points = polydata.GetPoints()
    if points is None:
        raise ValueError("VTK文件中没有节点数据")
    
    points_array = points.GetData()
    points_np = vtk_to_numpy(points_array)
    
    # 提取法向量
    normals = polydata.GetPointData().GetNormals()
    if normals is None:
        # 如果没有点法向量，尝试获取单元法向量
        cell_normals = polydata.GetCellData().GetNormals()
        if cell_normals is not None:
            # 将单元法向量转换为点法向量
            cell_normals_np = vtk_to_numpy(cell_normals)
            
            # 创建点法向量数组
            point_normals_np = np.zeros_like(points_np)
            
            # 对于每个单元，将其法向量添加到所有顶点
            for i in range(polydata.GetNumberOfCells()):
                cell = polydata.GetCell(i)
                point_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                for point_id in point_ids:
                    point_normals_np[point_id] += cell_normals_np[i]
            
            # 归一化法向量
            norms = np.linalg.norm(point_normals_np, axis=1)
            norms[norms == 0] = 1  # 避免除以零
            normals_np = point_normals_np / norms[:, np.newaxis]
        else:
            # 如果既没有点法向量也没有单元法向量，计算近似法向量
            print(f"警告: {vtk_file_path} 中没有法向量数据，将计算近似法向量")
            normals_np = compute_approximate_normals(polydata)
    else:
        normals_np = vtk_to_numpy(normals)
    
    return points_np, normals_np


def compute_approximate_normals(polydata):
    """
    计算多边形数据的近似法向量
    
    参数:
        polydata: VTK多边形数据
        
    返回:
        normals: 近似法向量数组 (N, 3)
    """
    # 创建法向量过滤器
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.SplittingOff()
    normals_filter.Update()
    
    # 获取计算后的法向量
    output_polydata = normals_filter.GetOutput()
    normals = output_polydata.GetPointData().GetNormals()
    
    if normals is None:
        # 如果计算失败，返回默认法向量
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        return np.ones((num_points, 3)) * 0.57735  # 单位向量
    
    return vtk_to_numpy(normals)


def save_nodes_and_normals(points, normals, output_dir, base_name):
    """
    保存节点和法向量到文本文件
    
    参数:
        points: 节点坐标数组
        normals: 法向量数组
        output_dir: 输出目录
        base_name: 基础文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 简化文件名：删除_surface_cellnormals_部分
    # 例如：Sphere_surface_cellnormals -> Sphere
    simple_name = base_name.replace('_surface_cellnormals', '')
    
    # 保存节点文件
    nodes_file = os.path.join(output_dir, f"{simple_name}_nodes.txt")
    np.savetxt(nodes_file, points, fmt='%.6f', delimiter=' ')
    print(f"节点文件已保存: {nodes_file}")
    
    # 保存法向量文件
    normals_file = os.path.join(output_dir, f"{simple_name}_normals.txt")
    np.savetxt(normals_file, normals, fmt='%.6f', delimiter=' ')
    print(f"法向量文件已保存: {normals_file}")


def process_vtk_directory(input_vtp_dir, output_txt_dir):
    """
    处理整个VTK目录结构
    
    参数:
        input_vtp_dir: 输入VTK目录路径
        output_txt_dir: 输出文本目录路径
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_vtp_dir):
        print(f"错误: 输入目录不存在: {input_vtp_dir}")
        return
    
    # 遍历输入目录的子目录
    for geometry_type in os.listdir(input_vtp_dir):
        geometry_path = os.path.join(input_vtp_dir, geometry_type)
        
        if not os.path.isdir(geometry_path):
            continue
        
        print(f"处理几何类型: {geometry_type}")
        
        # 创建对应的输出目录
        output_geometry_dir = os.path.join(output_txt_dir, geometry_type)
        os.makedirs(output_geometry_dir, exist_ok=True)
        
        # 处理该目录下的所有VTK文件
        for vtk_file in os.listdir(geometry_path):
            if vtk_file.endswith('.vtp'):
                vtk_file_path = os.path.join(geometry_path, vtk_file)
                
                # 提取基础文件名（去掉扩展名）
                base_name = os.path.splitext(vtk_file)[0]
                
                try:
                    print(f"处理文件: {vtk_file}")
                    
                    # 读取VTK文件
                    points, normals = read_vtk_file(vtk_file_path)
                    
                    # 保存节点和法向量
                    save_nodes_and_normals(points, normals, output_geometry_dir, base_name)
                    
                    print(f"成功处理: {vtk_file} - 节点数: {len(points)}")
                    
                except Exception as e:
                    print(f"处理文件 {vtk_file} 时出错: {e}")


def process_single_vtk_file(vtk_file_path, output_txt_dir):
    """
    处理单个VTK文件
    
    参数:
        vtk_file_path: VTK文件路径
        output_txt_dir: 输出文本目录路径
    """
    # 提取文件名和目录信息
    vtk_file_name = os.path.basename(vtk_file_path)
    vtk_dir_name = os.path.basename(os.path.dirname(vtk_file_path))
    
    # 创建对应的输出目录
    output_dir = os.path.join(output_txt_dir, vtk_dir_name)
    
    # 提取基础文件名
    base_name = os.path.splitext(vtk_file_name)[0]
    
    try:
        print(f"处理文件: {vtk_file_path}")
        
        # 读取VTK文件
        points, normals = read_vtk_file(vtk_file_path)
        
        # 保存节点和法向量
        save_nodes_and_normals(points, normals, output_dir, base_name)
        
        print(f"成功处理: {vtk_file_name} - 节点数: {len(points)}")
        
    except Exception as e:
        print(f"处理文件 {vtk_file_name} 时出错: {e}")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent
    input_vtp_dir = project_root / "data" / "input_vtp"
    output_txt_dir = project_root / "data" / "input_txt"
    
    print("VTK文件转换脚本")
    print("=" * 50)
    
    # 处理整个input_vtp目录
    print("处理整个input_vtp目录")
    process_vtk_directory(str(input_vtp_dir), str(output_txt_dir))
    
    print("=" * 50)
    print("转换完成!")


if __name__ == "__main__":
    main()