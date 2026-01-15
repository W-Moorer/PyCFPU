import argparse
import numpy as np
import pymeshlab as ml
import os
import tempfile
from pathlib import Path


def read_txt_pointcloud(txt_file):
    """
    读取txt文件中的点云数据
    """
    return np.loadtxt(txt_file)


def write_txt_point_cloud(points, filename):
    """
    将点云保存为txt文件
    """
    np.savetxt(filename, points, fmt="%.6f")


def poisson_disk_sampling_pymeshlab(points, target_num):
    """
    使用PyMeshLab进行泊松盘采样
    
    参数:
        points: 输入点云，形状为(N, 3)的numpy数组
        target_num: 目标采样点数
        
    返回:
        sampled_points: 采样后的点云
    """
    try:
        # 创建临时PLY文件供PyMeshLab使用
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as tmpfile:
            tmp_ply = tmpfile.name
            # 写入PLY格式
            with open(tmp_ply, 'w') as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                np.savetxt(f, points, fmt="%.6f %.6f %.6f")
        
        # 使用PyMeshLab进行泊松盘采样
        ms = ml.MeshSet()
        ms.load_new_mesh(tmp_ply)
        
        # 执行泊松盘采样（Best Sample模式）
        ms.generate_simplified_point_cloud(
            samplenum=target_num,
            bestsampleflag=True,
            bestsamplepool=10,
            exactnumflag=False
        )
        
        sampled_points = ms.current_mesh().vertex_matrix()
        
        # 清理临时文件
        os.remove(tmp_ply)
        
        return sampled_points
        
    except Exception as e:
        print(f"泊松盘采样失败: {e}")
        # 确保临时文件被清理
        if 'tmp_ply' in locals() and os.path.exists(tmp_ply):
            os.remove(tmp_ply)
        return None


def process_single_file(txt_file, divisor):
    """
    处理单个txt文件
    
    参数:
        txt_file: 输入txt文件路径
        divisor: 采样除数（输出点数 = 输入点数 / divisor）
        
    返回:
        success: 处理是否成功
    """
    try:
        print(f"处理文件: {txt_file}")
        
        # 读取点云数据
        points = read_txt_pointcloud(txt_file)
        if points is None:
            return False
        
        print(f"  原始点数: {len(points)}")
        
        # 计算目标采样点数
        target_num = max(10, len(points) // divisor)  # 最少保留10个点
        print(f"  目标采样点数: {target_num} (除数: {divisor})")
        
        # 执行泊松盘采样
        sampled_points = poisson_disk_sampling_pymeshlab(points, target_num)
        if sampled_points is None:
            return False
        
        print(f"  实际采样点数: {len(sampled_points)}")
        
        # 生成输出文件名：将_nodes替换为_patches
        output_file = txt_file.replace("_nodes.txt", "_patches.txt")
        
        # 保存采样结果
        write_txt_point_cloud(sampled_points, output_file)
        print(f"  采样结果已保存: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {txt_file} 时出错: {e}")
        return False


def batch_process_directory(input_dir, divisor):
    """
    批量处理目录中的所有node文件
    
    参数:
        input_dir: 输入目录路径
        divisor: 采样除数
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print(f"开始处理目录: {input_dir}")
    print(f"采样除数: {divisor}")
    print("=" * 60)
    
    total_files = 0
    success_files = 0
    
    # 递归遍历所有子目录
    for txt_file in input_path.rglob("*_nodes.txt"):
        # 跳过已经生成的patch文件
        if "_patches" in str(txt_file) or "_poisson" in str(txt_file):
            continue
        
        total_files += 1
        
        # 处理单个文件
        if process_single_file(str(txt_file), divisor):
            success_files += 1
        
        print("-" * 40)
    
    print("=" * 60)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_files}")
    print(f"失败文件: {total_files - success_files}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="泊松盘采样 - 支持单文件和批量处理")
    parser.add_argument("input", nargs='?', type=str, help="输入txt文件或目录路径")
    parser.add_argument("-n", "--divisor", type=int, default=10, 
                       help="采样除数（输出点数 = 输入点数 / divisor），默认: 10")
    parser.add_argument("-b", "--batch", action="store_true",
                       help="批量处理模式，处理整个目录")
    
    args = parser.parse_args()
    
    # 检查PyMeshLab是否可用
    try:
        import pymeshlab as ml
        print("PyMeshLab导入成功")
    except ImportError:
        print("错误: 需要安装pymeshlab库。请运行: pip install pymeshlab")
        return
    
    if args.batch:
        # 批量处理模式
        if not args.input:
            args.input = "data/input_txt"
        batch_process_directory(args.input, args.divisor)
    else:
        # 单文件处理模式
        if not args.input:
            print("错误: 单文件模式需要指定输入文件路径")
            return
        
        # 读取原始点云
        print(f"读取点云文件: {args.input}")
        x = read_txt_pointcloud(args.input)
        print(f"原始点数: {x.shape[0]}")

        target_num = max(1, x.shape[0] // args.divisor)
        print(f"目标采样点数: {target_num}")

        # 执行泊松盘采样
        sampled_points = poisson_disk_sampling_pymeshlab(x, target_num)
        if sampled_points is None:
            print("采样失败")
            return
            
        print(f"采样后点数: {sampled_points.shape[0]}")

        # 生成输出文件名：将_nodes替换为_patches
        output_file = args.input.replace("_nodes.txt", "_patches.txt")
        write_txt_point_cloud(sampled_points, output_file)
        print(f"采样点云保存到: {output_file}")


if __name__ == "__main__":
    main()