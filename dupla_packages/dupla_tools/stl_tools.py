from stl import mesh
import numpy as np

def center_stl(your_mesh):
    # 加载STL文件
    your_mesh = mesh.Mesh.from_file(rf'C:\Users\LAB Admin\Desktop\test\Segmentation_tibia_bill.stl')

    # 计算当前的中心
    current_center = np.mean(your_mesh.points.reshape(-1, 9), axis=0)[0:3]

    # 计算需要移动的距离（例如，移动到原点）
    translation = -current_center

    # 移动模型
    your_mesh.x += translation[0]
    your_mesh.y += translation[1]
    your_mesh.z += translation[2]

    # 保存修改后的STL
    your_mesh.save('modified_file.stl')
