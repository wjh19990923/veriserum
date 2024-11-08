import nibabel as nib


def show_info_nii(nii_file_path):
    # 替换为你的NIfTI文件路径
    nii_file_path = rf"C:\Users\LAB Admin\Desktop\Segmentation.nii"
    print(nii_file_path)
    # 加载NIfTI文件
    nii_image = nib.load(nii_file_path)

    # 获取图像数据
    image_data = nii_image.get_fdata()

    # 打印图像维度
    print("Image dimensions:", image_data.shape)

    # 获取并打印NIfTI文件的头信息
    header = nii_image.header
    print("Header information:")
    print(header)
    # 例如，获取并打印体素的大小
    print("Voxel size:", header.get_zooms())
    breakpoint()
    # 注意：使用.get_fdata()加载整个数据集可能会占用大量内存，特别是对于大型3D或4D图像数据。
    # 如果只需要访问图像的一小部分，考虑使用slicing（nii_image.dataobj[..., slice]）来减少内存使用。
