import os
import scipy.io as scio
import torch
import numpy as np

def Save_mat(epoch , output_dim , datasets , query_labels , retrieval_labels , query_img , retrieval_img , save_dir='..' , mode_name="CrossHash",map=0):
    '''
    save_dir: 保存文件的目录路径
    output_dim: 输出维度
    datasets: 数据集名称
    query_labels: 查询图像的标签信息（numpy数组）
    retrieval_labels: 检索图像的标签信息（numpy数组）
    query_img: 查询图像的数据（numpy数组）
    retrieval_img: 检索图像的数据（numpy数组）
    mode_name: 模型的名称
    '''
    save_dir = os.path.join(save_dir , f"Hash_code_and_label_{mode_name}_{datasets}")
    os.makedirs(save_dir,exist_ok=True)

    query_img = query_img.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()

    query_label = query_labels.numpy()
    retrieval_label = retrieval_labels.numpy()

    result_dict = {
        'c' : query_img ,
        'r_img' : retrieval_img ,
        'q_l' : query_label ,
        'r_l' : retrieval_label
    }

    filename = os.path.join(save_dir, f"{map}-{output_dim}-{epoch}-{datasets}-{mode_name}.mat")
    scio.savemat(filename, result_dict)

