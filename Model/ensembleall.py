import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    mix_r1 = "work_dir/uav/mix_Bjoint"
    mix_r2 = "work_dir/uav/mix_Bbone"
    mix_r3 = "work_dir/uav/mix_Bjoint_motion"
    mix_r4 = "work_dir/uav/mix_Bbone_motion"

    ctr2d_r1 = "work_dir/uav/ctr2d_Bjoint"
    ctr2d_r2 = "work_dir/uav/ctr2d_Bbone"
    ctr2d_r3 = "work_dir/uav/ctr2d_Bjoint_motion"
    ctr2d_r4 = "work_dir/uav/ctr2d_Bbone_motion"

    ctr3d_r1 = "work_dir/uav/ctr3d_Bjoint"
    ctr3d_r2 = "work_dir/uav/ctr3d_Bbone"
    ctr3d_r3 = "work_dir/uav/ctr3d_Bjoint_motion"
    ctr3d_r4 = "work_dir/uav/ctr3d_Bbone_motion"

    td_r1 = "work_dir/uav/tdgcn_Bjoint"
    td_r2 = "work_dir/uav/tdgcn_Bbone"
    td_r3 = "work_dir/uav/tdgcn_Bjoint_motion"
    td_r4 = "work_dir/uav/tdgcn_Bbone_motion"

    mst_r1 = "work_dir/uav/mstgcn_Bjoint"
    mst_r2 = "work_dir/uav/mstgcn_Bbone"
    mst_r3 = "work_dir/uav/mstgcn_Bjoint_motion"
    mst_r4 = "work_dir/uav/mstgcn_Bbone_motion"

    tr1 = "work_dir/uav/tegcn_Bjoint_bone"

    with open('./data/test_B_label.npy', 'rb') as f:
        label = np.load(f)

    with open(os.path.join(mix_r1, 'epoch1_test_score.pkl'), 'rb') as mix_r1:
        mix_r1 = list(pickle.load(mix_r1).items())
    with open(os.path.join(mix_r2, 'epoch1_test_score.pkl'), 'rb') as mix_r2:
        mix_r2 = list(pickle.load(mix_r2).items())
    with open(os.path.join(mix_r3, 'epoch1_test_score.pkl'), 'rb') as mix_r3:
        mix_r3 = list(pickle.load(mix_r3).items())
    with open(os.path.join(mix_r4, 'epoch1_test_score.pkl'), 'rb') as mix_r4:
        mix_r4 = list(pickle.load(mix_r4).items())

    with open(os.path.join(ctr2d_r1, 'epoch1_test_score.pkl'), 'rb') as ctr2d_r1:
        ctr2d_r1 = list(pickle.load(ctr2d_r1).items())
    with open(os.path.join(ctr2d_r2, 'epoch1_test_score.pkl'), 'rb') as ctr2d_r2:
        ctr2d_r2 = list(pickle.load(ctr2d_r2).items())
    with open(os.path.join(ctr2d_r3, 'epoch1_test_score.pkl'), 'rb') as ctr2d_r3:
        ctr2d_r3 = list(pickle.load(ctr2d_r3).items())
    with open(os.path.join(ctr2d_r4, 'epoch1_test_score.pkl'), 'rb') as ctr2d_r4:
        ctr2d_r4 = list(pickle.load(ctr2d_r4).items())

    with open(os.path.join(ctr3d_r1, 'epoch1_test_score.pkl'), 'rb') as ctr3d_r1:
        ctr3d_r1 = list(pickle.load(ctr3d_r1).items())
    with open(os.path.join(ctr3d_r2, 'epoch1_test_score.pkl'), 'rb') as ctr3d_r2:
        ctr3d_r2 = list(pickle.load(ctr3d_r2).items())
    with open(os.path.join(ctr3d_r3, 'epoch1_test_score.pkl'), 'rb') as ctr3d_r3:
        ctr3d_r3 = list(pickle.load(ctr3d_r3).items())
    with open(os.path.join(ctr3d_r4, 'epoch1_test_score.pkl'), 'rb') as ctr3d_r4:
        ctr3d_r4 = list(pickle.load(ctr3d_r4).items())

    with open(os.path.join(td_r1, 'epoch1_test_score.pkl'), 'rb') as td_r1:
        td_r1 = list(pickle.load(td_r1).items())
    with open(os.path.join(td_r2, 'epoch1_test_score.pkl'), 'rb') as td_r2:
        td_r2 = list(pickle.load(td_r2).items())
    with open(os.path.join(td_r3, 'epoch1_test_score.pkl'), 'rb') as td_r3:
        td_r3 = list(pickle.load(td_r3).items())
    with open(os.path.join(td_r4, 'epoch1_test_score.pkl'), 'rb') as td_r4:
        td_r4 = list(pickle.load(td_r4).items())

    with open(os.path.join(mst_r1, 'epoch1_test_score.pkl'), 'rb') as mst_r1:
        mst_r1 = list(pickle.load(mst_r1).items())
    with open(os.path.join(mst_r2, 'epoch1_test_score.pkl'), 'rb') as mst_r2:
        mst_r2 = list(pickle.load(mst_r2).items())
    with open(os.path.join(mst_r3, 'epoch1_test_score.pkl'), 'rb') as mst_r3:
        mst_r3 = list(pickle.load(mst_r3).items())
    with open(os.path.join(mst_r4, 'epoch1_test_score.pkl'), 'rb') as mst_r4:
        mst_r4 = list(pickle.load(mst_r4).items())

    # tr1 = np.load(os.path.join(tr1, 'epoch1_test_score.npy'))
    with open(os.path.join(tr1, 'epoch1_test_score.pkl'), 'rb') as tr1:
        tr1 = list(pickle.load(tr1).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0

    total_num = 0
    right_num = 0

    optimal_weights = [1.6, 1.6, 1.0878508856182485, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6, 1.6, 0.8425952903270515, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6]
    alpha = optimal_weights[:4]  # mix模型的权重
    alpha2 = optimal_weights[4:8]  # ctr2d模型的权重
    alpha3 = optimal_weights[8:12]  # ctr3d模型的权重
    alpha4 = optimal_weights[12:16]  # td模型的权重
    alpha5 = optimal_weights[16:20]  # mst模型的权重
    alpha6 = optimal_weights[20:21]  # mst模型的权重

    # 创建一个列表来存储融合结果
    fused_results = []

    for i in tqdm(range(len(label))):
        l = label[i]
        _, mix_r11 = mix_r1[i]
        _, mix_r22 = mix_r2[i]
        _, mix_r33 = mix_r3[i]
        _, mix_r44 = mix_r4[i]

        _, ctr2d_r11 = ctr2d_r1[i]
        _, ctr2d_r22 = ctr2d_r2[i]
        _, ctr2d_r33 = ctr2d_r3[i]
        _, ctr2d_r44 = ctr2d_r4[i]

        _, ctr3d_r11 = ctr3d_r1[i]
        _, ctr3d_r22 = ctr3d_r2[i]
        _, ctr3d_r33 = ctr3d_r3[i]
        _, ctr3d_r44 = ctr3d_r4[i]

        _, td_r11 = td_r1[i]
        _, td_r22 = td_r2[i]
        _, td_r33 = td_r3[i]
        _, td_r44 = td_r4[i]

        _, mst_r11 = mst_r1[i]
        _, mst_r22 = mst_r2[i]
        _, mst_r33 = mst_r3[i]
        _, mst_r44 = mst_r4[i]

        _, tr11 = tr1[i]

        result1 = mix_r11 * alpha[0] + mix_r22 * alpha[1] + mix_r33 * alpha[2] + mix_r44 * alpha[3]
        result2 = ctr2d_r11 * alpha2[0] + ctr2d_r22 * alpha2[1] + ctr2d_r33 * alpha2[2] + ctr2d_r44 * alpha2[3]
        result3 = ctr3d_r11 * alpha3[0] + ctr3d_r22 * alpha3[1] + ctr3d_r33 * alpha3[2] + ctr3d_r44 * alpha3[3]
        result4 = td_r11 * alpha4[0] + td_r22 * alpha4[1] + td_r33 * alpha4[2] + td_r44 * alpha4[3]
        result5 = mst_r11 * alpha5[0] + mst_r22 * alpha5[1] + mst_r33 * alpha5[2] + mst_r44 * alpha5[3]
        # result6 = tr1[i] * alpha3[0]  # 修改为索引 tr1[i]
        result6 = tr11 * alpha6[0]

        r = result1 + result2 + result3 + result4 + result5 + result6

        # 将融合结果添加到列表中
        fused_results.append(r)

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))
        total_num += 1

    # 将融合结果列表转换为numpy数组
    fused_results_array = np.array(fused_results)

    # 保存融合结果为prey.npy文件
    np.save('pred.npy', fused_results_array)

    acc = right_num / total_num
    print(acc, alpha)
    if acc > best:
        best = acc
        best_alpha = alpha
    acc5 = right_num_5 / total_num

    print(best, best_alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to prey.npy')