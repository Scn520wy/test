import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    predictions = []  # 用于存储所有融合后的预测结果
    for i in tqdm(range(len(label))):
        # l = label[i]
        # 将 l 转换为标量标签
        l = np.argmax(label[i])
        _, r_11 = r1[i]
        _, r_22 = r2[i]
        _, r_33 = r3[i]
        _, r_44 = r4[i]
        _, r_55 = r5[i]
        _, r_66 = r6[i]
        _, r_77 = r7[i]
        _, r_88 = r8[i]
        _, r_99 = r9[i]
        _, r_1010 = r10[i]
        _, r_1111 = r11[i]
        _, r_1212 = r12[i]
        _, r_1313 = r13[i]
        _, r_1414 = r14[i]
        _, r_1515 = r15[i]
        _, r_1616 = r16[i]
        _, r_1717 = r17[i]
        _, r_1818 = r18[i]
        _, r_1919 = r19[i]
        _, r_2020 = r20[i]
        r_2121 = r21[i]

        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r_1010 * weights[9] \
            + r_1111 * weights[10] \
            + r_1212 * weights[11] \
            + r_1313 * weights[12] \
            + r_1414 * weights[13] \
            + r_1515 * weights[14] \
            + r_1616 * weights[15] \
            + r_1717 * weights[16] \
            + r_1818 * weights[17] \
            + r_1919 * weights[18] \
            + r_2020 * weights[19] \
            + r_2121 * weights[20]

        # 记录每个样本的最终融合分数
        predictions.append(r)

        # 将预测结果转换为 NumPy 数组并保存
    predictions = np.array(predictions)
    np.save("predA.npy", predictions)

    return -np.mean(np.argmax(predictions, axis=1) == np.argmax(label, axis=1))

    #     # 将融合后的预测添加到列表中
    #     predictions.append(r)
    #
    #     r = np.argmax(r)
    #     right_num += int(r == int(l))
    #     total_num += 1
    # acc = right_num / total_num
    # print(acc)
    #
    # # 将预测结果列表转换为 NumPy 数组并返回
    # predictions = np.array(predictions)
    # np.save("pred.npy", predictions)
    #
    # return -acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default = 'V2')
    parser.add_argument('--mixformer_J_Score', default = 'work_dir/uav/mix_Bjoint_A/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_B_Score', default = 'work_dir/uav/mix_Bbone_A/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_JM_Score', default = 'work_dir/uav/mix_Bjoint_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_BM_Score', default = 'work_dir/uav/mix_Bbone_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J2d_Score', default = 'work_dir/uav/ctr2d_Bjoint_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B2d_Score', default = 'work_dir/uav/ctr2d_Bbone_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM2d_Score', default = 'work_dir/uav/ctr2d_Bjoint_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM2d_Score', default = 'work_dir/uav/ctr2d_Bbone_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J3d_Score', default = 'work_dir/uav/ctr3d_Bjoint_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B3d_Score', default = 'work_dir/uav/ctr3d_Bbone_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM3d_Score', default = 'work_dir/uav/ctr3d_Bjoint_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM3d_Score', default = 'work_dir/uav/ctr3d_Bbone_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_J2d_Score', default = 'work_dir/uav/tdgcn_Bjoint_A/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_B2d_Score', default = 'work_dir/uav/tdgcn_Bbone_A/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_JM2d_Score', default = 'work_dir/uav/tdgcn_Bjoint_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_BM2d_Score', default = 'work_dir/uav/tdgcn_Bbone_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_J2d_Score', default = 'work_dir/uav/mstgcn_Bjoint_A/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_B2d_Score', default = 'work_dir/uav/mstgcn_Bbone_A/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_JM2d_Score', default = 'work_dir/uav/mstgcn_Bjoint_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_BM2d_Score', default = 'work_dir/uav/mstgcn_Bbone_motion_A/epoch1_test_score.pkl')
    parser.add_argument('--tegcn_jb_Score', default='work_dir/uav/tegcn_jb_A/epoch1_test_score.npy')
    arg = parser.parse_args()

    benchmark = arg.benchmark
    if benchmark == 'V1':
        npz_data = np.load('data/uav/UAV2_joint.npz')
        label = npz_data['y_test']
    else:
        assert benchmark == 'V2'
        npz_data = np.load('data/uav/UAV2_joint.npz')
        label = npz_data['y_test']

    with open(arg.mixformer_J_Score, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.mixformer_B_Score, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.mixformer_JM_Score, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.mixformer_BM_Score, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.ctrgcn_J2d_Score, 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(arg.ctrgcn_B2d_Score, 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    with open(arg.ctrgcn_JM2d_Score, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(arg.ctrgcn_BM2d_Score, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(arg.ctrgcn_J3d_Score, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(arg.ctrgcn_B3d_Score, 'rb') as r10:
        r10 = list(pickle.load(r10).items())

    with open(arg.ctrgcn_JM3d_Score, 'rb') as r11:
        r11 = list(pickle.load(r11).items())

    with open(arg.ctrgcn_BM3d_Score, 'rb') as r12:
        r12 = list(pickle.load(r12).items())

    with open(arg.tdgcn_J2d_Score, 'rb') as r13:
        r13 = list(pickle.load(r13).items())

    with open(arg.tdgcn_B2d_Score, 'rb') as r14:
        r14 = list(pickle.load(r14).items())

    with open(arg.tdgcn_JM2d_Score, 'rb') as r15:
        r15 = list(pickle.load(r15).items())

    with open(arg.tdgcn_BM2d_Score, 'rb') as r16:
        r16 = list(pickle.load(r16).items())

    with open(arg.mstgcn_J2d_Score, 'rb') as r17:
        r17 = list(pickle.load(r17).items())

    with open(arg.mstgcn_B2d_Score, 'rb') as r18:
        r18 = list(pickle.load(r18).items())

    with open(arg.mstgcn_JM2d_Score, 'rb') as r19:
        r19 = list(pickle.load(r19).items())

    with open(arg.mstgcn_BM2d_Score, 'rb') as r20:
        r20 = list(pickle.load(r20).items())

    r21 = np.load(arg.tegcn_jb_Score)

    space = [(0, 1.6) for i in range(21)]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))
