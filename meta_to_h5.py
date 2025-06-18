import argparse
import os
import numpy as np
import pandas as pd
import pdb

def config():
    parser = argparse.ArgumentParser(description='SHREC meta')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='data folder path')
    parser.add_argument('--dataset',
                        required=True,
                        choices=['13', '14'],
                        help='dataset')
    args = parser.parse_args()
    return args


def get_df_sketches(sk_path):
    split = []
    cat = []
    paths = []
    ids = []
    # for root, _, files in os.walk(os.path.join(data_dir, sk_path)):
    for root, _, files in os.walk(sk_path):
        for f in files:
            # pdb.set_trace()
            if f[-3:] == 'png':
                split.append(root.split(os.path.sep)[-1])
                cat.append(root.split(os.path.sep)[-2])
                ids.append(os.path.splitext(f)[0])
                paths.append(os.path.join(sk_path, cat[-1], split[-1], f))

    df = pd.DataFrame(data={'cat': cat, 'split': split, 'id': ids},
                      index=paths)
    return df


def get_df_models(cad_anno, cad_path, pcd_path):
    # read meta file
    fpath = cad_anno

    with open(fpath, 'r') as f:
        content = f.readlines()

    labels = {}
    current_cat = ''
    for line in content[3:]:
        line = line.strip('\r\n')
        line = line.strip('\t')
        line = line.strip()
        if len(line.split()) == 3:
            current_cat = line.split()[0]
        elif line != '':
            labels[line] = current_cat

    # read model folder
    cat = []
    ids = []
    paths = []
    pcd_paths = []
    for root, _, files in os.walk(cad_path):
        for f in files:
            if f[-3:] == 'off':
                ids.append(os.path.splitext(f)[0])
                cat.append(labels[ids[-1][1:]])
                paths.append(os.path.join(cad_path, f))
                pcd_paths.append(os.path.join(pcd_path, f.replace('.off', '.npy')))

    df = pd.DataFrame(data={'cat': cat, 'id': ids},
                      index=paths)
    df_pcd = pd.DataFrame(data={'cat': cat, 'id': ids},
                      index=pcd_paths)
    return df, df_pcd


def split_models(df_sk, df_cad, df_pcds):
    vv, cc = np.unique(df_cad['cat'], return_counts=True)
    coi = vv[cc > 50]
    n_coi = cc[cc > 50]

    new_df_sk = df_sk.loc[df_sk['cat'].isin(coi)].copy()
    new_df_cad = df_cad.loc[df_cad['cat'].isin(coi)].copy()
    new_df_pcds = df_pcds.loc[df_pcds['cat'].isin(coi)].copy()

    # randomly split instances
    np.random.seed(1234)
    new_df_cad.loc[:, 'split'] = 'train'
    for c, n in zip(coi, n_coi):
        to_select = int(np.floor(n * 0.2))
        subset = new_df_cad.loc[new_df_cad['cat'] == c, 'id']
        id_to_select = np.random.choice(subset, size=to_select, replace=False)
        new_df_cad.loc[new_df_cad['id'].isin(id_to_select), 'split'] = 'test'
        new_df_pcds.loc[new_df_pcds['id'].isin(id_to_select), 'split'] = 'test'
    return new_df_sk, new_df_cad, new_df_pcds


def main():
    args = config()

    if args.dataset == '14':
        base = 'SHREC14'

        # get sketch labels
        # sk_path = os.path.join(base, 'SHREC14LSSTB_SKETCHES', 'SHREC14LSSTB_SKETCHES')        
        sk_path = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES'
        # df_sk = get_df_sketches(args.data_dir, sk_path)
        df_sk = get_df_sketches(sk_path)

        # cad_path = os.path.join(base, 'SHREC14LSSTB_TARGET_MODELS')
        cad_path = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS'
        # eval_path = os.path.join(base, 'SHREC14_Sketch_Evaluation_CVIU')
        # cad_anno = os.path.join(eval_path, 'SHREC14_SBR_Model.cla')
        cad_anno = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_models_train.cla'
        pcd_path = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np'

    # get cad labels
    df_cad, df_pcds = get_df_models(cad_anno, cad_path, pcd_path)

    # save_dir = os.path.join('labels', base)
    save_dir = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/orig'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # pdb.set_trace()

    df_sk.to_hdf(os.path.join(save_dir, 'sk_orig.hdf5'), 'sk')
    df_cad.to_hdf(os.path.join(save_dir, 'cad_orig.hdf5'), 'cad')
    df_pcds.to_hdf(os.path.join(save_dir, 'pcds_orig.hdf5'), 'pcd')

    with open(os.path.join(save_dir, 'cad.txt'), 'w') as f:
        for item in df_cad.index:
            f.write('%s\n' % item)

    if args.dataset == '14':
        # split between train and test cad models
        # following Qi et al BMVC 2018
        new_df_sk, new_df_cad, new_df_pcd = split_models(df_sk, df_cad, df_pcds)

        # save_dir = os.path.join('labels', 'PART-' + base)
        save_dir = '/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        new_df_sk.to_hdf(os.path.join(save_dir, 'sk_orig.hdf5'), 'sk')
        new_df_cad.to_hdf(os.path.join(save_dir, 'cad_orig.hdf5'), 'cad')
        new_df_pcd.to_hdf(os.path.join(save_dir, 'pcds_orig.hdf5'), 'pcd')

        with open(os.path.join(save_dir, 'pcd_split.txt'), 'w') as f:
            for item in new_df_pcd.index:
                f.write('%s\n' % item)

        with open(os.path.join(save_dir, 'pcd_classes_split.txt'), 'w') as f:
            for item in new_df_pcd.cat.unique():
                f.write('%s\n' % item)


if __name__ == "__main__":
    main()