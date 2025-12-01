#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=airawatcp
#SBATCH --time=48:00:00        
#SBATCH --error=outputs/job.%J.err     # Error log (J = job ID)
#SBATCH --output=outputs/job.%J.out    # Output log

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes." 
echo "Running $SLURM_NTASKS tasks."
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_skt_beit.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_skt_vit_vpt.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_pt_clip_map.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_conti_model.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_cross_lim_models.py

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_cor_map_cross.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_cross_lim_meta.py

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_conti_model.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_conti_model.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_cnn.py

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_hyp.py
# python /nlsasfs/home/neol/rushar/dars_sept.py
python /nlsasfs/home/neol/rushar/dars_oct.py

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_euc_zero.py

# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_perci_cross.py
# python /nlsasfs/home/neol/rushar/scripts/img_to_pcd/main_cross_attn_samp.py


# python train_cls.py train \
# 	--dataroot /nlsasfs/home/neol/rushar/meshmae/MeshMAE/dataset_hdf5/ \
# 	--batch_size 32 --augment_scale --n_classes 40 \
# 	--channels 10 --patch_size 64 --n_epoch 100 \
# 	--name "manifoldBase_fine_sn" \
# 	--weight_decay 0.05 \
# 	--lr 1e-4 --optim "adamw" \
# 	--depth 12 \
# 	--heads 12 \
# 	--lr_milestones "none" \
# 	--encoder_depth 12 \
# 	--decoder_depth 6 \
# 	--decoder_dim 512 \
# 	--decoder_num_heads 16 \
# 	--checkpoint "./checkpoints/shapenet_pretrain.pkl" \
# 	--num_warmup_steps "2" \
# 	--dim 768 

# python train_cls.py train \
#     --dataroot /nlsasfs/home/neol/rushar/meshmae_2/MeshMAE/dataset_hdf5 \
#     --batch_size 32 \
#     --augment_scale \
#     --n_classes 40 \
#     --channels 10 \
#     --patch_size 64 \
#     --n_epoch 100 \
#     --name "manifoldBase_fine_sn" \
#     --weight_decay 0.05 \
#     --lr 1e-4 \
#     --optim "adamw" \
#     --depth 12 \
#     --heads 12 \
#     --lr_milestones "none" \
#     --encoder_depth 12 \
#     --decoder_depth 6 \
#     --decoder_dim 512 \
#     --decoder_num_heads 16 \
#     --checkpoint "/nlsasfs/home/neol/rushar/meshmae_2/MeshMAE/checkpoints/shapenet_pretrain.pkl" \
#     --num_warmup_steps 2 \
#     --dim 768