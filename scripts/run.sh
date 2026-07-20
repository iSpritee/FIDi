TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 scripts/dist_train_voc_seg_neg.py \
--work_dir work_dir_voc --exp wo_Dr


TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 scripts/dist_train_coco_seg_neg.py \
--work_dir work_dir_coco --exp test


TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 scripts/dist_test_coco_seg_neg.py \
--work_dir work_dir_coco