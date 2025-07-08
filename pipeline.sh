# Step1: process the video, extract frames and save them in a folder || [NOTICE] Change the data-dir to your own path
DATA_DIR=/home/jiahao/Downloads/data/wildtrack_data_gt
# python s01_data_download.py --data-dir ${DATA_DIR} --duration 35 --fps 2 --output-folder Image_subsets


# # Step2: segment the foreground and background using grounded-sam
# mkdir submodules && cd submodules
# git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
# cd Grounded-Segment-Anything
# python -m pip install -e segment_anything
# pip install --no-build-isolation -e GroundingDINO
# pip install --upgrade "diffusers[torch]"
# git submodule update --init --recursive
# cd grounded-sam-osx && bash install.sh
# git clone https://github.com/xinyu1205/recognize-anything.git
# pip install -r ./recognize-anything/requirements.txt
# pip install -e ./recognize-anything/

# # download the pre-trained model
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# # Download the HQ-SAM from https://github.com/SysCV/sam-hq#model-checkpoints [optional] 
# wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8&export=download&authuser=0&confirm=t&uuid=bc8e6a82-132e-4582-8ec5-887647acd490&at=AO7h07ej-Yws8bBHfUlF2N8zsISP:1724719583150' -O sam_hq_vit_h.pth

# generate the foreground (human) mask
# TARGET1="people"
# python s02_sam2_wgt.py \
#         --root ${DATA_DIR} \
#         --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#         --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
#         --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
#         --input_image None \
#         --output_dir ${DATA_DIR}/masks/${TARGET1} \
#         --box_threshold 0.3 \
#         --text_threshold 0.25 \
#         --text_prompt ${TARGET1} \
#         --device "cuda:0"
# python s02_sam2.py \
#         --root ${DATA_DIR} \
#         --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#         --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
#         --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
#         --input_image None \
#         --output_dir ${DATA_DIR}/masks/${TARGET1} \
#         --box_threshold 0.3 \
#         --text_threshold 0.25 \
#         --text_prompt ${TARGET1} \
#         --device "cuda:0"

# generate the background (ground) mask
# TARGET2="ground"
# python s02_sam2.py \
#         --root ${DATA_DIR} \
#         --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#         --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
#         --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
#         --input_image None \
#         --output_dir ${DATA_DIR}/masks/${TARGET2} \
#         --box_threshold 0.3 \
#         --text_threshold 0.25 \
#         --text_prompt ${TARGET2} \
#         --device "cuda:0"

# Step3: Initialization for GS. Generate super-pixel for the foreground
# python s03_super_pixel.py \
#     --root ${DATA_DIR} \
#     --n_segments "30, 60"

# --- Iterative Matching Label --- #
# --- First Rround --- #

# Step4: per frame training for GS
# python s04_gs_us.py --root ${DATA_DIR} --round 1_1 --n-segments 30 --start-with 1 --end-with -1

# Step5: fine-tuning mono-depth estimation
# python s05_depth_finetune.py \
#         --epochs 200 \
#         --encoder vits \
#         --bs 1 \
#         --lr 0.000005 \
#         --save-path "output/1_1_w_dc_200" \
#         --dataset wildtrack \
#         --img-size 518 \
#         --min-depth 0.001 \
#         --max-depth 40 \
#         --pretrained-from checkpoints/depth_anything_v2_vits.pth \
#         # --pretrained-from output/1_1_w_dc/latest.pth \
#         --port 20596 \
#         --data_root ${DATA_DIR} \
#         --round 1_1

# --- Second Rround --- #
# Step6: second round intit -> supplement the missing region for the foreground using median depth 
# python s06_depth_based_gs_init.py \
#         --root ${DATA_DIR} \
#         --round 1_1 \
#         --human_depth_intvl 40.0 \
#         --depth_supp_num 200 \

# Step4: per frame training for GS
# python s04_gs_us.py --root ${DATA_DIR} --round 2_1 --n-segments 30 --start-with 1 --end-with -1
# python s04_gs_depth_prior.py \
#             --root ${DATA_DIR} \
#             --round '2_1' \
#             --n-segments '30' \
#             --start-with '0' \
#             --end-with '-1' \
#             --loss_bg_w '1.0' \
#             --loss_rgb_w '1.0' \
#             --loss_local_depth_w '0' \
#             --loss_global_depth_w '1e-5' \
#             --loss_rank_depth_w '0' \
#             --init_opacity '0.1' \
#             --loss_sparsity_w '0' \
#             --loss_quantization_w '0' \
#             --epochs '2000' \
#             --sparsity_epoch '4000' \
#             --pruning_epoch '4000' \
#             --bg_filter 'True' \
#             --depth_filter 'False' \
#             --depth_constraint 'cluster' \
#             --random_select '5'


# Step5: fine-tuning mono-depth estimation
# python s05_depth_finetune.py \
#         --epochs 200 \
#         --encoder vits \
#         --bs 1 \
#         --lr 0.000005 \
#         --save-path "output/depth_ft_2_2" \
#         --dataset wildtrack \
#         --img-size 518 \
#         --min-depth 0.001 \
#         --max-depth 40 \
#         --pretrained-from output/depth_ft_1_2/latest.pth \
#         # --pretrained-from output/1_1_w_dc/latest.pth \
#         --port 20596 \
#         --data_root ${DATA_DIR} \
#         --round 2_1

# --- Third Rround --- #
# Step6: third round intit -> supplement the missing region for the foreground using median depth 
# python s06_depth_based_gs_init.py \
#         --root ${DATA_DIR} \
#         --round 2_1 \
#         --human_depth_intvl 40.0 \
#         --depth_supp_num 200 \

# --- Cluster (unsupervised)--- #
# python unspervised_cluster.py \
#         --root ${DATA_DIR}\
#         --round 2_1 \
#         --n-segments 30 \
#         --start-with 0 \
#         --end-with -2 \
#         --min_gs_threshold 10

# --- Cluster (supervised)--- #
python supervised_cluster.py 
        