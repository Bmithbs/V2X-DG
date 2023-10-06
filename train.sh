export PYTHONPATH=$PYTHONPATH:/home/xhchen/hsk/CoBEVT/opv2v/opencood
export CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch --nproc_per_node=2  --use_env opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml --if_meta_training --mmd
nohup python  opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml --model_dir logs_pretrained/cobevt_2 --if_meta_training --mmd > output_3.txt &


# nohup python opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml --model_dir opencood/logs/corpbevt_2023_08_22_23_03_39 > output_cobevt.txt &

# nohup python opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt_static.yaml --model_dir opencood/logs/corpbevt_static_2023_08_22_23_00_41 > output_cobevt_static.txt &  

# nohup python opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/cvt_v2vnet.yaml --model_dir opencood/logs/cross_view_transformer_v2vnet_2023_08_19_21_46_12 > output_v2vnet.txt &  

# nohup python opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/cvt_v2vnet_static.yaml --model_dir /home/xhchen/workspace/haonan/CoBEVT/opv2v/opencood/logs/cross_view_transformer_v2vnet_static_2023_08_20_15_33_54 > output_v2vnet_static.txt &  
