export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0

if true; then
CKPT_PATH=data/ckpt/hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt # hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
BACKBONE=hr48
else
CKPT_PATH=data/ckpt/res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
BACKBONE=res50
fi

if false; then
python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
               --input_path data/test_samples/crowdpose_100024.jpg \
               --input_type image \
               --show_bbox --show_sideView --save_results
fi

if false; then
python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
               --input_path data/test_samples/downtown_runForBus_00 --input_type video \
               --show_bbox --show_sideView --save_results --make_video --frame_rate 30
fi

if true; then
python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
               --input_path /media/odin/stdrr/data/AGORA/images/validation_images_1280x720 --input_type folder \
               --show_bbox --show_sideView --save_results --dataset agora --record_time --compute_volume_metrics vol_parser --pymesh
fi
