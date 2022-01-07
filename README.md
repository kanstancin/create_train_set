# create_train_set
### how to run
```bash
bash configure_dirs.sh
python create_dataset.py
```

### download backgound imgs
```bash
# https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations
aws s3 --no-sign-request sync s3://open-images-dataset/test
```

### how to train YOLO
```bash
python train.py --img 640 --batch 16 --epochs 30 --data ../create_train_set/yaml/dataset.yaml --weights yolov5s.pt

python train.py --img 608 --batch 64 --epochs 70 --data ../create_train_set/yaml/dataset.yaml --weights yolov5s.pt --freeze 17 --hyp ../create_train_set/yaml/hyp.custom_v1.yaml
```

### how to predict
```bash
python detect.py --weights 'runs/train/exp9/weights/last.pt' \
--img 600 \
--conf 0.15 \
--iou 0.5 \
--source ../create_train_set/data/spaghetti_proc/ \
--exist-ok
```
