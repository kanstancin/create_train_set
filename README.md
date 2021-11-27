# create_train_set
### how to run
```bash
bash configure_dirs.sh
python create_dataset.py
```

### how to train YOLO
```bash
python train.py --img 640 --batch 16 --epochs 30 --data ../create_train_set/yaml/dataset.yaml --weights yolov5s.pt
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