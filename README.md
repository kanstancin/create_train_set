# create_train_set
### how to run
```bash
bash configure_dirs.sh
python create_dataset.py
```

### how to train YOLO
```bash
python train.py --img 640 --batch 16 --epochs 3 \
--data coco128.yaml --weights yolov5s.pt
```