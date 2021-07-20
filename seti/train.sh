rm -rf tf_logs/*
python train.py &
tensorboard dev upload --logdir ./tf_logs