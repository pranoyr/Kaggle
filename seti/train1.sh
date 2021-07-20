rm -rf tf_logs1/*
python train1.py &
tensorboard dev upload --logdir ./tf_logs1