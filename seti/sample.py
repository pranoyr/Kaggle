

from train_old_new_scratch import *

model = ResidualNet("ImageNet", 50, 1000, "CBAM")
model = nn.DataParallel(model)

# if torch.cuda.device_count() > 1:
# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
# model = nn.DataParallel(model)

# if resume_path:
#     checkpoint = torch.load(resume_path)
#     model.load_state_dict(checkpoint['state_dict'])
#     # epoch = checkpoint['epoch']
#     print("Model Restored")
    # start_epoch = epoch + 1

# model.fc2 = nn.Linear(2048, 1)
# model.to('device')


print(1.0e-3)
