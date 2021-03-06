import argparse


def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume_path', type=str,
						help='path to weight file')
	parser.add_argument('--batch_size', type=int, default=4,
						help='batch size')
	parser.add_argument('--lr', type=float,
						help='learning rate')
	parser.add_argument('--weight_decay', type=float, default = 0,
						help='weight decay')
	parser.add_argument('--num_workers', type=int, default=0,
						help='number of workers for data loaders')
	parser.add_argument('--epochs', type=int, default=100,
						help='number of Epochs')
	parser.add_argument('--scheduler', type=str, default="multi_step",
						help='scheduler')
	parser.add_argument('--begin_iter', type=int, default=1,
						help='starting iteration')
	parser.add_argument('--device_id', type=int, default=0,
						help='gpu id')
	# parser.add_argument('--multi_gpu', action='store_true', help='Enables multiple GPU training')
	args = parser.parse_args()

	return args
