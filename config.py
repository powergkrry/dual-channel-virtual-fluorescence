import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=16,
                      help='# of images in each batch of data')
data_arg.add_argument('--val_batch_size', type=int, default=8,
                      help='# of images in each batch of data')
data_arg.add_argument('--is_green', type=int, default=0,
                      help='Green or Red channel')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=40,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')
train_arg.add_argument('--final_activation', type=str, default="swish",
                       help='Activation function to use in the final layer')
train_arg.add_argument('--lamda', type=float, default=1e-4,
                       help='Weight of regularization, if used')
train_arg.add_argument('--loss', type=str, default="blur",
                       help='loss function to use. Choose between:\
                            "blur", "mse-r", "mse", "mae"')
train_arg.add_argument('--maxpool', action='store_true',
                       help='whether to use maxpooling over strided conv')
train_arg.add_argument('--polydecay', action='store_true',
                       help='use polynomial learning rate decay')
train_arg.add_argument('--plateaudecay', action='store_true',
                       help='use reduce LR on plateau')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--random_seed', type=int, default=0,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--gpu', type=str, default='1',
                      help='GPU on the server to use')
misc_arg.add_argument('--n_sample', type=int, default=21,
                      help='Number of subsample')
misc_arg.add_argument('--n_out_channels', type=int, default=1,
                      help='Number of fluorescence channel')
misc_arg.add_argument('--name', type=str, default="exp",
                      help='Name of the experiment')


# variance control
# misc_arg.add_argument('--noise', type=float, default=0.0,
#                       help='amount of noise to add to the formed image')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
