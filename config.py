import argparse

def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='设置随机种子')
    parser.add_argument("--epoch", type=int, default=5, help='迭代次数')
    parser.add_argument("--lr", type=float, default=2e-5, help='学习率')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument("--log_dir", type=str, default='./train.log', help='日志的存放位置')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs/', help='tensorboard的存放位置')
    parser.add_argument("--print_num", type=int, default=5, help='tensorboard的存放位置')
    args = parser.parse_args()
    return args
