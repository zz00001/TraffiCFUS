import time
t = time.time()
time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
output_file = './result/' + time_string + '/'

def parse_arguments(parser):
    parser.add_argument('--alpha', type=float, default=0.2, help='0.2')
    parser.add_argument('--beta', type=float, default=0.2, help='0.2')
    parser.add_argument('--gamma', type=float, default=0.025, help='0.025')
    parser.add_argument('--batch_size', type=int, default=64, help='64')
    parser.add_argument('--class_num', type=int, default=2, help='2')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='Learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=5, help='Learning rate decay step')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate:0.15')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--d_text', type=int, default=32, help='Text dimension weibo: 32')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--k', type=int, default=5, help='K-fold')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.0, help='')
    parser.add_argument('--num_class', type=int, default=2, help='')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epoch')
    parser.add_argument('--num_filter', type=int, default=2, help='Number of filter bank')
    parser.add_argument('--num_layer', type=int, default=1, help='Number of block layer')
    parser.add_argument('--patch_size', type=int, default=16, help='Image patch size')
    parser.add_argument('--patience', type=int, default=10, help='How long to wait after last time valid loss improved')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seq_len', type=int, default=80, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset')
    parser.add_argument('--use_parallel', type=bool, default=False, help='')
    parser.add_argument('--vocab_size', type=int, help='')

    parser.add_argument('--data_path', type=str, default='./trafficnet_dataset_v1/', help='')
    parser.add_argument('--dataset_type', type=str, default='as', help='as: accident and sparse_traffic')
    parser.add_argument('--output_path', type=str, default=output_file, help='')
    parser.add_argument('--remarks', type=str, default='TraffiCFUS', help='remark')
    parser.add_argument('--need_tsne_data', type=bool, default=False, help='need t-sne data or not')

    return parser
