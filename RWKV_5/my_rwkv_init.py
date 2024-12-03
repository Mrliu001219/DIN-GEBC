from argparse import ArgumentParser
import os

class my_parser:
    def __init__(self):
        self.head_size_a = 256
        self.dim_att = 0
        self.head_size_divisor = 8
        self.n_layer =  2
        self.n_embd = 768
        self.dim_ffn = 0
        self.my_pos_emb = 0
        self.pre_ffn =  0
        self.tiny_att_dim =  0
        self.tiny_att_layer = 0
        self.ctx_len = 12
        self.dropout = 0.01
        self.vocab_size = 4096
        self.head_qk =  0
        self.layerwise_lr = 1
        self.my_pile_stage = 1
        self.weight_decay = 0.3
        self.lr_init = 4e-3
        self.beta1 =  0.9
        self.beta2 =  0.99
        self.adam_eps = 1e-8
        self.grad_cp = 0
        self.my_qa_mask = 0
        self.accelerator = "gpu"
        self.my_testing = ''
        self.precision = "fp16"


def rwkv_init():

    args = my_parser()

    args.betas = (args.beta1, args.beta2)

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_JIT_ON"] = "0"

    from RWKV_5.src.model import RWKV
    model = RWKV(args)

    return model