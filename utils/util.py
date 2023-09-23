import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-csqa2_answer_list', default=['yes', 'no'], type=list,
                        help='label list of csqa2 dataset')

    # model
    parser.add_argument('-t5_model_type', default='./pretrained_model/t5-large', type=str,
                        help='model type or path of t5')

    parser.add_argument('-roberta_model_type', default='./pretrained_model/roberta-large', type=str,
                        help='model type or path of roberta')
    parser.add_argument('-model_type', default='roberta', type=str, choices=['t5', 'roberta'])
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-hidden_size', default=1024, type=int,
                        help='embedding dimension of encoder output, 768 for t5-base, 1024 for DeBERTaV3')
    parser.add_argument('-temperature', default=0.1, type=float, help='hyperparameter in contrastive learning')
    parser.add_argument('-down_size', default=256, type=int, help='down size in adapter')
    parser.add_argument('-nhead', default=8, type=int, help='multi-head in AttentionAdapter')
    parser.add_argument('-num_encoder_layer', default=24, type=int, help='number of encoder layer in plm')

    # train
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('-dataset_name', default='csqa2', type=str, choices=['csqa2', 'openbookqa'])
    parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('-deepspeed', action="store_true", help='whether to use deepspeed')
    parser.add_argument('-batch_size', default=16, type=int)

    parser.add_argument('-logname', type=str, default='main_log.log',
                        help='the directory where the training log output by logging is stored')
    parser.add_argument('-logdir', type=str, default='main_log',
                        help='the directory where the training log output by tensorboardX is stored')
    parser.add_argument('-gradient_acc_step', type=int, default=1, help='gradient accumulation step.')
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-l2', default=0.0, type=float, help='weight decay in optimizer')
    parser.add_argument('-num_knowledge', type=int, default=5)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-scheduler_num_cycles', type=int, default=1)
    parser.add_argument('-warmup_proportion', type=float, default=0.1)
    parser.add_argument('-restore', dest='restore', action='store_true', help='restore from the previously saved model')
    parser.add_argument('-max_epochs', default=300, type=int, help='number of training epoch')
    parser.add_argument('-print_freq', default=80, type=int, help='print frequence')
    parser.add_argument('-num_workers', default=12, type=int, help='num workers in dataloader')
    parser.add_argument('-save_path', default='./checkpoints', type=str)

    parser.add_argument('-csqa2_path', default='./data/csqa2', type=str)
    parser.add_argument('-openbookqa_path', default='./data/openbookqa', type=str)
    parser.add_argument('-add_adapter', action="store_true", help='add adapters to plm')
    parser.add_argument('-freeze_plm', action="store_true", help='freeze the parameters of pretrained model')
    parser.add_argument('-weight_init_option', default='bert', type=str, help='adapter weight initialization method')
    parser.add_argument('-adapter_scalar', default='1', type=str, help='scalar factor in adapter')
    parser.add_argument('-adapter_layernorm_option', default='out', type=str, choices=['in', 'out', 'both', 'none'],
                        help='position of layernorm in adapter')
    parser.add_argument('-adapter_dropout', default=0.1, type=float, help='dropout rate in adapter')

    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def get_logger(rank, filename):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s: %(message)s")

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    logger.setLevel(logging.INFO if rank in [-1, 0] else logging.ERROR)

    return logger


def set_gpu(gpus):
    '''
    gpus: List of GPUs to be used for the run
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl')
    # deepspeed.init_distributed()

    return local_rank


def cache(func):
    def wrapper(*args, **kwargs):
        file_dir = kwargs['file_dir']
        postfix = kwargs['postfix']
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        data_path = os.path.join(file_dir, f"{postfix}.pt")
        if not os.path.exists(data_path):
            print(f"cache file {data_path} not exist, reprocess and write to cache...")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            print(f"cache file {data_path} exist, loading cache file ...")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


def transpose(x):
    return x.transpose(-2, -1)


def normalize(x):
    return F.normalize(x, dim=-1)


def infonce(loss_func, query, positive_key, negative_keys, temperature=0.01):
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

    negative_logits = query @ transpose(negative_keys)

    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, requires_grad=False).to(query.device)
    loss = loss_func(logits / temperature, labels)

    return loss


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
        Distributed Sampler that subsamples indicies sequentially,
        making it easier to collate all results at the end.
        Even though we only use this sampler for eval and predict (no training),
        which means that the model params won't have to be synced (i.e. will not hang
        for synchronization even if varied number of forward passes), we still add extra
        samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
        to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
        """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
