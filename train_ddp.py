from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss, Embedding
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Tokenizer, RobertaTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

from utils.adapter import PLMAdapter
from utils.dataset import CSQA2Dataset, OpenbookQADataset, load_resources, pad_token
from utils.util import *


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(rank, args.logname)
        self.logger.info(args)

        _, self.relation2id, _, _ = load_resources()

        self.data_loader = {}
        self.logger.info('Loading dataset...')
        if args.dataset_name == 'csqa2':
            self.load_data(args.csqa2_path, args.dataset_name)
        elif args.dataset_name == 'openbookqa':
            self.load_data(args.openbookqa_path, args.dataset_name)

        if args.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.t5_model_type)
        elif args.model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model_type)

        vocab = self.tokenizer.get_vocab()
        pad = pad_token[args.model_type]
        self.model, parameters = self.create_model(len(vocab))
        plm_embed = self.model.get_input_embeddings()
        self.adapter_embed = Embedding.from_pretrained(plm_embed.weight).to(local_rank)
        self.adapter_embed.requires_grad_(requires_grad=False)
        self.rel_embed = self.model.create_relation_embed(self.relation2id, self.tokenizer, self.adapter_embed)

        self.model.to(local_rank)
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.optimizer, self.scheduler = self.create_optimizer(parameters)

        self.loss_qa = CrossEntropyLoss()

        self.loss_mask = CrossEntropyLoss(ignore_index=vocab[pad])
        self.loss_rela = CrossEntropyLoss()

    def load_data(self, dataset_path, dataset_name):
        for type in ['train', 'dev', 'test']:
            path = os.path.join(dataset_path, '.'.join([type, dataset_name, 'knowledge', 'json']))
            dataloader = self.get_dataloader(path, type, dataset_name)
            self.data_loader[type] = dataloader
            if type == 'train':
                self.t_total = len(dataloader) * self.args.max_epochs // self.args.gradient_acc_step

    def get_dataloader(self, data_path, type, dataset_name):
        if dataset_name == 'csqa2':
            data = CSQA2Dataset(self.args, data_path, type, self.relation2id)
        elif dataset_name == 'openbookqa':
            data = OpenbookQADataset(self.args, data_path, type, self.relation2id)

        if type == 'train':
            sampler = DistributedSampler(data)
        else:
            sampler = SequentialDistributedSampler(data, batch_size=self.args.batch_size)
        dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=False, pin_memory=True,
                                sampler=sampler, num_workers=args.num_workers)

        return dataloader

    def create_model(self, vocab_size):
        init_model = PLMAdapter(self.args, vocab_size)

        if self.args.freeze_plm:
            name_list = ['adapter', 'scorer', 'mask_proj', 'relation_proj']
            if self.args.model_type == 't5' and 'unicorn' in self.args.t5_model_type:
                name_list = ['adapter', 'scorer', 'mask_proj', 'relation_proj', 'bit']
            total_trainable_parameters, adapter_trainable_parameters = [], []

            optimizer_grouped_parameters = [{'params': [], 'weight_decay': self.args.l2}]
            for n, p in init_model.named_parameters():
                if any(nd in n for nd in name_list):
                    p.requires_grad = True
                    optimizer_grouped_parameters[0]['params'].append(p)
                    total_trainable_parameters.append(p.nelement())
                    if 'adapter' in n or 'scorer' in n:
                        adapter_trainable_parameters.append(p.nelement())
                else:
                    p.requires_grad = False
            total_trainable_parameters = sum(total_trainable_parameters)
            adapter_trainable_parameters = sum(adapter_trainable_parameters)
            total_parameters = sum([param.nelement() for param in init_model.parameters()])
            self.logger.info(
                'total parameters: %.3fM, trainable parameters: %.3fM, trainable adapter parameters: %.3fM' % (
                    total_parameters / 1e6, total_trainable_parameters / 1e6, adapter_trainable_parameters / 1e6))
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in init_model.named_parameters()],
                 'weight_decay': self.args.l2}
            ]
            total_parameters = sum([param.nelement() for param in init_model.parameters()])
            self.logger.info('total parameters: %.3fM, trainable parameters: %.3fM' % (
                total_parameters / 1e6, total_parameters / 1e6))

        return init_model, optimizer_grouped_parameters

    def create_optimizer(self, parameters):
        optimizer = AdamW(parameters, lr=self.args.lr)
        scheduler = self.make_scheduler(optimizer)

        return (optimizer, scheduler)

    def make_scheduler(self, optimizer):
        if self.args.warmup_proportion == -1:
            return get_constant_schedule(optimizer)
        else:
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_proportion * self.t_total,
                num_training_steps=self.t_total,
                num_cycles=self.args.scheduler_num_cycles)

    def load_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.model.module.load_state_dict(state_dict, strict=False)

    def save_model(self, save_path, epoch, loss, is_best):
        if rank == 0:
            state = {'epoch': epoch,
                     'loss': loss,
                     'state_dict': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'args': vars(self.args)}
            if is_best:
                torch.save(state, os.path.join(save_path, 'BEST_checkpoint.tar'))

    def qa_loss_acc(self, logits, answer, cands_list):
        label = torch.tensor([cands_list[i].split('#').index(answer[i]) for i in range(len(answer))],
                             device=local_rank).unsqueeze(1)

        loss = self.loss_qa(logits, label)
        argmax_logit = torch.argmax(logits, dim=1)
        accuracy = (argmax_logit == label).sum() / label.size(0)

        return accuracy * 100, loss

    def mask_loss(self, mask_adapter_out, knowledge_label):
        '''
        Args:
            mask_adapter_out: [batch_size, max_len, vocab_size]
            knowledge_label: [batch_size, max_len]
        '''

        m_loss = self.loss_mask(mask_adapter_out.permute(0, 2, 1), knowledge_label)

        return m_loss

    def relation_contrastive_loss(self, result_for_contrastive):
        query, positive_key = [], []
        for relation_tensor, rel_id in result_for_contrastive:
            query.append(relation_tensor)
            positive_key.append(self.rel_embed[rel_id])
        if len(query) == 0:
            loss = 0
        else:
            loss = infonce(loss_func=self.loss_rela,
                           query=normalize(torch.stack(query)),
                           positive_key=normalize(torch.stack(positive_key)),
                           negative_keys=normalize(self.rel_embed),
                           temperature=args.temperature)

        return loss

    def run_epoch(self, epoch, dataloader):
        self.model.train()

        losses, qa_losses, mask_losses, constra_losses = [], [], [], []
        accuracy = []

        for step, batch in enumerate(dataloader):
            token_id, attention_mask, knowledge_id, knowledge_att_mask, knowledge_label, head_tail_index, rel_ids, sign, answer, cands_list = batch
            token_id, attention_mask, = token_id.squeeze(2), attention_mask.squeeze(2)
            plm_logits_outs, mask_adapter_out, result_for_contrastive, id_k = self.model(token_id.to(local_rank),
                                                                                         attention_mask.to(local_rank),
                                                                                         knowledge_id.to(local_rank),
                                                                                         knowledge_att_mask.to(
                                                                                             local_rank),
                                                                                         sign,
                                                                                         head_tail_index.to(local_rank),
                                                                                         rel_ids,
                                                                                         self.adapter_embed)

            qa_acc, qa_loss = self.qa_loss_acc(plm_logits_outs, answer, cands_list)

            if self.args.add_adapter:
                ent_mask_loss = self.mask_loss(mask_adapter_out, knowledge_label[:, id_k, :].to(local_rank))
                rel_contrastive_loss = self.relation_contrastive_loss(result_for_contrastive)

            else:
                ent_mask_loss, rel_contrastive_loss = 0, 0

            loss = qa_loss + ent_mask_loss + rel_contrastive_loss

            accuracy.append(qa_acc.item())
            losses.append(loss.item())
            qa_losses.append(qa_loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.args.add_adapter:
                mask_losses.append(ent_mask_loss.item())
                constra_losses.append(
                    rel_contrastive_loss.item() if rel_contrastive_loss != 0 else rel_contrastive_loss)
            else:
                mask_losses.append(0)
                constra_losses.append(0)

            if step % args.print_freq == 0:
                tmp_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('step_num/train_loss', np.mean(losses), step + epoch * len(dataloader))
                writer.add_scalar('step_num/learning_rate', tmp_lr, step + epoch * len(dataloader))
                self.logger.info(
                    'Epoch: [{0}][{1}/{2}]\t lr: {3:.9f}\t'
                    'Total Loss: {4:.4f}\t QA Loss: {5:.4f}\t Mask Loss: {6:.4f}\t Contrastive Loss: {7:.4f}\t'
                    'Train Acc: {8:.4f}\n'.format(epoch, step, len(dataloader), tmp_lr, np.mean(losses),
                                                  np.mean(qa_losses),
                                                  np.mean(mask_losses), np.mean(constra_losses), np.mean(accuracy)))

        loss, acc = np.mean(losses), np.mean(accuracy)

        return loss, acc

    def evaluate(self, dataloader):
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                token_id, attention_mask, knowledge_id, knowledge_att_mask, knowledge_label, head_tail_index, rel_ids, sign, answer, cands_list = batch
                token_id, attention_mask, = token_id.squeeze(2), attention_mask.squeeze(2)
                plm_logits_outs, _, _, _ = self.model(token_id.to(local_rank), attention_mask.to(local_rank),
                                                      knowledge_id.to(local_rank), knowledge_att_mask.to(local_rank),
                                                      sign, head_tail_index.to(local_rank), rel_ids, self.adapter_embed)

                label = [cands_list[i].split('#').index(answer[i]) for i in range(len(answer))]
                pred = torch.argmax(plm_logits_outs, dim=1).squeeze().tolist()

                if type(pred) == int:
                    pred = [pred]

                labels.extend(label)
                preds.extend(pred)

            preds = distributed_concat(torch.tensor(preds).to(local_rank),
                                       len(dataloader.sampler.dataset))
            labels = distributed_concat(torch.tensor(labels).to(local_rank),
                                        len(dataloader.sampler.dataset))

        acc = (np.array(preds.cpu()) == np.array(labels.cpu())).sum() / len(labels) * 100

        return acc

    def fit(self):
        best_acc = 0

        if args.restore:
            self.load_model(os.path.join(args.save_path, 'roberta-large.tar'))

        for epoch in range(self.args.max_epochs):
            train_loss, train_acc = self.run_epoch(epoch, self.data_loader['train'])
            dev_acc = self.evaluate(self.data_loader['dev'])
            if args.dataset_name == 'openbookqa':
                test_acc = self.evaluate(self.data_loader['test'])
            else:
                test_acc = 0

            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            writer.add_scalar('epoch/valid_acc', dev_acc, epoch)

            self.logger.info(
                '\nEpoch: [{0}]\t Train Loss:{1:.4f}\t Train Acc: {2:.4f}\t '
                'Dev Acc: {3:.4f}\t Test Acc: {4:.4f}\n'.format(
                    epoch, train_loss, train_acc, dev_acc, test_acc))

            if dev_acc > best_acc:
                best_acc = dev_acc
                is_best = True
            else:
                is_best = False

            self.save_model(args.save_path, epoch, train_loss, is_best)


if __name__ == '__main__':
    args = parse_args()
    local_rank = set_device(args)
    writer = SummaryWriter(os.path.join('./runs', args.logdir))

    rank = torch.distributed.get_rank()
    set_seed(2 + rank)

    model = Runner(args)
    model.fit()
