from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.args.model_dir = self.args.save_meta_path + '/session0_max_acc.pth'

        if os.path.exists(self.args.model_dir):
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        for k, v in self.model.named_parameters():
            if 'encoder' in k or 'mlp_1' in k or 'mlp_2' in k or 'instance_gcn' in k or 'proto_gcn' in k:
                v.requires_grad = False

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr_pretrain, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_new)

        return optimizer, scheduler

    def get_dataloader(self, session, base_meta = False):
        if session == 0:
            if base_meta :
                trainset, trainloader, testloader = get_meta_dataloader(self.args)
            else:
                trainset, trainloader, testloader = get_pretrain_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        self.best_model_dict = torch.load(self.args.model_dir)['params']
        self.graph_node = torch.load(self.args.model_dir)['buffer']

        state_dict = self.best_model_dict
        new_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           ((k in new_state_dict) and (v.shape == new_state_dict[k].shape))}
        new_state_dict.update(pretrained_dict)
        self.model.load_state_dict(new_state_dict)
        self.model.module.graph_node = self.graph_node

        args.start_session = 1
        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)
            print('new classes for this session:\n', np.unique(train_set.targets))
            optimizer, scheduler = self.get_optimizer_base()
            
            tqdm_gen = tqdm(trainloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                self.model(data, label, session)

            # test model with all seen class
            tsl, tsa = test_novel(self.model, testloader, 0, args, session)

            print("float('%.3f' % (tsa * 100))",float('%.3f' % (tsa * 100)))

            # save better model
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))

            save_model_dir = os.path.join(self.args.save_novel_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('Saving model to :%s' % save_model_dir)
            print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

            result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

            result_list.append(self.trlog['max_acc'])
            print(self.trlog['max_acc'])

            t_end_time = time.time()
            total_time = (t_end_time - t_start_time) / 60
            result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
            print('Best epoch:', self.trlog['max_acc_epoch'])
            print('Total time used %.2f mins' % total_time)
            save_list_to_txt(os.path.join(self.args.save_novel_path, 'results.txt'), result_list)


    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_pretrain_path = self.args.save_path + 'pretrain/'
        self.args.save_meta_path = self.args.save_path + 'meta/'
        self.args.save_novel_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_pretrain_path = os.path.join('checkpoint', self.args.save_pretrain_path)
        self.args.save_meta_path = os.path.join('checkpoint', self.args.save_meta_path)
        self.args.save_novel_path = os.path.join('checkpoint', self.args.save_novel_path)
        ensure_path(self.args.save_novel_path)
        return None

    def test(self):
        args = self.args

        self.args.model_dir = self.args.save_novel_path + '/session8_max_acc.pth'
        self.best_model_dict = torch.load(self.args.model_dir)['params']
        self.graph_node = torch.load(self.args.model_dir)['buffer']

        state_dict = self.best_model_dict
        new_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           ((k in new_state_dict) and (v.shape == new_state_dict[k].shape))}
        new_state_dict.update(pretrained_dict)
        self.model.load_state_dict(new_state_dict)
        self.model.module.graph_node = self.graph_node

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session, True)
            tsl, tsa = test_meta(self.model, testloader, 0, args, session)
            print("session",session, tsa)
