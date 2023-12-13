from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.args.model_dir = self.args.save_pretrain_path + '/session0_max_acc.pth'
        # self.args.model_dir = self.args.save_meta_path + '/session0_max_acc.pth'

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
            # if 'encoder' in k :
            if 'encoder' in k or 'fc' in k:
            # if 'encoder' in k or 'mlp_1' in k or 'instance_gcn' in k or 'gcn_feature' in k or 'fc' in k:
                v.requires_grad = False

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr_pretrain, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_pretrain)

        return optimizer, scheduler

    def get_dataloader(self, session, base_meta = False):
        if session == 0:
            if base_meta :
                # trainset, trainloader, testloader = get_meta_dataloader(self.args)
                trainset, trainloader, testloader = get_pretrain_dataloader(self.args)
            else:
                trainset, trainloader, testloader = get_pretrain_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]

        masknum=3
        mask=np.zeros((args.pretrain_class,args.num_classes))
        for i in range(args.num_classes-args.pretrain_class):
            picked_dummy=np.random.choice(args.pretrain_class,masknum,replace=False)
            mask[:,i+args.pretrain_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()

        state_dict = self.best_model_dict
        new_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           ((k in new_state_dict) and (v.shape == new_state_dict[k].shape))}
        new_state_dict.update(pretrained_dict)
        self.model.load_state_dict(new_state_dict)

        for session in range(args.start_session, args.sessions):

            # self.model.load_state_dict(self.best_model_dict)
            train_set, trainloader, testloader = self.get_dataloader(session, True)
            if session == 0:

                ##########################################################################################
                # base_meta stage
                # train_set, trainloader, testloader = self.get_dataloader(session,True)
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_meta):
                # for epoch in range(99,args.epochs_meta):
                    start_time = time.time()
                    # train base sess
                    tl, ta = meta_train(self.model, trainloader, optimizer, scheduler, epoch, args, session,mask)
                    # test model with all seen class
                    tsl, tsa = test_meta(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_meta_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_meta_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_meta - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_meta_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test_meta(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.dummy_classifiers = deepcopy(self.model.module.fc.weight.detach())
                self.dummy_classifiers = F.normalize(self.dummy_classifiers[self.args.meta_class:, :], p=2, dim=-1)
            else:
                self.model.module.mode = self.args.new_mode
                print("training session: [%d]" % session)
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa = self.test_intergrate(self.model, testloader, 0, args, session, validation=True)

                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_meta_path, 'session' + str(session) + '_max_acc.pth')
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))


        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_meta_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_pretrain_path = self.args.save_path + 'pretrain/'
        self.args.save_meta_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_pretrain_path = os.path.join('checkpoint', self.args.save_pretrain_path)
        self.args.save_meta_path = os.path.join('checkpoint', self.args.save_meta_path)
        ensure_path(self.args.save_meta_path)
        return None

    def test_intergrate(self, model, testloader, epoch, args, session, validation=True):
        test_class = args.meta_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5 = Averager()
        lgt = torch.tensor([])
        lbs = torch.tensor([])

        proj_matrix = torch.mm(self.dummy_classifiers,
                               F.normalize(torch.transpose(model.module.fc.weight[:test_class, :], 1, 0), p=2, dim=-1))
        print("proj_matrix",proj_matrix.shape)
        eta = args.eta

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                emb = model.module.get_feature_test(data,test_class)
                proj = torch.mm(F.normalize(emb, p=2, dim=-1), torch.transpose(self.dummy_classifiers, 1, 0))
                topk, indices = torch.topk(proj, 40)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1 = torch.mm(res_logit, proj_matrix)
                logits2 = model.module.predict(data,test_class)[:, :test_class]
                logits = eta * F.softmax(logits1, dim=1) + (1 - eta) * F.softmax(logits2, dim=1)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc = count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)
                lgt = torch.cat([lgt, logits.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])

            vl = vl.item()
            va = va.item()
            va5 = va5.item()
            print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        return vl, va


    def test(self):
        args = self.args

        self.args.model_dir = self.args.save_meta_path + '/session0_max_acc.pth'
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

    def test(self):
        args = self.args

        self.args.model_dir = self.args.save_meta_path + '/session0_max_acc.pth'
        print("self.args.model_dir",self.args.model_dir)
        self.best_model_dict = torch.load(self.args.model_dir)['params']

        state_dict = self.best_model_dict
        new_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           ((k in new_state_dict) and (v.shape == new_state_dict[k].shape))}
        new_state_dict.update(pretrained_dict)
        self.model.load_state_dict(new_state_dict)

        train_set, trainloader, testloader = self.get_dataloader(0)
        if not args.not_data_init:
            self.model.load_state_dict(self.best_model_dict)
            self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
            best_model_dir = os.path.join(args.save_meta_path, 'session' + str(0) + '_max_acc.pth')
            print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            torch.save(dict(params=self.model.state_dict()), best_model_dir)

            self.model.module.mode = 'avg_cos'
            tsl, tsa = test_meta(self.model, testloader, 0, args, 0)

            print('The new best test acc of base session={:.3f}'.format(float('%.3f' % (tsa * 100))))

        self.dummy_classifiers = deepcopy(self.model.module.fc.weight.detach())
        self.dummy_classifiers = F.normalize(self.dummy_classifiers[self.args.pretrain_class:, :], p=2, dim=-1)

        pre = []
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            if session > 0:
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                # tsl, tsa = test_meta(self.model, testloader, 0, args, session)
                tsl, tsa = self.test_intergrate(self.model, testloader, 0, args, session, validation=True)
            else:
                tsl, tsa = test_meta(self.model, testloader, 0, args, session)
            tsa = float('%.3f' % (tsa * 100))
            pre.append(tsa)
            print("session",session, tsa)
        print(pre)