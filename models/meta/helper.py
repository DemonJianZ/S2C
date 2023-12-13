from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F

def meta_train(model, trainloader, optimizer, scheduler, epoch, args, session,mask):
    meta_class = args.meta_class + session * args.way
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.cuda() for _ in batch]
        # support_size = args.episode_way * args.episode_shot
        # support_data = data[:support_size]
        # support_label = train_label[:support_size]
        # query_data = data[support_size:]
        # query_label = train_label[support_size:]

        logits = model(data, train_label)
        prom_loss = model.module.prom_loss()

        logits_ = logits[:, :meta_class]
        loss = F.cross_entropy(logits_, train_label)
        loss =loss

        acc = count_acc(logits_, train_label)

        logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
        logits_masked_chosen = logits_masked * mask[train_label]
        pseudo_label = torch.argmax(logits_masked_chosen[:, args.pretrain_class:], dim=-1) + args.pretrain_class
        loss2 = F.cross_entropy(logits_masked, pseudo_label)

        index = torch.randperm(data.size(0)).cuda()
        pre_emb1 = model.module.pre_encode(data)
        mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
        mixed_logits = model.module.post_encode(data,mixed_data)

        newys = train_label[index]
        idx_chosen = newys != train_label
        mixed_logits = mixed_logits[idx_chosen]

        pseudo_label1 = torch.argmax(mixed_logits[:, args.pretrain_class:],
                                     dim=-1) + args.pretrain_class  # new class label
        pseudo_label2 = torch.argmax(mixed_logits[:, :args.pretrain_class], dim=-1)  # old class label
        loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
        novel_logits_masked = mixed_logits.masked_fill(
            F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
        loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
        total_loss = torch.mean(loss) + args.balance * (loss2 + loss3 + loss4)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()

    return tl, ta

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            embedding = model.module.get_feature_test(data,args.meta_class)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.meta_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.meta_class] = proto_list

    return model


def test_meta(model, testloader, epoch, args, session):
    test_class = args.meta_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model.module.predict(data,test_class)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

