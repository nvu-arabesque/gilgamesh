"""
@credit: 
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
"""

import torch
import torch.nn.functional as F
import tqdm
import math


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """ Apply label smoothing if needed """

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction="sum")
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    """ Epoch operation in training phase"""

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Training)   "
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(
            lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx)
        )

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing
        )
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    """ Epoch operation in evaluation phase """

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Validation) "
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(
                lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx)
            )

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False
            )

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    """ Start training """

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        print(
            "[Info] Training performance will be written to file: {} and {}".format(
                log_train_file, log_valid_file
            )
        )

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy\n")

    def print_performances(header, loss, accu, start_time):
        print(
            "  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, "
            "elapse: {elapse:3.3f} min".format(
                header=f"({header})",
                ppl=math.exp(min(loss, 100)),
                accu=100 * accu,
                elapse=(time.time() - start_time) / 60,
            )
        )

    # valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print("[ Epoch", epoch_i, "]")

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing
        )
        print_performances("Training", train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        print_performances("Validation", valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {"epoch": epoch_i, "settings": opt, "model": model.state_dict()}

        if opt.save_model:
            if opt.save_mode == "all":
                model_name = opt.save_model + "_accu_{accu:3.3f}.chkpt".format(
                    accu=100 * valid_accu
                )
                torch.save(checkpoint, model_name)
            elif opt.save_mode == "best":
                model_name = opt.save_model + ".chkpt"
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print("    - [Info] The checkpoint file has been updated.")

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as log_tf, open(
                log_valid_file, "a"
            ) as log_vf:
                log_tf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)),
                        accu=100 * train_accu,
                    )
                )
                log_vf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)),
                        accu=100 * valid_accu,
                    )
                )
