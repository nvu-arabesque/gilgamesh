"""
adapted from/@credit:
- https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop
- https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/fec78a687210851f055f792d45300d27cc60ae41/train.py
"""

import math
import time
from typing import Callable
import torch
from torch.nn import functional as F
from tqdm import tqdm
from gilgamesh.transformer.transformer import Transformer


class TransformerTrainer:
    def __init__(self):
        self.trainer = None
        self.loss = torch.nn.MSELoss()

    def train_epoch(
        self,
        model: Transformer,
        training_data,
        optimizer,
        opt,
        device,
        smoothing,
    ):
        """ Epoch operation in training phase"""

        model.train()
        total_loss = 0

        desc = "  - (Training)   "
        for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src, tgt, tgt_mask, inp = batch.src, batch.tgt, batch.tgt_mask, batch.inp
            src = src.to(device)
            # forward
            optimizer.zero_grad()
            pred = model.forward(source_seq=src, target_seq=inp, tgt_mask=tgt_mask)
            # backward and update parameters
            loss = self.cal_performance(pred, tgt, smoothing=smoothing)
            loss.backward()
            optimizer.step()
            # note keeping
            total_loss += loss.item()
        return total_loss

    def eval_epoch(self, model, validation_data, device, opt):
        """ Epoch operation in evaluation phase """

        model.eval()
        total_loss = 0
        desc = "  - (Validation) "
        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

                # prepare data
                src, tgt, tgt_mask, inp = (
                    batch.src,
                    batch.tgt,
                    batch.tgt_mask,
                    batch.inp,
                )
                # forward
                pred = model.forward(source_seq=src, target_seq=inp, tgt_mask=tgt_mask)

                # cal loss
                loss = self.cal_performance(pred, tgt, smoothing=False)
                # note keeping
                total_loss += loss.item()

        return total_loss

    def cal_performance(
        self, x: torch.Tensor, y: torch.Tensor, smoothing: bool = False
    ):
        """ Apply label smoothing if needed """
        return self.loss(x.contiguous().view(-1), y.contiguous().view(-1))

    def print_performances(self, header, loss, start_time):
        print(
            "  - {header:12} ppl: {ppl: 8.5f}"
            "elapse: {elapse:3.3f} min".format(
                header=f"({header})",
                ppl=math.exp(min(loss, 100)),
                accu="Omitted",
                elapse=(time.time() - start_time) / 60,
            )
        )

    def train(
        self,
        model,
        training_data,
        validation_data,
        optimizer,
        device,
        epoch=100,
        opt=None,
        smoothing=False,
    ):
        """ Start training """

        log_train_file, log_valid_file = None, None

        # if opt.log:
        #     log_train_file = opt.log + ".train.log"
        #     log_valid_file = opt.log + ".valid.log"

        #     print(
        #         "[Info] Training performance will be written to file: {} and {}".format(
        #             log_train_file, log_valid_file
        #         )
        #     )

        #     with open(log_train_file, "w") as log_tf, open(
        #         log_valid_file, "w"
        #     ) as log_vf:
        #         log_tf.write("epoch,loss,ppl,accuracy\n")
        #         log_vf.write("epoch,loss,ppl,accuracy\n")
        # valid_accus = []
        valid_losses = []
        for epoch_i in range(epoch):
            print("[ Epoch", epoch_i, "]")

            start = time.time()
            train_loss = self.train_epoch(
                model,
                training_data,
                optimizer,
                opt,
                device,
                smoothing=smoothing,
            )
            self.print_performances("Training", train_loss, start)

            start = time.time()
            valid_loss = self.eval_epoch(model, validation_data, device, opt)
            self.print_performances("Validation", valid_loss, start)

            valid_losses += [valid_loss]

            checkpoint = {
                "epoch": epoch_i,
                "settings": opt,
                "model": model.state_dict(),
            }

            # if opt.save_model:
            #     if opt.save_mode == "all":
            #         model_name = opt.save_model + "_accu_{accu:3.3f}.chkpt".format(
            #             accu=100 * valid_accu
            #         )
            #         torch.save(checkpoint, model_name)
            #     elif opt.save_mode == "best":
            #         model_name = opt.save_model + ".chkpt"
            #         if valid_loss <= min(valid_losses):
            #             torch.save(checkpoint, model_name)
            #             print("    - [Info] The checkpoint file has been updated.")

            if log_train_file and log_valid_file:
                with open(log_train_file, "a") as log_tf, open(
                    log_valid_file, "a"
                ) as log_vf:
                    log_tf.write(
                        "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                            epoch=epoch_i,
                            loss=train_loss,
                            ppl=math.exp(min(train_loss, 100)),
                            accu="Omitted",
                        )
                    )
                    log_vf.write(
                        "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                            epoch=epoch_i,
                            loss=valid_loss,
                            ppl=math.exp(min(valid_loss, 100)),
                            accu="Omitted",
                        )
                    )
