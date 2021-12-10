from tqdm import tqdm
import torch
import config

class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train(self, train_dataloader, epoch, grad_scaler):
        self.model.train()
        self.model.zero_grad()

        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        epoch_loss = 0
        dataset_len = 0

        for i, batch_data in bar:

            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)
            
            with torch.cuda.amp.autocast(enabled=config.AMP):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            dataset_len += 1

            bar.set_postfix(epoch=epoch, loss=epoch_loss / dataset_len,
                        lr=self.optimizer.param_groups[0]['lr'])
        
        self.scheduler.step(epoch_loss / dataset_len)
        return epoch_loss / dataset_len
    
    @torch.no_grad()
    def evaluate(self, val_dataloader, epoch, logger=None):
        self.model.eval()

        bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        epoch_loss = 0
        dataset_len = 0

        for i, batch_data in bar:
            images = batch_data['inputs'].to(config.DEVICE)
            labels = batch_data['labels'].to(config.DEVICE)

            preds = self.model(images)
            loss = self.loss_fn(preds, labels)

            epoch_loss += loss.item()
            dataset_len += 1

            bar.set_postfix(epoch=epoch, loss=epoch_loss / dataset_len)

        return epoch_loss / dataset_len