import pytorch_lightning as pl
from torch import optim, nn, utils, Tensor
from model import TransformerModel_no_softmax
from loss import edl_digamma_loss
from dataset import prepare_data,tabular_transformer
from torch.utils.data import DataLoader
import torch
from torchmetrics import Accuracy, F1Score, AUROC
from utils import one_hot_embedding, relu_evidence


class LitTransformer(pl.LightningModule):

    def __init__(self, model: nn.Module, uncertainty:bool=False,lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.uncertainty = uncertainty
        self.num_classes = 2
        self.annealing_step = 10
        if uncertainty:
            self.loss = edl_digamma_loss
        else:
            self.loss = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)
        self.val_f1 = F1Score(task = "binary", num_classes=2)
        self.test_f1 = F1Score(task = "binary", num_classes=2)
        self.val_auroc = AUROC(task = "binary")
        self.test_auroc = AUROC(task = "binary")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: utils.data.DataLoader, batch_idx: int) -> Tensor:
        x, y = batch
        epoch_num = self.current_epoch
        if self.uncertainty:
            #y = one_hot_embedding(y, self.num_classes)
           
            outputs = model(x)
            #_, preds = torch.max(outputs, 1)
            #preds = outputs[:,1]

            loss = self.loss(
                outputs, y.float(), epoch_num, self.num_classes, self.annealing_step, self.device
            )

            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            y_hat = prob#[:,1]
 

        else:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(y_hat, y)
                 , on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: utils.data.DataLoader, batch_idx: int) -> Tensor:
        x, y = batch
        if self.uncertainty:
            #y = one_hot_embedding(y, self.num_classes)
            outputs = model(x)

            loss = self.loss(
                outputs, y.float(), self.current_epoch, self.num_classes, self.annealing_step, self.device
            )
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            y_hat = prob#[:,1]
        else:

            y_hat = self.model(x)
            loss = self.loss(y_hat, y)

        self.log('val_loss', loss,on_epoch=True, prog_bar=True)
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_auroc(y_hat, y)
        return loss

    def test_step(self, batch: utils.data.DataLoader, batch_idx: int) -> Tensor:
        x, y = batch
        if self.uncertainty:
            #y = one_hot_embedding(y, self.num_classes)
            outputs = model(x)
            loss = self.loss(
                outputs, y.float(), self.current_epoch, self.num_classes, self.annealing_step, self.device
            )
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            y_hat = prob#[:,1]
        else:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        # y_pred = torch.argmax(y_hat, dim=1)

        self.log('test_loss', loss)
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_auroc(y_hat, y)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.log('val_f1_epoch', self.val_f1.compute(), on_epoch=True, prog_bar=True)
        self.log('val_auroc_epoch', self.val_auroc.compute(), on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_acc.compute())
        self.log('test_f1_epoch', self.test_f1.compute())
        self.log('test_auroc_epoch', self.test_auroc.compute())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
        return ([optimizer], [scheduler])

    def predict(self, x: Tensor) -> Tensor:
        return self.model(x)
    
if __name__ =="__main__":
    #ops
    dataset = "PIE"
    uncertainty = False
    ip_dim = 7
    seq_len = 15
    d_model = 32
    nhead = 2
    lr = 5e-3
    weight_decay = 1e-4
    bs = 32
    nlayers = 2
    dropout = 0.1


    prep = prepare_data(dataset)
    train_data, val_data, test_data = prep.train_data, prep.val_data, prep.test_data
    train_dataset = tabular_transformer(train_data)
    val_dataset = tabular_transformer(val_data)
    test_dataset = tabular_transformer(test_data)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = TransformerModel_no_softmax(ip_dim=ip_dim, seq_len=seq_len, d_model=d_model,
                                         nhead=nhead, d_hid=d_model, nlayers=nlayers, dropout=dropout).double()
    lit_model = LitTransformer(model, uncertainty=uncertainty, lr=lr, weight_decay=weight_decay)

    #early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc_epoch',
        min_delta=1e-5,
        patience=20,
        verbose=True,
        mode='max'
    )

    #trainer
    trainer = pl.Trainer(accelerator="cuda",devices=1, max_epochs=1000, callbacks=[early_stop_callback])
    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader)
    print("Done")
