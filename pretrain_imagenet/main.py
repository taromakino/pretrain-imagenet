import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from erm import ERM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def main(args):
    pl.seed_everything(args.seed)
    data_train, data_val = make_data(args.batch_size, args.n_workers)
    model = ERM(args.lr, args.weight_decay)
    trainer = pl.Trainer(
        logger=CSVLogger(args.dpath, name='', version=args.seed),
        callbacks=[
            ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
        max_epochs=args.n_epochs,
        deterministic=True)
    trainer.fit(model, data_train, data_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=100)
    main(parser.parse_args())