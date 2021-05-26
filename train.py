import time
import setproctitle

import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
try:
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast
except:
    print("Loading amp module error.")
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

import functional.dataset.TrainLoader as TrainLoader
from models.mamp import MAMP
from parser_parameters import train_argument_parser
from functions import *

from raft_core.raft import *
from raft_core.utils.utils import *

import warnings
warnings.filterwarnings("ignore")


def main():
    args.training = True
    proc_name = args.proc_name
    os.makedirs(args.savepath, exist_ok=True)
    os.makedirs(os.path.join(args.savepath, proc_name), exist_ok=True)

    log = setup_logger(os.path.join(args.savepath, proc_name, 'training.log'))
    writer = SummaryWriter(os.path.join(args.savepath, 'runs', proc_name))

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    setproctitle.setproctitle(proc_name)

    torch.backends.cudnn.benchmark = True

    TrainData = TrainLoader.train_dataloader(args.datapath, args.ref_num)
    TrainImgLoader = torch.utils.data.DataLoader(
        TrainLoader.train_image_folder(args.datapath, TrainData, args, True),
        batch_size=args.bsize, shuffle=True, num_workers=args.worker, drop_last=True
    )

    model = MAMP(args).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    scaler = GradScaler(enabled=args.is_amp)
    log.info("AMP SWITCH STATUS IS {}".format(args.is_amp))

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            if args.is_amp:
                scaler.load_state_dict(checkpoint['scaler'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    model = nn.DataParallel(model).cuda()

    for epoch in range(start_epoch, args.epochs):
        train(TrainImgLoader, model, optimizer, scaler, log, writer, epoch)

        TrainData = TrainLoader.train_dataloader(args.datapath, args.ref_num)
        TrainImgLoader = torch.utils.data.DataLoader(
            TrainLoader.train_image_folder(args.datapath, TrainData, args, True),
            batch_size=args.bsize, shuffle=True, num_workers=args.worker, drop_last=True
        )

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


iteration = 0
def train(dataloader, model, optimizer, scaler, log, writer, epoch):
    global iteration
    log_loss = AverageMeter()
    batch_num = len(dataloader)
    b_s = time.perf_counter()

    for batch_ind, (images_lab, images_flow, meta) in enumerate(dataloader):
        model.train()

        adjust_lr(optimizer, epoch, batch_ind, batch_num)

        images_flow = None
        images_lab_msk = [lab.clone().cuda() for lab in images_lab]
        images_lab = [r.cuda() for r in images_lab]

        _, ch = model.module.dropout2d_lab(images_lab)

        b, c, h, w = images_lab[0].size()
        img_mem = [x for x in images_lab[:-1]]
        img_query = images_lab[-1]
        msk_mem = [x[:, ch] for x in images_lab_msk[:-1]]
        msk_query = images_lab_msk[-1][:, ch]

        results = model(img_mem, msk_mem, img_query, None)
        results = F.interpolate(results, (h, w), mode='bilinear')
        sum_loss = F.smooth_l1_loss(results * 20, msk_query * 20, reduction='mean')

        if not args.is_amp:
            sum_loss.backward()
            optimizer.step()
        else:
            scaler.scale(sum_loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad()
        if -10000 < sum_loss.item() < 100000:
            log_loss.update(sum_loss.item())
        iteration = iteration + 1
        writer.add_scalar("Training loss", sum_loss.item(), iteration)

        info = 'Loss = {:.3f}({:.3f})'.format(log_loss.val, log_loss.avg)
        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        if batch_ind % 20 == 0:
            for param_group in optimizer.param_groups:
                lr_now = param_group['lr']
            log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
                epoch, batch_ind, batch_num, info, batch_time, lr_now))

    savefilename = os.path.join(args.savepath, f'{args.proc_name}_ckpt_epoch_{epoch+1}.pt')
    log.info("Saving checkpoint {}.".format(savefilename))

    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if args.is_amp else None,
    }, savefilename)


def adjust_lr(optimizer, epoch, batch_ind, batch_num):
    iteration = (batch_ind + epoch * batch_num) * args.bsize

    if iteration <= 400000:
        lr = args.lr
    elif iteration <= 600000:
        lr = args.lr * 0.5
    elif iteration <= 800000:
        lr = args.lr * 0.25
    elif iteration <= 1000000:
        lr = args.lr * 0.125
    else:
        lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = train_argument_parser().parse_args()
    main()
