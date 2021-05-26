import time
import torch.nn.parallel
import torch.backends.cudnn
from tqdm import tqdm
import setproctitle
import traceback
import shutil

from models.mamp import MAMP
from parser_parameters import test_argument_parser
from functions import *
from functional.dataset.TestLoader import *
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from functional.utils.mask_io import save_mask, zip_folder

from raft_core.raft import *
from raft_core.utils.utils import *

import warnings
warnings.filterwarnings("ignore")


def main():
    args.training = False

    os.makedirs(args.savepath, exist_ok=True)
    os.makedirs(os.path.join(args.savepath, args.proc_name.split('.')[0]), exist_ok=True)
    log = setup_logger(os.path.join(args.savepath, args.proc_name.split('.')[0], 'benchmark.log'))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    setproctitle.setproctitle(args.proc_name.split('.')[0])
    TestData = dataloader_davis(args.datapath)
    TestImgLoader = torch.utils.data.DataLoader(
        test_image_folder_davis(TestData, False),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )

    model = MAMP(args)
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    log.info("AMP SWITCH STATUS IS {}".format(args.is_amp))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).cuda()

    # Optical flow
    if args.optical_flow_warp:
        args.small = False  # raft-small.pth
        args.mixed_precision = True
        of_model = nn.DataParallel(RAFT(args))
        ckpt_name = 'ckpt/raft-small.pth' if args.small else 'ckpt/raft-sintel.pth'
        of_model.load_state_dict(torch.load(ckpt_name))
        of_model.cuda()
        of_model.eval()
    else:
        of_model = None

    start_full_time = time.time()
    with torch.no_grad():
        evaluate(TestImgLoader, model, log, of_model)
    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def evaluate(dataloader, model, log, of_model):
    model.eval()
    torch.backends.cudnn.benchmark = True

    Fs = AverageMeter()
    Js = AverageMeter()
    video_num = len(dataloader)
    folder = os.path.join(args.savepath, args.proc_name.split('.')[0], 'DAVIS')
    try:
        shutil.rmtree(folder)
    except:
        a = None
    os.makedirs(folder, exist_ok=True)

    log.info("Start testing.")
    for video_index, (images, annotations, meta) in tqdm(enumerate(dataloader)):
        PFs = AverageMeter()
        PJs = AverageMeter()
        ModelTime = AverageMeter()

        video_name = meta["video_name"][0]
        annotation_index = [x.item() for x in meta["annotation_index"]]
        frame_names = [x[0] for x in meta["frame_names"]]
        height, width = meta['height'][0].item(), meta['width'][0].item()
        padded_height, padded_width = height, width
        abs_frame_path = [x[0] for x in meta["abs_frame_path"]]
        video_frames = None
        if args.optical_flow_warp:
            video_frames = [load_image(x) for x in abs_frame_path]

        if args.pad_divisible > 1:
            divisible = args.pad_divisible
            cur_b, cur_c, cur_h, cur_w = images[0].shape
            pad_h = 0 if (cur_h % divisible) == 0 else divisible - (cur_h % divisible)
            pad_w = 0 if (cur_w % divisible) == 0 else divisible - (cur_w % divisible)

            if (pad_h + pad_w) != 0:
                pad = nn.ZeroPad2d(padding=(0, pad_w, 0, pad_h))
                images = [pad(x) for x in images]
                annotations = [pad(x) for x in annotations]
                video_frames = [pad(x) for x in video_frames] if args.optical_flow_warp else None
                padded_height += pad_h
                padded_width += pad_w

        outputs = [annotations[0].contiguous()]
        N = len(images)
        for i in range(N - 1):
            ref_index = get_davis_ref_index(i, 2, args.memory_length)
            img_mem = [images[ind].cuda() for ind in ref_index]
            msk_mem = [outputs[ind].cuda() for ind in ref_index]
            img_query = images[i + 1].cuda()
            msk_query = annotations[i + 1].cuda()

            flow_img_mem, flow_img_query, optical_flows = [], [], []
            if args.optical_flow_warp:
                flow_img_mem = [video_frames[ind].cuda() for ind in ref_index]
                flow_img_query = video_frames[i + 1].cuda()
                for img, ind in zip(flow_img_mem, ref_index):
                    long_gap = 15
                    iter_num = 5 if (i + 1) - ind > long_gap else 2
                    with torch.no_grad():
                        _, flow_up = of_model(flow_img_query, img, iters=iter_num, test_mode=True, up_scale=2)
                    flow_up = clamp_optical_flow(flow_up)
                    optical_flows.append(flow_up)

            with torch.no_grad():
                s_ = time.time()
                _output = model(img_mem, msk_mem, img_query, optical_flows)
                _output = F.interpolate(_output, (padded_height, padded_width), mode='bilinear')
                output = torch.argmax(_output, 1, keepdim=True).float()
                ModelTime.update(time.time() - s_)

            outputs.append(output.cpu())
            max_class = msk_query.max()
            js, fs = [], []
            for classid in range(1, max_class + 1):
                obj_true = (msk_query[:, :, :height, :width] == classid).cpu().numpy()[0, 0]  # unpadding is required
                obj_pred = (output[:, :, :height, :width] == classid).cpu().numpy()[0, 0]
                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)
                fs.append(f)
                js.append(j)
                Fs.update(f)
                Js.update(j)
                PFs.update(f)
                PJs.update(j)

            output_folder = os.path.join(folder, video_name)
            os.makedirs(output_folder, exist_ok=True)
            if i == 0:
                output_file = os.path.join(output_folder, frame_names[0])
                out_img = annotations[0][0, 0][:height, :width].cpu().numpy().astype(np.uint8)
                save_mask(output_file, out_img)

            output_file = os.path.join(output_folder, frame_names[i + 1])
            out_img = output[0, 0][:height, :width].cpu().numpy().astype(np.uint8)
            save_mask(output_file, out_img)

            # torch.cuda.empty_cache()

        # J&F accumulated performance; PJ&F performance on one video.
        performance = '\t'.join([
            'Js: ({:.3f}). Fs: ({:.3f}). J&F: ({:.4f}),  PJs: ({:.3f}). PFs: ({:.3f}). PJ&F: ({:.4f}). FPS: ({:.1f})'
                .format(Js.avg, Fs.avg, (Js.avg + Fs.avg) / 2, PJs.avg, PFs.avg, (PJs.avg + PFs.avg) / 2, 1 / ModelTime.avg)])
        log.info('[{}/{}] {}: {}'.format(video_index + 1, video_num, video_name, performance))

    return None


if __name__ == '__main__':
    args = test_argument_parser().parse_args()
    main()

