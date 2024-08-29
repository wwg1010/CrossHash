import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.backends.cudnn as cudnn
from utils.tools import *
from model.crossformer import CrossFormer
import torch
import torch.optim as optim
import time
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
torch.multiprocessing.set_sharing_strategy('file_system')
from relative_similarity import *
from centroids_generator import *
import torch.nn.functional as F
from loss.loss import RelaHashLoss
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler
from loguru import logger
import CrossFormer_utils as utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

from crossFormer_args import parse_option
from loss.hypbird import margin_contrastive


def build_model(config, args, bit):
    model_type = config.MODEL.TYPE
    if model_type == 'cross-scale':
        model = CrossFormer(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.CROS.PATCH_SIZE,
                            in_chans=config.MODEL.CROS.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.CROS.EMBED_DIM,
                            depths=config.MODEL.CROS.DEPTHS,
                            num_heads=config.MODEL.CROS.NUM_HEADS,
                            group_size=config.MODEL.CROS.GROUP_SIZE,
                            mlp_ratio=config.MODEL.CROS.MLP_RATIO,
                            qkv_bias=config.MODEL.CROS.QKV_BIAS,
                            qk_scale=config.MODEL.CROS.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.CROS.APE,
                            patch_norm=config.MODEL.CROS.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            merge_size=config.MODEL.CROS.MERGE_SIZE,
                            hash_bit=bit)
    elif model_type == 'ViT-B_16':
        device = config['device']
        vit_config = VIT_CONFIGS[config["model_type"]]
        model = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=21, hash_bit=64).to(device)
        print('==> Loading from pretrained model..')
        model.load_from(np.load(config["pretrained_dir"]))
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def get_config():
    config = {
        "alpha": 0.1,
        'info': "[CrossFormer]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "datasets": "mirflickr",
        # "datasets": 'nuswide_21',
        # "datasets":'coco',
        "Label_dim" : 38,
        "epoch": 300,
        "test_map": 20,
        "save_path": "save/CrossFormer",
        "device": torch.device("cuda:0"),
        'test_device':torch.device("cuda:0"),
        "bit_list": [16,32,64,128],
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 10,
        "model_type": "CrossFormer",
        "top_img": 100,
        "Beta":10,
        'm':0.6,

    }
    config = config_dataset(config)
    return config


def train_val(config, bit, args=None, configs=None):
    device = torch.device(config['device'])

    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed(3407)

    cudnn.benchmark = True

    print(f"Creating model: {args.cfg}")

    model = build_model(configs, args, bit)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    input_size = [1, 3, 224, 224]
    input = torch.randn(input_size).cuda()
    from torchprofile import profile_macs
    if 'asmlp' not in args.cfg:
        macs = profile_macs(model.eval(), input)
        print('model flops:', macs, 'input_size:', input_size)
        model.train()

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    print('learning rate: ', args.lr)
    optimizer = create_optimizer(args, model)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    loss_scaler = ApexScaler()
    print('Using NVIDIA APEX AMP. Training in mixed precision.')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print('Pretrained Weiught Loaded')
        checkpoint['model'].pop('head.weight')
        checkpoint['model'].pop('head.bias')
        model.load_state_dict(checkpoint['model'], strict=False)
        print('\b\b\b\bLoaded Pretrained Model')

    model.to(config["device"])
    relative_similarity = RelativeSimilarity(nbit=bit, nclass=config["Label_dim"], batchsize=config["batch_size"])
    rela_optimizer = optim.Adam(relative_similarity.parameters(), lr=1e-5)

    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    quan_loss = RelaHashLoss(multiclass=True, beta=config['Beta'], m=config['m'])  # cifar10 _ False
    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["datasets"]), end="")
        model.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                u = model(image)
                logits = relative_similarity(u)
                w = (label.float() @ label.float().T > 0).float()
                socore = F.normalize(u) @ F.normalize(u).T
                loss2 = margin_contrastive(socore, w)
                q_loss = quan_loss(logits, label)
                train_loss += q_loss.item() + config["alpha"] * loss2.item()
            optimizer.zero_grad()
            rela_optimizer.zero_grad()

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(q_loss + config["alpha"] * loss2, optimizer, clip_grad=None,
                        parameters=model.parameters(), create_graph=is_second_order)
            rela_optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b train_loss:%.4f" % (train_loss))

        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % (train_loss))
        if (epoch + 1) % config['test_map'] == 0:
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, model, bit, epoch,
                                           num_dataset)
            model.to(config["device"])


if __name__ == '__main__':
    args, configs = parse_option()
    config = get_config()
    logger.add('logs/{time}' + config["info"] + '_' + config["datasets"] + ' alpha ' + str(config["alpha"]) + '.log',
               rotation='50 MB', level='DEBUG')
    logger.info(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/CrossHash_{config['datasets']}_{bit}.json"
        train_val(config, bit, args, configs)