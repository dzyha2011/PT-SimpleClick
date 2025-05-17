import os
import argparse
import importlib.util

import torch
from isegm.utils.exp import init_experiment


def main():
    args = parse_args()
    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.distributed = 'WORLD_SIZE' in os.environ
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='models/iter_mask/plainvit_base448_sbd_itermask.py',
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=16,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--ngpus', type=int, default=2,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='0,1', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')
    
    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=13,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument("--local_rank", type=int, default=0)

    # parameters for experimenting
    parser.add_argument('--layerwise-decay', action='store_true', 
                        help='layer wise decay for transformer blocks.')

    parser.add_argument('--upsample', type=str, default='x1', 
                        help='upsample the output.')

    parser.add_argument('--random-split', action='store_true', 
                        help='random split the patch instead of window split.')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()
    from scripts.evaluate_model import main as test
    import shutil
    import os
    dir = os.listdir("E:\\code\\SimpleClick\\weights\\iter_mask\\sbd_plainvit_base448")
    basedir = "E:/code/SimpleClick/weights/iter_mask/sbd_plainvit_base448"
    path2pth = "checkpoints\\last_checkpoint.pth"
    # srcfile = "E:\\code\\SimpleClick\\weights\\iter_mask\\sbd_plainvit_base448\\003\\checkpoints\\last_checkpoint.pth"
    srcfile = os.path.join(basedir,dir[-1],path2pth)
    tarfolder = "E:\\code\\SimpleClick\\weights\\simpleclick_models"
    shutil.copy(srcfile,tarfolder)
    test()