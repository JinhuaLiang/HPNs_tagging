import os
import sys
import time
import tqdm
import yaml
import hydra
# import wandb
import torch
import numpy as np
from string import Template
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Optional

sys.path.insert(0, '../')
import src.label_ops as lops
from src.manifolds import make_folder, set_logger
from pytorch.models import Vgg8, MatchingNetwork
from pytorch.utils import collate_data, loss_fn, task_label, prepare_fewshot_task
from pytorch.eval_kit import Metrics
from pytorch.data import ESC50, ESC50FewShotSampler
from pytorch.fewshot import Protonet_onEpisode, Matchnet_onEpisode, HierProtonet_onEpisode

# Automatically allocate a spare gpu
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')

# Set configuration file
config_path = '../cfg'
config_name = 'fewshot.yaml'
# Monitor the model performance, see more in https://github.com/wandb/client
with open(os.path.join(config_path, config_name)) as f:
    yaml_data = yaml.safe_load(f)
t = Template(yaml_data['OUTPUTS']['DIR'].replace('.', '__'))
output_dir = t.substitute(DATASOURCE__NAME=yaml_data['defaults'][0]['DATASOURCE'],
                          LEARNER__ON_EPISODE=yaml_data['LEARNER']['ON_EPISODE'],
                          LEARNER__DATASAMPLING=yaml_data['LEARNER']['DATASAMPLING']['train'],
                          TRAINER__LEARNING_RATE=float(yaml_data['TRAINER']['LEARNING_RATE']),
                          LEARNER__BATCH_SIZE=yaml_data['LEARNER']['BATCH_SIZE'])
output_dir = os.path.join(output_dir, f"{time.strftime('%b-%d_%H-%M', time.localtime())}")
# os.environ['WANDB_DIR'] = output_dir
# make_folder(os.environ['WANDB_DIR'])
# wandb.init(project="AudioTagging", entity="jinhua_liang")
# Multiple processing
torch.multiprocessing.set_sharing_strategy('file_system')
# Set root logger
logger = set_logger(log_dir=output_dir)


def train(cfg: OmegaConf) -> None:
    """ Train a specific model"""
    # Define local dir, param, and func
    dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
    n_novel_cls = cfg['LEARNER']['NUM_NVL_CLS']
    n_support_per_cls = cfg['LEARNER']['NUM_SUPPORT_PER_CLS']
    n_query_per_cls = cfg['LEARNER']['NUM_QUERY_PER_CLS']
    device = torch.device('cuda') if cfg['LEARNER']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')
    # Cross-validation
    history = [{} for _ in range(cfg['TRAINER']['K'])]  # k-folds cross validation
    for fold in range(cfg['TRAINER']['K']):  # K is defined arbitrarily as we can constitute different splits
        logger.info('====================================================================================')
        logger.info(f"Experiments: {fold + 1}/{cfg['TRAINER']['K']}")
        # Set dataset & evaluator
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
            all_fold = [x for x in range(1, 6)]  # use all the fold per class

            trainset = ESC50(
                wav_dir=os.path.join(dataset_dir, 'audio'),
                csv_path=csv_path,
                fold=all_fold,
                num_class=cfg['DATASOURCE']['NUM_CLASS'],
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE']
            )
            # Set up data split for few shot learning & evaluation
            novel_splt_size = int(cfg['DATASOURCE']['NUM_CLASS'] * cfg['LEARNER']['LABEL_SPLIT'][0])
            if cfg['LEARNER']['LABEL_SPLIT'][1] == 'random':
                base_split, novel_split = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])),
                                                            n_novel_cls=novel_splt_size)
            elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'uniform':
                base_split, novel_split = lops.uniform_split(csv_path=csv_path, n_novel_cls=novel_splt_size)
            elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'parent':
                base_split, novel_split = lops.parent_split(csv_path=csv_path, prnt_id=(fold + 1),
                                                            n_novel_cls=novel_splt_size)
            elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'select':
                assert len(lops.esc50_select_ids) == cfg['TRAINER']['K']
                novel_split = lops.esc50_select_ids[fold]
                base_split = [x for x in range(cfg['DATASOURCE']['NUM_CLASS']) if x not in novel_split]
                if len(novel_split) != novel_splt_size:
                    logger.warning(
                        f"Mismatched split size: select {len(novel_splt_size)} labels instead of {novel_splt_size}")
                assert (set(base_split) & set(novel_split) == set())
            evaluator = Evaluator(cfg, eval_fold=all_fold, slt_cls=novel_split)

        else:
            logger.warning(f"Cannot recognise `dataset` {cfg['DATASOURCE']['NAME']}")

        # Set data sampler
        if cfg['LEARNER']['DATASAMPLING']['train'] == 'esc50_fs':
            trainsampler = ESC50FewShotSampler(
                dataset=trainset,
                num_nvl_cls=n_novel_cls,
                num_sample_per_cls=n_support_per_cls,
                num_queries_per_cls=n_query_per_cls,
                num_task=cfg['LEARNER']['NUM_TASK'],
                batch_size=cfg['LEARNER']['BATCH_SIZE'],
                fix_split=base_split,
                require_pid=cfg['TRAINER']['REQUIRE_PID']
            )

        else:
            logger.warning(f"{cfg['LEARNER']['DATASAMPLING']['train']} is not in the pool of sampler.")

        trainloader = DataLoader(trainset, batch_sampler=trainsampler, collate_fn=collate_data, num_workers=4,
                                 pin_memory=True)

        if cfg['LEARNER']['MODEL']['NAME'] == 'vgg8':
            model = Vgg8(
                num_class=n_novel_cls,
                sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
                n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
                win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
                hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
                f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
                f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
                n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
                window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
                include_top=False
            ).to(device)
        elif cfg['LEARNER']['MODEL']['NAME'] == 'match_net':
            model = MatchingNetwork(
                lstm_layers=1,
                lstm_input_size=256,
                unrolling_steps=2,
                device=device,
                mono=cfg['DATASOURCE']['SINGLE_TARGET']
            ).to(device)
        else:
            raise ValueError(f"Cannot recognise the model 'cfg['LEARNER']['MODEL']['NAME']'.")
        # Set loss & optimiser for back propagation in the training
        criterion = loss_fn(name=cfg['LEARNER']['LOSS_FN'],
                            single_target=cfg['DATASOURCE']['SINGLE_TARGET'],
                            reduction='sum')
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg['TRAINER']['LEARNING_RATE'])
        # Set on-episode func for few shot learning
        if cfg['LEARNER']['ON_EPISODE'] == 'proto':
            train_on_episode = Protonet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                distance='l2_distance',
                is_mono=True,
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['LEARNER']['ON_EPISODE'] == 'match':
            train_on_episode = Matchnet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                distance='cosine',
                is_mono=cfg['DATASOURCE']['SINGLE_TARGET'],
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )
        elif cfg['LEARNER']['ON_EPISODE'] == 'hier_proto':
            train_on_episode = HierProtonet_onEpisode(
                n_nvl_cls=n_novel_cls,
                n_support_per_cls=n_support_per_cls,
                n_query_per_cls=n_query_per_cls,
                height=2,
                alpha=-1,
                distance='l2_distance',
                criterion=criterion,
                is_train=True,
                optimiser=optimiser,
                require_y=False
            )

        # Resume training if ckpt of pretrained model is provided
        if cfg['TRAINER']['RESUME_TRAINING']:
            ckpt = torch.load(cfg['LEARNER']['MODEL']['PRETRAINED_PATH'])
            model.load_state_dict(ckpt)
            logger.info("Load the weight from {cfg['LEARNER']['MODEL']['PRETRAINED_PATH']}")

        """ Training on episodes."""
        best_score = 0
        for epoch in range(cfg['TRAINER']['EPOCHS']):
            model.train()

            running_loss = 0
            with tqdm.tqdm(total=len(trainloader), desc=f"epoch {epoch}/{cfg['TRAINER']['EPOCHS']}") as t:
                for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(trainloader)):
                    loss = train_on_episode(model, batch_x.to(device), batch_y.to(device))
                    running_loss += loss
                    # wandb.log({'train_loss': (running_loss / (i + 1))})
                    # wandb.watch(model)
                    t.set_postfix(loss=f"{(running_loss / (i + 1)):.4f}")
                    t.update()

            logger.info(f"Epoch {epoch}: train_loss={(running_loss / len(trainloader))}")
            # Evaluate the model
            statistics = evaluator.evaluate(model)
            # Log the metrics to wandb
            # wandb.log(statistics)
            # Add to history values
            for k in statistics.keys():
                if k not in history[fold].keys():
                    history[fold][k] = [statistics[k]]
                else:
                    history[fold][k].append(statistics[k])
            # Save the checkpoint when conducting the best performance
            current_score = statistics['acc'] if cfg['DATASOURCE']['SINGLE_TARGET'] else statistics['map']
            if output_dir and (current_score > best_score):
                ckpt_dir = os.path.join(output_dir, 'ckpts')
                make_folder(ckpt_dir)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"epoch_{epoch}-{current_score:.4f}.pth"))
                best_score = current_score
            logger.info(f"epoch {epoch}: the best performance of training model is {best_score}")

    # Calculate macro result over multiple folds
    result = dict()
    for fold_h in history:
        for key in fold_h.keys():
            if key in result.keys():
                result[key] += np.amax(fold_h[key])
            else:
                result[key] = np.amax(fold_h[key])

    logger.info(f"Summarise this trained model's performance:")
    for key in result.keys():
        result[key] /= cfg['TRAINER']['K']  # average the result of different folds
        logger.info(f"Overall_eval_{key}={result[key]:.6f}")


def test(cfg: OmegaConf, novel_cls: Optional[list] = None) -> None:
    """ Evaluate the trained model."""
    # Define local dir, param, and func
    dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
    n_novel_cls = cfg['LEARNER']['NUM_NVL_CLS']
    device = torch.device('cuda') if cfg['LEARNER']['CUDA'] and torch.cuda.is_available() else torch.device('cpu')
    # Modify some opts in cfg
    cfg['LEARNER']['NUM_QUERY_PER_CLS'] = cfg['TESTER']['NUM_QUERY_PER_CLS']
    cfg['LEARNER']['SHUFFLE_CLASS'] = cfg['TESTER']['SHUFFLE_CLASS']
    # Set up data split for evaluation
    if cfg['DATASOURCE']['NAME'] == 'esc50':
        csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
        all_fold = [x for x in range(1, 6)]  # use all the fold per class
        fold_id = 1  # fold id reserved for test [1, 6)

        novel_splt_size = int(cfg['DATASOURCE']['NUM_CLASS'] * cfg['LEARNER']['LABEL_SPLIT'][0])
        if cfg['LEARNER']['LABEL_SPLIT'][1] == 'random':
            _, novel_cls = lops.random_split(list(range(cfg['DATASOURCE']['NUM_CLASS'])),
                                             n_novel_cls=novel_splt_size)
        elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'uniform':
            _, novel_cls = lops.uniform_split(csv_path=csv_path, n_novel_cls=novel_splt_size)
        elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'parent':
            _, novel_cls = lops.parent_split(csv_path=csv_path, prnt_id=fold_id, n_novel_cls=novel_splt_size)
        elif cfg['LEARNER']['LABEL_SPLIT'][1] == 'select':
            if len(novel_cls) != novel_splt_size:
                logger.warning(
                    f"Mismatched split size: select {len(novel_splt_size)} labels instead of {novel_splt_size}")
        logger.info(f"Now testing on {novel_cls}")
        evaluator = Evaluator(cfg, eval_fold=all_fold, slt_cls=novel_cls)

    else:
        logger.warning(f"Cannot recognise `dataset` {cfg['DATASOURCE']['NAME']}")

    # Set audio encoder
    if cfg['LEARNER']['MODEL']['NAME'] == 'vgg8':
        model = Vgg8(
            num_class=n_novel_cls,
            sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'],
            n_fft=cfg['FEATURE_EXTRACTOR']['N_FFT'],
            win_length=cfg['FEATURE_EXTRACTOR']['WIN_LENGTH'],
            hop_length=cfg['FEATURE_EXTRACTOR']['HOP_LENGTH'],
            f_min=cfg['FEATURE_EXTRACTOR']['F_MIN'],
            f_max=cfg['FEATURE_EXTRACTOR']['F_MAX'],
            n_mels=cfg['FEATURE_EXTRACTOR']['N_MELS'],
            window_type=cfg['FEATURE_EXTRACTOR']['WINDOW_TYPE'],
            include_top=False
        ).to(device)
    elif cfg['LEARNER']['MODEL']['NAME'] == 'match_net':
        model = MatchingNetwork(
            lstm_layers=1,
            lstm_input_size=256,
            unrolling_steps=2,
            device=device,
            mono=cfg['DATASOURCE']['SINGLE_TARGET']
        ).to(device)
    else:
        raise ValueError(f"Cannot recognise the model 'cfg['LEARNER']['MODEL']['NAME']'.")

    # Load weights of trained model
    if not cfg['LEARNER']['MODEL']['PRETRAINED_PATH']:
        logger.warning("Checkpoint of a trained model must be provided.")
    else:
        ckpt = torch.load(cfg['LEARNER']['MODEL']['PRETRAINED_PATH'])
        model.load_state_dict(ckpt)
        logger.info(f"Load the weight from '{cfg['LEARNER']['MODEL']['PRETRAINED_PATH']}'.")

    statistics = evaluator.evaluate(model)
    for k in statistics.keys():
        _res = np.mean(statistics[k])
        logger.info(f"Overall_eval_{k}={_res:.6f}")


class Evaluator(object):
    def __init__(self, cfg: OmegaConf, **kwargs):
        # Opts
        dataset_dir = cfg['DATASOURCE']['DATASET_DIR']
        mode = cfg['LEARNER']['MODE']
        self.single_target = cfg['DATASOURCE']['SINGLE_TARGET']
        self.criterion = loss_fn(name=cfg['LEARNER']['LOSS_FN'], single_target=cfg['DATASOURCE']['SINGLE_TARGET'],
                                 reduction='sum')
        avg = None if cfg['LEARNER']['MODE'] == 'test' else 'macro'
        self.metrics = Metrics(single_target=cfg['DATASOURCE']['SINGLE_TARGET'], average=avg)
        self.device = torch.device('cuda') if cfg['LEARNER']['CUDA'] and torch.cuda.is_available() else torch.device(
            'cpu')
        # Few shot params
        self.n_novel_cls = cfg['LEARNER']['NUM_NVL_CLS']
        self.n_support_per_cls = cfg['LEARNER']['NUM_SUPPORT_PER_CLS']
        self.n_query_per_cls = cfg['LEARNER']['NUM_QUERY_PER_CLS']
        slt_cls = kwargs['slt_cls']
        # Set data source
        if cfg['DATASOURCE']['NAME'] == 'esc50':
            fold = kwargs['eval_fold']
            dataset = ESC50(wav_dir=os.path.join(dataset_dir, 'audio'),
                            csv_path=os.path.join(dataset_dir, 'meta', 'esc50.csv'),
                            fold=fold,
                            num_class=cfg['DATASOURCE']['NUM_CLASS'],
                            sample_rate=cfg['FEATURE_EXTRACTOR']['SAMPLE_RATE'])
            logger.info(f"Now evaluate the model on the classes {slt_cls} with the fold={fold}")

        # Set data sampling
        if cfg['LEARNER']['DATASAMPLING']['eval'] == 'esc50_fs':
            datasampler = ESC50FewShotSampler(
                dataset=dataset,
                num_nvl_cls=self.n_novel_cls,
                num_sample_per_cls=self.n_support_per_cls,
                num_queries_per_cls=self.n_query_per_cls,
                num_task=cfg['LEARNER']['NUM_TASK'],
                batch_size=cfg['LEARNER']['BATCH_SIZE'],
                fix_split=slt_cls,
                require_pid=cfg['TRAINER']['REQUIRE_PID']
            )
        else:
            logger.warning(f"{cfg['LEARNER']['DATASAMPLING']['eval']} is not in the pool of sampler.")

        self.dataloader = DataLoader(dataset, batch_sampler=datasampler, collate_fn=collate_data, num_workers=4,
                                     pin_memory=True)

        # Set on-episode func for few shot learning
        if cfg['LEARNER']['ON_EPISODE'] == 'proto':
            self.eval_on_episode = Protonet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                distance='l2_distance',
                is_mono=self.single_target,
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['LEARNER']['ON_EPISODE'] == 'match':
            self.eval_on_episode = Matchnet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                distance='cosine',
                is_mono=self.single_target,
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )
        elif cfg['LEARNER']['ON_EPISODE'] == 'hier_proto':
            self.eval_on_episode = HierProtonet_onEpisode(
                n_nvl_cls=self.n_novel_cls,
                n_support_per_cls=self.n_support_per_cls,
                n_query_per_cls=self.n_query_per_cls,
                height=2,
                alpha=-1,
                distance='l2_distance',
                criterion=self.criterion,
                is_train=False,
                optimiser=None,
                require_y=True
            )


    def evaluate(self, model: torch.nn.Module) -> dict:
        """ Evaluate the model's performance."""
        model.eval()

        running_loss = 0
        predictions = list()
        labels = list()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(self.dataloader)):
                loss, batch_outputs, ground_truth = self.eval_on_episode(model, batch_x.to(self.device),
                                                                         batch_y.to(self.device))
                running_loss += loss
                if self.single_target:
                    preds = np.argmax(batch_outputs.numpy(), axis=1).tolist()  # convert ont-hot label into label idx
                    gt = ground_truth.numpy().tolist()
                else:
                    batch_outputs, ground_truth = batch_outputs.numpy(), ground_truth.numpy()
                    n_samples, _ = batch_outputs.shape
                    assert n_samples == ground_truth.shape[0]
                    preds = [batch_outputs[i, :] for i in range(n_samples)]
                    gt = [ground_truth[i, :] for i in range(n_samples)]
                predictions.extend(preds)
                labels.extend(gt)
        statistics = self.metrics.summary(labels, predictions)
        statistics['loss'] = running_loss / len(self.dataloader)
        for k, v in statistics.items():
            logger.info(f"val_{k}={v}")
        return statistics


@hydra.main(config_path=config_path, config_name=config_name.split('/')[0])
def main(cfg: OmegaConf) -> None:
    logger.info(
        f"====================================================================================\n"
        f"Configuration:"
        f"{cfg}"
    )
    if cfg['LEARNER']['MODE'] == 'train':
        train(cfg)
    elif cfg['LEARNER']['MODE'] == 'test':
        test(cfg)


if __name__ == '__main__':
    main()
