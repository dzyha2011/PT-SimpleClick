import os
import random
import logging
from copy import deepcopy
from collections import defaultdict

import torch.nn.functional as F

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 iterloss_weights=None
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        if iterloss_weights is not None:
            self.iterloss_weights = iterloss_weights
        else:
            self.iterloss_weights = list(range(1,max_num_next_clicks+1))
        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None

            # with torch.no_grad():
            num_iters = random.randint(1, self.max_num_next_clicks)
            loss = 0.0
            for click_indx in range(num_iters):
                if self.click_models is None or click_indx >= len(self.click_models):
                    eval_model = self.net
                else:
                    eval_model = self.click_models[click_indx]

                net_input = torch.cat((image, prev_output.detach()), dim=1) \
                    if self.net.with_prev_mask else image
                # output = self._forward(self.net, net_input, points)
                # with torch.cuda.amp.autocast():
                output = eval_model(net_input,points)
                ############################################################################
                if click_indx < num_iters - 1:
                    points = get_next_points(output['instances'], orig_gt_mask, points, click_indx + 1)
                if click_indx >= 1:
                    pre_mask = torch.where(prev_output > 0.5,1.0,0.0)
                    x = list(range(image.shape[0]))
                    y = list(torch.where(points[:,:,-1]==click_indx)[1].cpu().numpy())
                    if len(y) < image.shape[0]:
                        temp_points = points[:,:,-1].cpu()
                        temp_clicks = click_indx * torch.ones(points.shape[0],points.shape[1])
                        diff, _ = torch.abs(temp_clicks - temp_points).min(1)
                        ind = torch.where(diff!=0)[0].numpy()[0]
                        y.insert(ind,0)
                    newly_click = points[x,y,:2].unsqueeze(1)
                    if click_indx == 1:
                        newly_click = newly_click.repeat(1,points.shape[1],1)
                        init_points = deepcopy(points)
                        init_points[init_points==-1]=1000
                        dist_temp = newly_click - init_points[:,:,:2]
                        dist = dist_temp[:,:,0]**2 + dist_temp[:,:,1]**2
                        dist[dist==0] = 100000
                        dist_min, _ = dist.min(1)
                        dist_min = torch.sqrt(dist_min)
                    else:
                        xx = list(range(image.shape[0]))
                        yy = list(torch.where(points[:,:,-1]==click_indx-1)[1].cpu().numpy())
                        if len(yy) < image.shape[0]:
                            temp_points = points[:,:,-1].cpu()
                            temp_clicks = (click_indx - 1)  * torch.ones(points.shape[0],points.shape[1])
                            diff, _ = torch.abs(temp_clicks - temp_points).min(1)
                            ind = torch.where(diff!=0)[0].numpy()[0]
                            yy.insert(ind,0)
                        prev_click = points[xx,yy,:2]
                        dist_temp = newly_click.squeeze(1)-prev_click
                        dist = dist_temp[:,0]**2 + dist_temp[:,1]**2
                        dist_min = torch.sqrt(dist)

                    # similar_weights = torch.exp(-(dist_min-dist_min.mean())**2/dist_min.var()**2).cpu().numpy()
                    # similar_weights = (1/dist_min).cpu().numpy()
                    similar_weights = torch.exp(-dist_min/10)
                    if len(y) < image.shape[0]:
                        similar_weights[ind] = 0
                    loss = self.add_loss(
                        'similar_loss', loss, losses_logging, validation,
                        lambda: (output['instances'], pre_mask.detach()),
                        similar_weight=similar_weights)
                loss = self.add_loss(
                    'instance_loss',loss,losses_logging,validation,
                    lambda: (output['instances'],batch_data['instances']),
                    iter_step = click_indx + 1
                )

                

                # loss = self.add_loss(
                #     'instance_aux_loss', loss, losses_logging, validation,
                #     lambda: (output['instances'], batch_data['instances']),
                #     iterloss_step=click_indx,
                #     iterloss_weight=self.iterloss_weights[click_indx])

                prev_output = torch.sigmoid(output['instances'])

                if self.net.with_prev_mask and self.prev_mask_drop_prob > 0:
                    zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])
                

            # weight_matrix = self.get_weight_matrix(batch_data['instances'])
            weight_matrix = F.interpolate(batch_data['instances'],size=(28,28),mode='nearest')

            loss = self.add_loss('affinity_matrix_loss', loss, losses_logging, validation,
                                lambda: (output['affinity_matrix'], weight_matrix),
                                iter_step=click_indx+1
                                )

            loss = self.add_loss('instance_aux_loss',loss,losses_logging,validation,
                                 lambda: (output['instances_aux'], batch_data['instances']),
                                 iter_step = click_indx + 1
                                 )
            batch_data['points'] = points

        if self.is_master:
            with torch.no_grad():
                for m in metrics:
                    m.update(*(output.get(x) for x in m.pred_outputs),
                                *(batch_data[x] for x in m.gt_outputs))
        # loss.requires_grad_(True)
        return loss, losses_logging, batch_data, output

    # def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
    #     loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
    #     loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
    #     if loss_weight > 0.0:
    #         loss_criterion = loss_cfg.get(loss_name)
    #         loss = loss_criterion(*lambda_loss_inputs())
    #         loss = torch.mean(loss)
    #         losses_logging[loss_name] = loss
    #         loss = loss_weight * loss
    #         total_loss = total_loss + loss

    #     return total_loss
    def get_weight_matrix(self,target_mask):
        target_mask_resize = F.interpolate(target_mask,size=(28,28),mode='nearest')
        target_flatten_f = target_mask_resize.flatten(2).transpose(-2,-1)
        target_flatten_b = 1 - target_flatten_f
        weight_matrix = target_flatten_f @ target_flatten_f.transpose(-2,-1) + target_flatten_b @ target_flatten_b.transpose(-2,-1)
        return weight_matrix

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs, iter_step=None,similar_weight=None):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            if similar_weight is not None:
                # loss = similar_weight * loss.cpu()
                # loss = loss.to(self.device)
                for i in range(loss.shape[0]):
                    loss[i] = loss[i]*similar_weight[i]
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            if iter_step is not None and similar_weight is None:
                loss = loss * iter_step
            total_loss = total_loss + loss

        return total_loss

    # def add_loss(self, loss_name, total_loss, losses_logging, validation,
    #              lambda_loss_inputs, iterloss_step=None, iterloss_weight=1):
    #     loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
    #     loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
    #     if loss_weight > 0.0:
    #         loss_criterion = loss_cfg.get(loss_name)
    #         loss = loss_criterion(*lambda_loss_inputs())
    #         loss = torch.mean(loss)

    #         if iterloss_step is not None:
    #             losses_logging[
    #                 loss_name + f'_{iterloss_step}_{iterloss_weight}'
    #             ] = loss 
    #             loss = loss_weight * loss * iterloss_weight
    #         else:
    #             # iter mask (RITM)
    #             losses_logging[loss_name] = loss
    #             loss = loss_weight * loss

    #         total_loss = total_loss + loss

    #     return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    # new_state_dict = torch.load(path_to_weights, map_location='cpu')['models']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
