import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from isegm.model.ops import DistMaps, BatchImageNormalize, ScaleLayer
# from lattice import GCN, lattice
# import dgl
from .modeling.resnet import ResNet50


class ISModel(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, cpu_dist_maps=False,
                 use_rgb_conv=False, use_leaky_relu=False, # the two arguments only used for RITM
                 with_prev_mask=False, norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if self.with_prev_mask:
            self.coord_feature_ch += 1

        # self.resnet = ResNet50(output_stride=32,BatchNorm=nn.BatchNorm2d)
        # self.d1 = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1,stride=1)
        
        if use_rgb_conv:
            # Only RITM models need to transform the coordinate features, though they don't use 
            # exact 'rgb_conv'. We keep 'use_rgb_conv' only for compatible issues.
            # The simpleclick models use a patch embedding layer instead 
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        else:
            self.maps_transform=nn.Identity()

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps, use_disks=use_disks)

        self.dist_maps_10 = DistMaps(norm_radius=10,spatial_scale=1.0,cpu_mode=cpu_dist_maps,use_disks=use_disks)

        self.init_weights()

    ## load state_dict from pretrained weights
    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path,map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict['state_dict'].items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        # interative_map = coord_features[:,1:]
        # resnet_input = torch.cat([image,interative_map],1)
        # layer1,layer2,layer3,layer4 = self.resnet(resnet_input)
        # external_input = [layer1,layer2,layer3,layer4]
        coord_features = self.maps_transform(coord_features)
        coord_weights = self.get_coord_weights(image,points)
        outputs = self.backbone_forward(image, coord_features,coord_weights)

        # resnet_pred = self.d1(layer4)

        # outputs['mid_pred'] = nn.functional.interpolate(resnet_pred, size=image.size()[2:],
        #                                                  mode='bilinear', align_corners=True)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features

    def get_coord_weights(self,image,points):
        coord_features = self.dist_maps_10(image,points)
        # coord_features = 1 - coord_features
        B,C,H,W = coord_features.shape
        coord_features = coord_features.view(B,C,H//16,16,W//16,16)
        coord_features = torch.mean(coord_features,3)
        coord_features = torch.mean(coord_features,-1)
        # coord_ints = torch.where(coord_features>0,1,0)
        # coord_ints = coord_ints.float()
        # coord_ints_flatten = (coord_ints[:,0]-coord_ints[:,1]).flatten(1).unsqueeze(1).unsqueeze(3)
        # coord_ints_matrix = torch.matmul(coord_ints_flatten,coord_ints_flatten.permute(0,1,3,2))
        # coord_features_flatten = coord_features.flatten(2)[:,0]-coord_features.flatten(2)[:,1]
        # coord_features_flatten = coord_features_flatten.unsqueeze(1).unsqueeze(3).repeat(1,1,1,(H//16)*(W//16))
        # coord_distance_matrix = torch.abs(coord_features_flatten-coord_features_flatten.permute(0,1,3,2))
        # coord_distance_matrix = torch.abs(1-coord_distance_matrix)
        # coord_weights = coord_distance_matrix * coord_ints_matrix
        # coord_neg_matrix = torch.where(coord_ints_matrix<0,-2,0)
        # final_coord_weights = coord_weights + coord_neg_matrix
        # return coord_features_flatten.unsqueeze(-1)
        return coord_features.flatten(2).transpose(2,1)

def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
