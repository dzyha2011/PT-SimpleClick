import math
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed,PriorEmbed, Mlp
from .modeling.swin_transformer import SwinTransfomerSegHead
# from .modeling.deeplab_v3 import DeepLabV3Plus
from .group_vit import GroupingBlock

class FusionBlock(nn.Module):
    def __init__(self, in_dim=768, num_heads=8, bias=False, attn_drop=0., proj_drop=0.):
        super(FusionBlock,self).__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim ** -0.5
        self.wf = nn.Linear(in_features=in_dim,out_features=in_dim,bias=bias)
        self.wp1 = nn.Linear(in_features=in_dim,out_features=in_dim,bias=bias)
        self.wp2 = nn.Linear(in_features=in_dim,out_features=in_dim,bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features=in_dim,out_features=in_dim,bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self,x,p):
        # x: image features
        # p: prior features
        B,NX,C = x.shape
        _,NP,_ = p.shape
        wxq = self.wf(x).reshape(B,NX,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        wpk = self.wp1(p).reshape(B,NP,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        wpv = self.wp2(p).reshape(B,NP,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        attn = (wxq @ wpk.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ wpv).transpose(1,2).reshape(B, NX, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class ConvNet(nn.Module):
#     def __init__(self,inchannel=2048):
#         super(ConvNet).__init__()


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        # self.prior_embed = PriorEmbed(
        #     img_size=backbone_params['img_size'],
        #     patch_size=backbone_params['patch_size'],
        #     in_chans=3 if self.with_prev_mask else 2
        # )
        self.mlp = Mlp(in_features=backbone_params['embed_dim'])
        self.fuse = FusionBlock(in_dim=backbone_params['embed_dim'],num_heads=backbone_params['num_heads'])

        # self.group = GroupingBlock(dim=backbone_params['embed_dim'],
        #     out_dim=backbone_params['embed_dim'],
        #     num_heads=backbone_params['num_heads'],
        #     num_group_token=784,
        #     num_output_group=784,
        #     norm_layer=nn.LayerNorm,
        #     hard=True,
        #     gumbel=True)
        # self.resnet_backbone = DeepLabV3Plus()
        # self.c1 = nn.Conv2d(in_channels=2048,out_channels=backbone_params['embed_dim'],kernel_size=1,stride=1)
        # self.c1 = nn.Sequential(
        #     nn.Conv2d(in_channels=256,out_channels=head_params['in_channels'][0],kernel_size=1,stride=1),
        #     nn.GroupNorm(head_params['in_channels'][0],head_params['in_channels'][0]),
        #     nn.ReLU()
        # )
        # self.c2 = nn.Sequential(
        #     nn.Conv2d(in_channels=512,out_channels=head_params['in_channels'][1],kernel_size=1,stride=1),
        #     nn.GroupNorm(head_params['in_channels'][1],head_params['in_channels'][1]),
        #     nn.ReLU()
        # )
        # self.c3 = nn.Sequential(
        #     nn.Conv2d(in_channels=1024,out_channels=head_params['in_channels'][2],kernel_size=1,stride=1),
        #     nn.GroupNorm(head_params['in_channels'][2],head_params['in_channels'][2]),
        #     nn.ReLU()
        # )
        # self.c4 = nn.Sequential(
        #     nn.Conv2d(in_channels=2048,out_channels=head_params['in_channels'][3],kernel_size=1,stride=1),
        #     nn.GroupNorm(head_params['in_channels'][3],head_params['in_channels'][3]),
        #     nn.ReLU()
        # )
        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image, coord_features=None,coord_weights=None):
        # layer1,layer4 = external_input[0],external_input[1]
        # layer1,layer2,layer3,layer4 = self.c1(external_input[0]),self.c2(external_input[1]),self.c3(external_input[2]),self.c4(external_input[3])
        # external_input = [layer1,layer2,layer3,layer4]
        coord_features = self.patch_embed_coords(coord_features)
        # coord_features = self.prior_embed(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, coord_weights,self.random_split)

        instances_aux = backbone_features[0]
        backbone_features = backbone_features[1]
        instances_aux = self.mlp(instances_aux)
        backbone_features += self.fuse(backbone_features,instances_aux)
        # backbone_features,_ = self.group(instances_aux,backbone_features)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        # backbone_features += layer4
        multi_scale_features = self.neck(backbone_features)

        return {'instances': self.head(multi_scale_features), 'instances_aux': instances_aux}
