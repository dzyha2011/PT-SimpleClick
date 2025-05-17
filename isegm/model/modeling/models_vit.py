import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict
from .pos_embed import interpolate_pos_embed

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        

    def forward(self, x):
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        return x

class Gated_Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Gated_Mlp, self).__init__()
        self.gated_mlp = nn.Sequential(
            nn.Linear(in_features+1,in_features+1),
            act_layer(),
            nn.Linear(in_features+1, 1),
            # norm_layer(1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Linear(in_features,in_features),
            # norm_layer(1),
            act_layer(),
            nn.Linear(in_features,1)
            # act_layer()
        )
        # self.reset_parameters()

    def forward(self, input_features, gating_features):
        alphas = self.gated_mlp(torch.cat([input_features,gating_features],dim=-1))
        input_features = input_features * (alphas + 1)
        input_features = self.fuse(input_features)
        return input_features
    
    # def reset_parameters(self):
    #     nn.init.xavier_normal_(self.weight)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

class WeightStream(nn.Module):
    def __init__(self,in_features):
        super(WeightStream,self).__init__()
        self.gated1 = Gated_Mlp(in_features)
        self.gated2 = Gated_Mlp(in_features)
        self.mlp1 = Mlp(1,in_features,1)
        self.mlp2 = Mlp(1,in_features,1)
    
    def forward(self,input_features,gating_features):
        gating_features = self.gated1(input_features[0],gating_features)
        gating_features = self.mlp1(gating_features)
        gating_features = self.gated2(input_features[1],gating_features)
        gating_features = self.mlp2(gating_features)
        affinity_matrix = torch.matmul(gating_features,gating_features.permute(0,2,1))
        return affinity_matrix


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    ''' Multi-head self-attention '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, additional_matrix=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if additional_matrix is not None:
            additional_matrix = additional_matrix.repeat(1,self.num_heads,1,1)
            attn = attn + additional_matrix
            attn[attn>1.0] = 1.0
            attn[attn<0.0] = 0.0

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False, attn_drop=0., 
        proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, 
            proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

    def forward(self, x):
        if isinstance(x,list):
            additional_matrix = x[1]
            x = x[0]
            x = x + self.attn(self.norm1(x),additional_matrix)
        else:
            x = x + self.attn(self.norm1(x))


        x = x + self.mlp(self.norm2(x))
        return x

class PriorEmbed(nn.Module):
    def __init__(self,img_size=(224,224),patch_size=(16,16),in_chans=3,embed_dim=768,norm_layer=None):
        super().__init__()
        self.in_chans=in_chans
        self.img_size = img_size
        # self.ks = img_size[0]//patch_size[0]
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans,in_chans,kernel_size=self.patch_size,stride=self.patch_size,groups=in_chans)
        self.norm = norm_layer(in_chans) if norm_layer else nn.Identity()
        self.mlp = Mlp(self.num_patches,out_features=self.embed_dim)
    
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2)
        x = self.norm(x)
        x = self.mlp(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H % self.img_size[0] == 0 and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # assert C == self.in_chans, \
        #     f"Input image chanel ({C}) doesn't match model ({self.in_chans})"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling 
    """    
    def __init__(self, img_size=(224,224), patch_size=(16, 16), in_chans=3, num_classes=1000, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, pos_drop_rate=0., attn_drop_rate=0., 
                 proj_drop_rate=0., norm_layer=None, act_layer=None, cls_feature_dim=None, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.ws = WeightStream(embed_dim)
        # self.linear_embedding = nn.Linear(768,embed_dim)
        self.linear_embedding = Mlp(in_features=768,out_features=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.prior_token = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 4, embed_dim)) # learnable positional embedding
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
              attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, norm_layer=norm_layer, 
              act_layer=act_layer)
            for _ in range(depth)])

        # self.conv1 = nn.Conv2d(self.embed_dim,1,1)
        # self.patch_embed_stage1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        #     embed_dim=embed_dim)

        # self.conv2 = nn.Conv2d(self.embed_dim,1,1)
        # self.patch_embed_stage2 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        #     embed_dim=embed_dim)

        self.fc_norm = norm_layer(embed_dim)

        # feature representation for classification
        if cls_feature_dim:
            self.num_features = cls_feature_dim
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, cls_feature_dim)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # classification head(s)
        self.head = nn.Linear(self.num_features, num_classes)

        self.init_weights()

    def init_weights_from_pretrained(self, pretrained_path):
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % pretrained_path)
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            else:
                checkpoint_model = checkpoint['model']

            # interpolate position embedding
            interpolate_pos_embed(self, checkpoint_model)

            # load pre-trained model
            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def init_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively similar to normal_(std=0.02) 
        # as the default cutoff in trunc_normal_(std=.02) is too big (-2., 2.)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.prior_token,std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def shuffle(self, x):
        """
        in: x (B, N, C)
        out: x_shuffle (B, N, C), ids_restore (B, N)
        """
        B, N, C = x.shape
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_shuffle = torch.gather(x, 1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

        return x_shuffle, ids_restore

    def unshuffle(self, x, ids_restore):
        B, N, C = x.shape
        x_unshuffle = torch.gather(x, 1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        return x_unshuffle

    def split(self, x):
        B, N, C = x.shape
        num_tokens_per_split = 224 * 224
        num_splits = max(1, N // num_tokens_per_split)
        out = []
        for i in range(num_splits):
            if i == num_splits - 1:
                out.append(x[:, i*num_tokens_per_split:])
                return out
            out.append(x[:, i*num_tokens_per_split:(i+1)*num_tokens_per_split])

    # window split for finetuning on larger size (the pretraining size should be 224 x 224)
    def patchify(self, x):
        """
        in: (B, N, C)
        out: (B*win_w*win_h, N//(win_w*win_h), C)
        """
        B, N, C = x.shape
        grid_h, grid_w = self.patch_embed.grid_size
        win_h_grid = 224 // self.patch_embed.patch_size[0]
        win_w_grid = 224 // self.patch_embed.patch_size[1]
        win_h, win_w = grid_h // win_h_grid, grid_w // win_w_grid
        x = x.view(B, win_h, grid_h // win_h, win_w, grid_w // win_w, C)
        x_patchified = x.permute((0, 1, 3, 2, 4, 5)).contiguous()
        x_patchified = x_patchified.view(B * win_h * win_w, grid_h * grid_w // (win_h * win_w), C)

        return x_patchified

    # recover the window split
    def unpatchify(self, x):
        """
        in: (B*win_h*win_w, N//(win_h*win_w), C)
        out: (B, N, C)
        """
        B, N, C = x.shape
        grid_h, grid_w = self.patch_embed.grid_size
        win_h_grid = 224 // self.patch_embed.patch_size[0]
        win_w_grid = 224 // self.patch_embed.patch_size[1]
        win_h, win_w = grid_h // win_h_grid, grid_w // win_w_grid
        x = x.view(B // (win_h * win_w), win_h, win_w, grid_h // win_h, grid_w // win_w, C)
        x = x.permute((0, 1, 3, 2, 4, 5)).contiguous().view(B // (win_h * win_w), win_h * win_w * N, C)

        return x

    def forward_backbone(self, x, additional_features=None,coord_weights=None, shuffle=False):
        B,C,H,W=x.shape
        xs = x.view(B,3,H//16,16,W//16,16)
        xs = xs.permute(0,2,4,1,3,5).flatten(3)
        xs = xs.permute(0,3,1,2).flatten(2).transpose(2,1)
        x = self.patch_embed(x)

        # x_prior_f = (xs*coord_weights[:,:,0].unsqueeze(-1)).transpose(2,1).mean(-1)
        # temp_f = xs*coord_weights[:,:,0].unsqueeze(-1)
        # temp_f = temp_f.transpose(2,1)
        # f_sign = torch.where(temp_f[:,0,:]!=0,1,0)
        # f_sign = f_sign.sum(-1).unsqueeze(-1)+0.0001
        # x_prior_f = temp_f.sum(-1)/f_sign

        # # x_prior_b = (xs*coord_weights[:,:,1].unsqueeze(-1)).transpose(2,1).mean(-1)
        # temp_b = xs*coord_weights[:,:,1].unsqueeze(-1)
        # temp_b = temp_b.transpose(2,1)
        # b_sign = torch.where(temp_b[:,0,:]!=0,1,0)
        # b_sign = b_sign.sum(-1).unsqueeze(-1)+0.0001
        # x_prior_b = temp_b.sum(-1)/b_sign
        # # x_prior_f,x_prior_b = x_prior_f/(x_prior_f+x_prior_b+0.0001),x_prior_b/(x_prior_f+x_prior_b+0.0001)
        # x_prior = torch.cat([x_prior_f.unsqueeze(1),x_prior_b.unsqueeze(1)],dim=1)

        # x_prior = self.linear_embedding(x_prior)
        
        if additional_features is not None:
            x += additional_features
            # x = torch.cat((additional_features,x),dim=1)

        if coord_weights is not None:
            prior_token = self.prior_token.expand(x.shape[0],-1,-1)
            x = torch.cat((prior_token, x), dim=1)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # x = self.pos_drop(x + self.pos_embed[:, 1:])
        x = self.pos_drop(x + self.pos_embed)
        num_blocks = len(self.blocks)
        assert num_blocks % 4 == 0
        # input_features = []

        if shuffle:
            for i in range(1, num_blocks + 1):
                x, ids_restore = self.shuffle(x)
                x_split = self.split(x)
                x_split = [self.blocks[i-1](x_split[j]) for j in range(len(x_split))]
                x = torch.cat(x_split, dim=1)
                x = self.unshuffle(x, ids_restore)
        else:
            num_blocks_per_group = 6 if num_blocks == 12 else num_blocks // 4
            is_patchified = False
            for i in range(1, num_blocks + 1):
                # if i % num_blocks_per_group:
                #     if not is_patchified:
                #         external_tokens = x[:,:4]
                #         x = self.patchify(x[:,4:])
                #         is_patchified = True
                #     else:
                #         pass # do nothing
                # else:
                #     x = self.unpatchify(x)
                #     x = torch.cat([external_tokens,x],1)
                #     ## enhancement (added)
                #     # x = [x,coord_weights]
                #     # input_features.append(x)
                #     is_patchified = False
                x = self.blocks[i-1](x)
        # affinity_matrix = self.ws(input_features,coord_weights)
        return x[:,2:4],x[:,4:]


    # def enhance(self, x, additional_features):
    #     B,N,C = x.shape
    #     grid_size = self.patch_embed.grid_size
    #     x = x.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)

        if self.global_pool:
            x = x[:, 1:].mean(dim=1) # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.fc_norm(x)
            x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        return x


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=160, depth=8, num_heads=4, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=(14,14), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model