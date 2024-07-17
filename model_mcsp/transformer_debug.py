#cosine aware masking
#将重建目标换成了余弦相似度，并改变了相应的encoder、decoder的代码，以使真实值和目标值的维度相匹配




import torch
import torch.nn as nn
import torch.nn.functional as F#函数库，如激活函数、损失函数、池化、卷积等
import math#数学函数，如三角函数、指数函数、对数函数等
import warnings#导入警告模块，用于处理警告信息（显示、忽略或记录到文件中等）
from .drop import DropPath
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # in_features is the number of input features.
        # hidden_features and out_features default to in_features if not specified.
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # self.fc1 is the first fully connected layer that reduces the dimensions from in_features to hidden_features.
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # act_layer defines the activation function, defaulting to GELU. self.act instantiates this activation.
        self.act = act_layer()
        
        # self.fc2 is the second fully connected layer that reduces the dimensions from hidden_features to out_features
        # which is usually the number of classes.
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # self.drop defines the dropout rate to regularize the network. nn.Dropout is instantiated with this rate
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    # The constructor method to initialize the layers and parameters needed for Attention module.
    # proj_drop representing the dropout probability applied after the output projection in the attention mechanism
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # Set the number of heads for multi-head attention
        self.num_heads = num_heads
        
        # Calculate the dimension for each head
        head_dim = dim // num_heads
        
        
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # Set the scaling factor for query-key dot product
        self.scale = head_dim ** -0.5
        
        # Initialize the Dropout layer for attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Initialize the linear layer to project output back to original dimension dim
        self.proj = nn.Linear(dim, dim)
        
        # Initialize the linear layer to map input to queries, keys, values
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        """The first parameter dim defines the input dimension.
        The input to self.qkv will be the output from the previous layer, which has dimension dim.
        The second parameter dim * 3 defines the output dimension.
        In attention, the input is projected to queries (Q), keys (K) and values (V).
        So the output dimension needs to be 3 times the input dimension, to separate out Q, K, V.
        bias=qkv_bias allows adding a bias parameter or not. If True, a bias will be added.
        So in summary, this linear layer projects the input (of dim size) to an output that is 3 * dim size,
        to separate it into Q, K, V projections.
        """
        
        # Initialize the dropout layer for output and assigned to self.proj_drop.
        # It will be applied after the output projection in the forward pass.
        self.proj_drop = nn.Dropout(proj_drop)
        
    # x is the input tensor that will be passed to the module.
    # seqlen=1 is defining a default value for the sequence length, if it's not passed.
    def forward(self, x, seqlen=1):
        # getting the shape/dimensions of the input tensor x
        # B refers to the batch size, which is the number of examples in the batch
        # N refers to the number of tokens/sequence length/sequence dims in each example
        # C represents the number of channels/features in each token
        B, N, C = x.shape
        
        # It maps the input x to the query, key and value representations and
        # reshape reorganizes the dimensions to prepare for multi-head attention.
        # B,N are batch and sequence dims
        # 3 separates out q,k,v
        # num_heads splits C into multiple heads and C//num_heads is dim of each head
        # permute(2, 0, 3, 1, 4) rearranges the dimensions of the tensor. 
        # The shape of qkv after permute is: (3, B, self.num_heads, N, C // self.num_heads).
        # This is done to prepare the tensor for the subsequent attention calculation.
        # Specifically, it moves the dimensions related to q, k, and v to the front, 
        # aligning with the standard format used in self-attention mechanisms.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # make torchscript happy (cannot use tensor as tuple)
        # This line unpacks the three components of qkv into separate tensors
        # 'q', 'k', and 'v' have the same shape (B, self.num_heads, N, C // self.num_heads)
        # In other words, the shape is (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        #Calculate multi-head attention
        x = self.forward_attention(q, k, v)

        #Project attention output
        x = self.proj(x)
        
        #Add Dropout
        x = self.proj_drop(x)
        
        #Return output of Attention module
        return x

    # Defines the core attention calculation
    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        
        # transpose(-2, -1) is used to swap the last two dimensions of the k tensor
        # To achieve the transposing of the k tensor/matrix.
        # It helps align the dimensions correctly for the multiplication.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # The softmax function is applied along the last dimension (dim=-1) of the attn tensor.
        # This operation normalizes the attention scores across the sequence length.
        attn = attn.softmax(dim=-1)
        
        # The result is then passed through the dropout layer
        attn = self.attn_drop(attn)
        
        # The attention scores (attn) are used to compute a weighted sum of the values (v).
        x = attn @ v
        
        # The result (x) is transposed using x.transpose(1, 2), swapping the second and third dimensions
        # This is done to align the dimensions for concatenation across attention heads.
        # The result is a tensor with the same shape as the input query tensor
        # but with values rearranged based on attention.
        x = x.transpose(1,2).reshape(B, N, C*self.num_heads)
        
        # the output of the self-attention mechanism
        return x

class Block(nn.Module):
    # This is the constructor of a class responsible for initializing instances of the class
    # dim: The dimension of the input features. 
    # mlp_ratio controls the size of the hidden layers in the MLP relative to the input feature dimension
    # mlp_out_ratio controls the size of the output layer in the MLP relative to the input feature dimension.
    # qkv_bias: A boolean indicating whether to enable bias in the query, key, and value computations of the attention mechanism.
    # qk_scale: A scaling factor for scaling the attention scores.
    # drop_path: A value used to implement stochastic depth. If greater than 0, it indicates random dropping of paths during training for regularization.
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # assert 'stage' in st_mode
        
        # Creating an instance of a normalization layer (norm_layer) and assigns it to the attribute self.norm1. 
        # The normalization layer is applied to the input features before the attention mechanism.
        self.norm1 = norm_layer(dim)
        
        # Creating an instance of an attention mechanism and assigns it to the attribute self.attn
        # proj_drop representing the dropout probability applied after the output projection in the attention mechanism
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # nn.Identity() is a PyTorch module representing the identity operation,
        # It does not modify the input and simply returns it unchanged
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        
        # By multiplying the input dimension by mlp_ratio, the dimension of the hidden layer is obtained
        # By multiplying the input dimension by mlp_out_ratio, the dimension of the output layer is obtained.
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    """x: The input tensor to the block.
    seqlen: A sequence length parameter (default is 1).
    The forward pass consists of:
    Applying attention to the normalized input features (self.attn(self.norm1(x), seqlen)).
    Adding the drop path regularization to the attention output and adding it to the input (x + self.drop_path(...)).
    Applying normalization to the combined output.
    Applying an MLP to the normalized output.
    Adding the drop path regularization to the MLP output and adding it to the normalized output.
    Returning the final output.
    """
    def forward(self, x, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SkeleEmbed(nn.Module):
    # Transforming input skeleton data into a format suitable for processing by a transformer-based model.

    def __init__(
        self,
        dim_in=3,
        dim_feat=256,#论文中的Ce
        num_frames=120,
        num_joints=25,
        patch_size=1, # each patch contains only one joint
        t_patch_size=4, # each patch contains four consecutive frames along the time dimension
    ):
        super().__init__()
        assert num_frames % t_patch_size == 0
        num_patches = (
            (num_joints // patch_size) * (num_frames // t_patch_size)#关节点个数
        )
        self.input_size = (
            num_frames // t_patch_size,
            num_joints // patch_size
        )#(Te, V)
        print(
            f"num_joints {num_joints} patch_size {patch_size} num_frames {num_frames} t_patch_size {t_patch_size}"
        )

        self.num_joints = num_joints
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = num_joints // patch_size
        self.t_grid_size = num_frames // t_patch_size

        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)#原论文中的骨架嵌入/投影方式
        """self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(dim_feat),
            nn.LeakyReLU(0.1))#STTFormer中的投影方式"""

    def forward(self, x):
        _, T, V, C = x.shape
        x = torch.einsum("ntsc->ncts", x)  # [N, C, T, V]
        
        assert (
            V == self.num_joints
        ), f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert (
            T == self.num_frames
        ), f"Input skeleton length ({T}) doesn't match model ({self.num_frames})."
        
        x = self.proj(x)#线性投影[N, dim_feat==256, Te==TP==30, V==VP==25]
        x = torch.einsum("ncts->ntsc", x)  # [N, Te==TP==30, V==VP==25, dim_feat==256]   torch.Size([2, 30, 25, 256])
        return x

class Transformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256,
                 depth=5, decoder_depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, norm_skes_loss=False):
        super().__init__()
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.norm_skes_loss = norm_skes_loss
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_dim_feat)

        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, decoder_dim_feat))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, decoder_dim_feat))
        trunc_normal_(self.decoder_temp_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

        self.decoder_pred = nn.Linear(
            decoder_dim_feat,
            #t_patch_size * patch_size * dim_in,
            t_patch_size * patch_size,
            bias=True
        ) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Initialize weights
        self.apply(self._init_weights)
        # --------------------------------------------------------------------------


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cosine_aware_random_masking(self, x, x_orig, mask_ratio, tau): #余弦感知随机mask
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [NM, L, D], sequence，注：我推出的输入到该函数中的x的形状为[N, T*V, C],#torch.Size([2, 750, 256])
        x_orig: patchified original skeleton sequence,注：我推出的输入的x_orig的形状是[N, TP, VP, C],# torch.Size([2, 30, 25, 12])
        tau: τ
        """
        NM, L, D = x.shape
        _, TP, VP, _ = x_orig.shape
        x_orig_shape = x_orig.shape
        
        len_keep = int(L * (1 - mask_ratio))#（输入样本中）保留的关节点的个数

        """x_orig_motion = torch.zeros_like(x_orig)
        x_orig_motion[:, 1:, :, :] = torch.abs(x_orig[:, 1:, :, :] - x_orig[:, :-1, :, :])
        x_orig_motion[:, 0, :, :] = x_orig_motion[:, 1, :, :] #可以换成0填充，即x_orig_motion[:, 0, :, :] = 0,或者改成和extract_motion中一样，在最后一个维度进行填充"""
        
        # 创建形状为 [NM, T, V] 的零张量
        x_orig_cosine_similarity = torch.zeros(x_orig_shape[:-1], dtype=x_orig.dtype, device=x_orig.device)# torch.Size([2, 30, 25])
        
        # 计算同一帧中相邻索引位置关节点之间的余弦相似度
        # 取前后两个索引位置的关节点向量
        u = x_orig[:, :, :-1]
        v = x_orig[:, :, 1:]
        
        # 计算余弦相似度
        x_orig_cosine_similarity[:, :, :-1] = F.cosine_similarity(u, v, dim=-1) #F.cosine_similarity(u, v, dim=-1)的形状是[NM, TP, VP-1]
        x_orig_cosine_similarity[:, :, -1:] = 0 # 最后一个关节点的余弦相似度设置为0，# NM, TP, VP
        
        
        #计算I
        x_orig_cosine_similarity = x_orig_cosine_similarity.reshape(NM, L)
        
        # π == Softmax(I/τ )
        # Scaling logits by tau before softmax
        x_orig_cosine_similarity /= tau
       
        x_orig_cosine_similarity_prob = F.softmax(x_orig_cosine_similarity, dim=-1)
        
        u = torch.rand(NM, L, device=x.device)  # Uniformly distributed random numbers, 取值范围[0,1]
        g = -torch.log(-torch.log(u + 1e-10) +1e-10)  # Gumbel noise
        noise = torch.log(x_orig_cosine_similarity_prob + 1e-10) + g  # Add Gumbel noise to log probabilities

        # sort noise for each sample
        ids_shuffle = torch.argsort( #返回张量中元素值（从大到小）排序后的索引
            noise, dim=1, descending=True
        )  # descending: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)#用于恢复原始张量的位置索引顺序

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # index=ids_keep.unsqueeze(-1).repeat(1, 1, D):
        # converts the two-dimensional index into three dimensions to match the shape of the input x
        # ids_keep.unsqueeze(-1): adds a dimension of size 1 to the existing 2D tensor ids_keep. The shape becomes (batch, len_keep, 1)
        # repeat(1, 1, D): Repeat the first dimension 1 time, the second dimension 1 time, and the third dimension D times, this expands the shape to (batch, len_keep, D)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([NM, L], device=x.device)
        mask[:, :len_keep] = 0#mask为0表示没mask，mask为1该元素被mask
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # x_masked: The sampled subsequence(tensor)
        # mask: The sampling mark for each frame(tensor, 0 or 1)
        # ids_restore: The indexes for order restoration(tensor)
        # ids_keep: The original indexes of kept frames(tensor)
        return x_masked, mask, ids_restore, ids_keep

    def random_masking(self, x, mask_ratio):

        #Perform per-sample random masking by per-sample shuffling.
        #Per-sample shuffling is done by argsort random noise.
        #x: [N, L, D], sequence
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, motion_aware_tau):# 传入encoder时x形状：[NM, T, V, C] torch.Size([2, 120, 25, 3])
        x_orig = self.patchify(x) #x_orig保留嵌入之前的信息，后面mask会用到。形状[N, (T/1)*V, u*p*C]   #torch.Size([2, 3000, 3])
        # embed skeletons
        x = self.joints_embed(x)# 经过embedding后x的形状，[N, TP, VP, dim_feat=256]  torch.Size([2, 120, 25, 256])

        NM, TP, VP, _ = x.shape# N, TP, VP, _

        # add pos & temp embed
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :] #torch.Size([2, 30, 25, 256])

        # masking: length -> length * mask_ratio
        x = x.reshape(NM, TP * VP, -1) #torch.Size([2, 750, 256])
        
        if motion_aware_tau > 0:
            x_orig = x_orig.reshape(shape=(NM, TP, VP, -1))# torch.Size([2, 30, 25, 12])
            x, mask, ids_restore, _ = self.cosine_aware_random_masking(x, x_orig, mask_ratio, motion_aware_tau)#x形状：torch.Size([2, 149, 256]). 其中149=750*mask_ratio(0.8)，表示保留的关节个数。
        else:   
            x, mask, ids_restore, _ = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for idx, blk in enumerate(self.blocks): # torch.Size([2, 149, 256])
            x = blk(x)

        x = self.norm(x)#从encoder中输出的张量的形状：torch.Size([2, 149, 256])
 
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):# 传入decoder时张量的形状：torch.Size([2, 149, 256])
        NM = x.shape[0] #2
        TP = self.joints_embed.t_grid_size #30
        VP = self.joints_embed.grid_size #25

        # embed tokens
        x = self.decoder_embed(x) # self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        C = x.shape[-1]

        # append intra mask tokens to sequence
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        # the dimension of x is :(batch, len_keep, D)
        mask_tokens = self.mask_token.repeat(NM, TP * VP - x.shape[1], 1)#torch.Size([2, 601, 256])
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token torch.Size([2, 750, 256])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle  torch.Size([2, 750, 256])
        x = x_.view([NM, TP, VP, C])# Reshape x_ into a four-dimensional tensor  torch.Size([2, 30, 25, 256])

        # add pos & temp embed
        x = x + self.decoder_pos_embed[:, :, :VP, :] + self.decoder_temp_embed[:, :TP, :, :]  # NM, TP, VP, C  torch.Size([2, 30, 25, 256])
        
        # apply Transformer blocks
        x = x.reshape(NM, TP * VP, C) #torch.Size([2, 750, 256])

        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x)  #torch.Size([2, 750, 256])
        
        
        x = self.decoder_norm(x)#torch.Size([2, 750, 256])
        
        # predictor projection
        #a motion prediction head , which takes decoded features as input and predicts the temporal motion of the input skeleton sequence.注意这里预测头输出的形状一定要匹配原始的motion形状
        x = self.decoder_pred(x) #从forward_decoder中输出的张量的形状：torch.Size([2, 750, 4])

        return x
    
    def patchify(self, imgs):
        """
        encoder中：
        传入的imgs: [NM, T, V, C] 
        返回的x: (N, L, t_patch_size * patch_size * C)
        
        forward_loss中：
        传入的imgs: [NM, T, V, 1] 
        返回的x: (N, L, t_patch_size * patch_size * 1)
        
        """

        NM, T, V, C = imgs.shape

        p = 1
        u = 4
        assert V % p == 0 and T % u == 0
        VP = V // p # It calculates the number of patches in the spatial dimensions
        TP = T // u # It calculates the number of patches in the temporal dimensions

        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))#torch.Size([2, 30, 4, 25, 1, 1])
        x = torch.einsum("ntuvpc->ntvupc", x)
        x = x.reshape(shape=(NM, TP * VP, u * p * C))

        return x

    """def patchify(self, imgs):
        '''
        encoder中：
        传入的imgs: [NM, T, V, C] 
        返回的x: (N, L, t_patch_size * patch_size * C)
        
        forward_loss中：
        传入的imgs: [NM, T, V, 1] 
        返回的x: (N, L, t_patch_size * patch_size * 1)
        
        '''

        NM, T, V, C = imgs.shape

        p = 1
        u = 4
        assert V % p == 0 and T % u == 0
        VP = V // p # It calculates the number of patches in the spatial dimensions
        TP = T // u # It calculates the number of patches in the temporal dimensions

        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))#torch.Size([2, 30, 4, 25, 1, 3])
        x = torch.einsum("ntuvpc->ntvupc", x)#torch.Size([2, 30, 25, 4, 1, 3]
        x = x.reshape(shape=(NM, TP * VP, u * p * C))#torch.Size([2, 750, 12])

        return x"""
    
        
    """#原论文
    def extract_motion(self, x, motion_stride=1):
        '''
        imgs: [NM, T, V, 3]
        '''
        # generate motion
        
        # Creating a zero tensor x_motion with the same shape as the input x to store the extracted motion information.
        x_motion = torch.zeros_like(x)
        
        
        x_motion[:, :-motion_stride, :, :] = x[:, motion_stride:, :, :] - x[:, :-motion_stride, :, :]
        x_motion[:, -motion_stride:, :, :] = 0
        return x_motion"""
        
    #重建帧内关节点之间的余弦相似度
    def extract_cos(self, x, motion_stride=1):
        '''
        imgs: [NM, T, V, C]
        x_cos: [NM, T, V, 1]
        '''
        # generate cosine value
        
        # 获取张量 x 的形状
        x_shape = x.shape
        # 创建形状为 [NM, T, V] 的零张量
        x_cos = torch.zeros(x_shape[:-1] + (1,), dtype=x.dtype, device=x.device)
        
        # 计算同一帧中相邻索引位置关节点之间的余弦相似度
        # 取前后两个索引位置的关节点向量
        u = x[:, :, :-motion_stride, :]
        v = x[:, :, motion_stride:, :]
        
        # 计算余弦相似度
        x_cos[:, :, :-motion_stride, :] = F.cosine_similarity(u, v, dim=-1).unsqueeze(-1) #F.cosine_similarity(u, v, dim=-1)的形状是[NM, T, V-1, 1]
        x_cos[:, :, -motion_stride:, :] = 0 # 最后一个关节点的余弦相似度设置为0, # [NM, T, V, 1]
        
        return x_cos
    
    """#重建原关节
    def extract_motion(self, x):
        '''
        imgs: [NM, T, V, 3]
        '''
        # generate motion
        
        # Creating a zero tensor x_motion with the same shape as the input x to store the extracted motion information.
        x_motion = x
        
        
        return x_motion"""

    def forward_loss(self, imgs, pred, mask):#这里的imgs相当于传入的x_cos,形状为#torch.Size([2, 120, 25, 1]), pred的形状：torch.Size([2, 750, 4])
        """
        imgs: [NM, T, V, 1]
        pred: [NM, TP * VP, t_patch_size * patch_size * 1]
        mask: [NM, TP * VP], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # [NM, TP * VP, t_patch_size*patch_size*1], [2, 750, 4]

        # Normalization 
        if self.norm_skes_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [NM, TP * VP], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed joints

        return loss

    def forward(self, x, mask_ratio=0.80, motion_stride=1, motion_aware_tau=0.75, **kwargs):
        N, C, T, V, M = x.shape#[2, 3, 120, 25, 1]
        

        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)#torch.Size([2, 120, 25, 3])

        x_cos = self.extract_cos(x, motion_stride)#torch.Size([2, 120, 25, 1])#先提取运动信息，作为真实值（目标）

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, motion_aware_tau)#latentshape: torch.Size([2, 149, 256])   #maskshape: torch.Size([2, 750])
        pred = self.forward_decoder(latent, ids_restore)  # [NM, TP * VP, C] torch.Size([2, 750, 4])

        loss = self.forward_loss(x_cos, pred, mask)
        
        return loss, pred, mask

    