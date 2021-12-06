
# coding: utf-8

# ### Group#: 28 - Liver UNET-R
# ### Couse: CISC867
# ### Team: Ramtin Mojtahedi - Vignesh Rao

# In[1]:


#define CUDA version
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
torch.cuda.is_available()
torch.version.cuda


# ### Install Libraries and Packages
# 

# In[11]:


# sing_strategy('file_system')#multi-processing 
import matplotlib.pyplot as plt#plot
import numpy as np#import numpy
from tqdm import tqdm#display progression bar
import glob#for finding image files
import os
get_ipython().run_line_magic('matplotlib', 'inline')
#-------------------------------------------------------MONAI related libraries
from monai.losses import DiceCELoss#import dice loss
from monai.inferers import sliding_window_inference#import sliding window for inferencing
from monai.utils import first, set_determinism #set the first and determinism(randomness)
from monai.apps import DecathlonDataset#import MSD data set downloader
#data augmentation
from monai.transforms import (
    AsDiscrete,#make thresholding to make discrete part of the object
    Compose,#make transformation
    CropForegroundd,#removes all zero borders to focus on the valid body area of the images and labels.
    RandCropByPosNegLabeld, #randomly crop patch samples from big image based on pos(foreground) / neg(background) ratio
    AddChanneld,#as the original data doesn't have channel dim, add 1 dim to construct "channel first" shape.
    ToTensord, #making as a tensor to the device
    LoadImaged,#loading the image
    Orientationd,# unifies the data orientation based on the affine matrix
    RandFlipd,#flip the image
    #RandCropByPosNegLabeld, #random crop
    RandShiftIntensityd,#shifting intensity
    ScaleIntensityRanged,#scaling intensity to [0-1]
    Spacingd,#spatial transfomr
    RandRotate90d,#rotate 90 degree
    RandSpatialCropd,#random spatial crop
    RandScaleIntensityd,#random scaling intensity
    EnsureTyped,#ensure type dimension
    EnsureType,
    SpatialCrop,
    RandAffined,
    EnsureChannelFirstd, #ensure that the first dimension and channel is available
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ResizeWithPadOrCrop,
    NormalizeIntensityd #normalize intensity
)

from monai.networks.layers import Norm#import MONAI normalization
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, compute_hausdorff_distance, compute_average_surface_distance #import MONAI metrics
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch#making decollah batch and dataloader
from monai.losses import DiceLoss
#import MONAI data loader and related loibraries
from monai.data import (
    DataLoader,#loading data
    CacheDataset,#By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline. If the requested data is not in the cache, all transforms will run normally
    load_decathlon_datalist,#oad image/label paths of decathlon challenge from JSON file
    decollate_batch,#simplify the post-processing transforms and enable flexible operations on a batch of model outputs.
)


# ### Data Augmentation

# In[12]:


roi_size=192
z_size = 32
sliding_batch_size = 6
#define training transformation
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),#loading images
        AddChanneld(keys=["image", "label"]),#adding channel images to device using label and IDs
        Spacingd(#define bilinear interpolation voxel spacing
            keys=["image", "label"],
            # pixdim=(0.7652890625, 0.7652890625, 1.0),#based on the median    0.6646484375
            pixdim=(0.6646484375, 0.6646484375, 2),#set the spacing
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),#change the orientation
#         ScaleIntensityRanged(#change the scale intensity
#             keys=["image"],
#             a_min=-57,
#             a_max=164,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
# #         ),
#         ScaleIntensityRanged(#change the scale intensity
#             keys=["image"],
#             a_min=-150,
#             a_max=250,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),

        # RandSpatialCropd(keys=["image", "label"], roi_size=[roi_size, roi_size, roi_size], random_size=False),
        CropForegroundd(keys=["image", "label"], source_key="image"),#crop the foreground
        # ResizeWithPadOrCrop(spatial_size=(64, 64, 64)),
        RandCropByPosNegLabeld(#random crop based on the ratio of neg(background) and positive(foreground)
            keys=["image", "label"],
            label_key="label",
            spatial_size=(roi_size, roi_size, z_size),#transformer input size based on the ViT paper
            pos=1,#define as default
            neg=1,#define as default
            num_samples=6,#number of samples in each crop
            image_key="image",
            image_threshold=0,
            allow_smaller = False,
            allow_missing_keys=False
        ),
        RandFlipd(#random flip x-axis
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(#random flip y-axis
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(#random flip z-axis
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(#random rotate withnumber of random rotations of three
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(#random shifting intensity
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
        ToTensord(keys=["image", "label"]),#set the tensor to the device
    ]
)
#define preprocessing on the validation data set
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),#loading images
        AddChanneld(keys=["image", "label"]),#adding channesto the device
        Spacingd(
            keys=["image", "label"],
            # pixdim=(0.7652890625, 0.7652890625, 1.0),#adding pixel dimension   
            pixdim=(0.6646484375, 0.6646484375, 2),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"), #3D orientation
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
        # ScaleIntensityRanged(#scale intensity
        #     keys=["image"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True
        # ),
        # ResizeWithPadOrCrop(spatial_size=(64, 64, 64)),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)


# In[13]:


#load images
root_dir = os.getcwd()
train_images = sorted(
    glob.glob(os.path.join(root_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(# metric_values_HD95 = []

    glob.glob(os.path.join(root_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-26], data_dicts[-26:]
# bs_size = 2


# In[15]:


import monai,os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=4)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True)


# ### Define and Train Model

# In[ ]:


import torch.nn as nn # define nn architechture 
from monai.networks.layers import get_norm_layer # import normalization layer from MONAI - similar to Pytorch nn.norm
from monai.networks.blocks.dynunet_block import get_conv_layer# import conv3D from MONAI

#this class is for those blocks that consis of basic conv
class Conv_Encod(nn.Module):
    '''
    This class is used for Conv blocks in the encoder side of the network.   
    '''
    def __init__(
        self,
        size_Conv: int,
        input_channels: int, # number of input channels, which is euqal to 3
        output_channels: int, # number of output channel, which is euqal to 3
        kernel_size: int, # kernel size. In the ViT paper defined as 3
        stride: int = 1, #default stride value
        norm_name: str = 'batch', #layer normalization type based on the model's graph
    ):
        
        super().__init__() #initialize the class
        #perform 3D conv
#         self.convolution1 = nn.Conv3d( 
#             input_channels,
#             output_channels,
#             kernel_size=3,
#             stride=1,
#             padding=0
#         )
 
    #calculate convolution forthe three block 
        self.convolution1 = get_conv_layer(3, input_channels, output_channels, kernel_size=kernel_size, stride=1, conv_only=True)
        self.convolution2 = get_conv_layer(3, output_channels, output_channels, kernel_size=kernel_size, stride=1, conv_only=True)   
        self.convolution3 = get_conv_layer(3, input_channels, output_channels, kernel_size=kernel_size, stride=1, conv_only=True)           
        
        #defien the 
        self.relu_act = nn.ReLU(inplace=False) #default Pytorch Reluas shown in the model's graph
        self.contraction = input_channels != output_channels #find whether it is for contraction or expansion side of network
        stride_status = np.atleast_1d(stride) #stride value for contraction or expansion(down or up sampling)
        if not np.all(stride_status == 1):
            self.contraction = True #it is in the contraction(downsampling) side
        self.normalization1 = get_norm_layer(norm_name, size_Conv, output_channels)#layer normalization
        self.normalization2 = get_norm_layer(norm_name, size_Conv, output_channels)#layer normalization
        self.normalization3 = get_norm_layer(norm_name, size_Conv, output_channels)#layer normalization

        #define network structure
    def forward(self, input_value):
        res_value = input_value
        output = self.convolution1(input_value) #calculate first conv
        output = self.normalization1(output) #layer normalization
        output = self.relu_act(output) #giving to the activation function
        output = self.convolution2(output) ##calculate second conv
        output = self.normalization2(output) #layer normalization
        if self.contraction:
            res_value = self.convolution3(res_value) #calculate res value
            res_value = self.normalization3(res_value)#normaliza res value
        output += res_value 
        output = self.relu_act(output) #give the value to the activation function and output
        return output #return the encoder value


# In[ ]:


from monai.networks.blocks.convolutions import Convolution #import convolution and transpose convolution function similar to Pytorch's CONV and ConvTranspose
from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer #import the Res block for the UNET develped by "Automated Design of Deep Learning Methods for Biomedical Image Segmentation"

# this class is for blocks that has conv and deconv for encoders values 
class Conv_DecConv_Encod(nn.Module):
    '''
    This class is used for the cond-deconv blocks of the encoder side of the network.
    '''
    def __init__(
        self,
        size_Conv: int, #spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int,#number of channels in the output image
        number_layer: int, #number of Conv layers
        kernel_size: Union[Sequence[int], int],# based on the paper's suggestion
        stride: Union[Sequence[int], int],#defaultstride
        upsample_stride: Union[Sequence[int], int], # Based on the model's graph
        norm_name: str = 'batch'#layer normalization type based on the model's graph
                 ):
        
        super().__init__()#initialize the class

        # define the transpose conv
        self.transp_conv_layer = get_conv_layer(size_Conv, input_channels, output_channels, upsample_stride, stride, conv_only=True, is_transposed=True) 
        #list two modules to make them as a block
        self.tr_blocks = nn.ModuleList
        (
         [nn.Sequential(
        get_conv_layer(size_Conv,output_channels,output_channels,upsample_stride,upsample_stride,conv_only=True,is_transposed=True),
        UnetResBlock(spatial_dims=size_Conv, in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,stride=stride,norm_name=norm_name))
        for i in range(number_layer) # 12 trasnformers
         ] 
        )
    #network structure
    def forward(self, input_x):
        input_x = self.transp_conv_layer(input_x)# deconv the value
        for tr_blk in self.tr_blocks:
            input_x = tr_blk(input_x)
        return input_x


# In[ ]:


from monai.networks.blocks import UnetResBlock # this is the residual block basedon the U-Net architecture (2015)
#This class is used for decode blocks of the network
class decoder_expansion_blk(nn.Module):
    """
    This class is for the conv-deconv blocks in the decoder side of the network.
    """
    def __init__(
        self,
        size_Conv: int,#spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int,#number of channels in the output image
        kernel_size: Union[Sequence[int], int],#based on the paper's suggestion
        upsample_stride: Union[Sequence[int], int],# Based on the model's graph
        norm_name: str = 'batch'#layer normalization type based on the model's graph
    ):
    
        super().__init__()#initialize the class
        
        self.transp_conv_layer = get_conv_layer(size_Conv, input_channels, output_channels, kernel_size=upsample_stride, conv_only=True,is_transposed=True) 
        self.conv_block = UnetResBlock(size_Conv,output_channels+output_channels,output_channels,kernel_size,stride=1,norm_name=norm_name)
    #define architecture - number of skip connections is equal to channels
    def forward(self, input_value, skip_connection):
        output_decoder = self.transp_conv_layer(input_value)# deconv
        output_decoder = torch.cat((output, skip_connection), dim=1)
        output_decoder = self.conv_block(output_decoder)
        return output_decoder


# In[ ]:


# # This class is for the output of the U-Net architecture
class output(nn.Module):
    #initialize class
    def __init__(
        self, 
        size_Conv: int,#spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int,#number of channels in the output image
    ):
        super().__init__()#initialize the class
        self.conv_output = get_conv_layer(size_Conv, input_channels, output_channels, kernel_size=1, stride=1, bias=True, conv_only=True)
     #network architecture
    def forward(self, input_value):
        return self.conv_output(input_value)


# In[ ]:


#This class if for linear projection of the input features
class linear_projection(nn.Module):
    def __init__(
        self, 
        feature: int, #linear projection of the features
        hidden_size: int, #number of hidden layers
        feature_size: int #size of the features
        ):
        super().__init__()#initialize the class
        #new_projection
        new_projection = (feature.size(0), *feature_size, hidden_size)
        feature = features_projection.view(new_projection)
        new_projection = (0, len(feature.shape) - 1) + tuple(d + 1 for d in range(len(feature_size)))
        feature = feature.permute(new_projection).contiguous()
        return feature


# In[ ]:


# from monai.networks.nets.vit import ViT# import ViT architecture from MONAI
# '''
# UNETR is is similar to the popular CNN-based architecture known as UNET. 
# The proposed architecture add transformer to the decoder section. 
# The transfromer and its idea is achieved from the paper "An image is worth 16x16 words: Transformers for image recognition at scale.
# The used transformer called ViT-B16 with the parameters of L = 12 layers, an embedding size of K = 768 and patch size of 16*16*16 that is implemented in MONAI
# The used ViT parameters introduced in MONAI are as follows:

#         in_channels (int) – dimension of input channels.

#         img_size (Union[Sequence[int], int]) – dimension of input image.

#         patch_size (Union[Sequence[int], int]) – dimension of patch size.

#         hidden_size (int) – dimension of hidden layer.

#         mlp_dim (int) – dimension of feedforward layer.

#         num_layers (int) – number of transformer blocks.

#         num_heads (int) – number of attention heads.

#         pos_embed (str) – position embedding layer type.

#         classification (bool) – bool argument to determine if classification is used.

#         dropout_rate (float) – faction of the input units to drop.
# '''
# # Define the imported transformers parameters based on the selected paper and its original paper
transformer_ViT = ViT(
    in_channels = 3, # this is for Spleen data 
    img_size = (16, 16, 16),# patch size images 
    patch_size = (16, 16, 16), # Define the Patch size as advised by the paper
+ # this is suggested based on the original and selected paper
    pos_embed = 'conv', # In the original paper it is considered inside the convolutional layer
    classification = False, #this is not used for classifcation
    dropout_rate = 0.1 # suggested by the original paper
)


# In[ ]:


#define the class of reproduced UNETR(UNETR_R)
class UNETR_R(nn.Module):
    """
    This class is for the proposed UNET-R model implemented by the previously introduced classes.
    The inputs are as follows:
    input_channels_UNETR #number of channels in the input image(e.g., CT Spleen is 1 and MRI is 4)
    output_channels_UNETR #number of channels in the input image(e.g., CT Spleen is 2 and MRI is 4)
    img_size_UNETR #size of the output image (defined by paper as (96,96,96)
    norm_name #normalization layer type
    """
    def __init__(
        self,
        input_channels_UNETR: int,#number of channels in the input image
        output_channels_UNETR: int,#number of channels in the output image
        img_size_UNETR: (int,int,int), #size of the output image
        norm_name: str='batch', #normalization layer defined as batch normalization
         ):

        super().__init__()# initialize the class
        img_size = ensure_tuple_rep((96, 96, 96), 3) #defined by the paper
        self.patch_size = ensure_tuple_rep(16, 3)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.embedding_size = 768 #embedding size C = 768
        
        #setup the layers of the block as showsn in the above figure
        #encoder section(compression)
        self.transformer_ViT = ViT(input_channels_UNETR,img_size_UNETR,self.patch_size,768,3072,12,12,'perceptron',classification=False,dropout_rate=0.1, spatial_dims=3) 
        self.encoder_3 = Conv_Encod(size_Conv=3,input_channels=input_channels_UNETR,output_channels=16,kernel_size=3,stride=1,norm_name= 'batch')
        self.encoder_6 = Conv_DecConv_Encod(size_Conv=3,input_channels=768,output_channels=32,number_layer=2,kernel_size=3,stride=1,upsample_stride=2,norm_name='batch')
        self.encoder_9 = Conv_DecConv_Encod(size_Conv=3,input_channels=768,output_channels=64,number_layer=1,kernel_size=3,stride=1,upsample_stride=2,norm_name='batch')
        self.encoder_12 = Conv_DecConv_Encod(size_Conv=3,input_channels=768,output_channels=128,number_layer=0,kernel_size=3,stride=1,upsample_stride=2,norm_name='batch')
        #decoder section(expanstion)
        self.decoder_12 = decoder_expansion_blk(size_Conv=3,input_channels=768,output_channels=128,kernel_size=3,upsample_stride=2,norm_name='batch')
        self.decoder_9 = decoder_expansion_blk(size_Conv=3,input_channels=128,output_channels=64,kernel_size=3,upsample_stride=2,norm_name='batch')
        self.decoder_6 = decoder_expansion_blk(size_Conv=3,input_channels=64,output_channels=32,kernel_size=3,upsample_stride=2,norm_name='batch')
        self.decoder_3 = decoder_expansion_blk(size_Conv=3,input_channels=32,output_channels=16,kernel_size=3,upsample_stride=2,norm_name='batch')
        self.output_UNETR_R = output(3, input_channels=16, output_channels=output_channels_UNETR)
    
    #define the architecture of the class
    def forward(self, inputs):
        embedded_values, embedding_size_out = self.transformer_ViT(x_in)#output of the transformer
        encoder_3_value = self.encoder_3(inputs)#output value for encoder 3
        embedded_values_3 = embedding_size_out[3]#output of the third encoder
        encoder_6_value = self.encoder_6(self.linear_projection(embedded_values_3, self.embedding_size, self.feat_size))#output of the encoder 6
        embedded_values_6 = embedding_size_out[6] #values for the encoder 6
        encoder_9_value = self.encoder_9(self.linear_projection(embedded_values_6, self.embedding_size, self.feat_size))
        embedded_values_9 = embedding_size_out[9]#values for encoder 9 
        encoder_12_value = self.encoder_12(self.linear_projection(embedded_values_9, self.embedding_size, self.feat_size))#output for the encoder 12
        decoder_12_value = self.linear_projection(embedded_values, self.embedding_size, self.feat_size)#output for decoder 12
        decoder_9_value = self.decoder_12(decoder_12_value, encoder_12_value)#output for decoder 9
        decoder_6_value = self.decoder_9(decoder_9_value, encoder_9_value)#output for decoder 6
        decoder_3_value = self.decoder_6(decoder_6_value, encoder_6_value)#output for decoder 3
        output_UNETR_R = self.decoder_3(decoder_3_value, encoder_3_value)#output for the UNETR using skip connections
        return self.output_UNETR_R(output_UNETR_R) #output


# In[24]:



model = UNETR_R(
   in_channels=1,
   out_channels=3,
   img_size=(roi_size, roi_size, z_size),
   feature_size=16,
   hidden_size=768,
   mlp_dim=3072,
   num_heads=12,
   pos_embed="perceptron",
   norm_name="instance",
   res_block=True,
   dropout_rate=0.0,
).to(device)

set_determinism(seed=0)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)


# In[ ]:


#train the model
def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    dice_liver_vals = list()
    dice_tumour_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (roi_size, roi_size, z_size), 6, model)
            val_labels_list =       (val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_organ = dice_metric_batch.aggregate()

            metric_values_liver = dice_organ[0].item()
            metric_values_tumour = dice_organ[1].item()
            dice_vals.append(dice)
            dice_liver_vals.append(metric_values_liver)
            dice_tumour_vals.append(metric_values_tumour)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)(Liver=%2.5f)(Tumour=%2.5f)" % (global_step, 10.0, dice, metric_values_liver, metric_values_tumour)
            )
        dice_metric.reset()
        dice_metric_batch.reset()
    mean_dice_val = np.mean(dice_vals)
    mean_dice_liver_val = np.mean(dice_liver_vals)
    mean_dice_tumour_val = np.mean(dice_tumour_vals)
    
    return mean_dice_val, mean_dice_liver_val, mean_dice_tumour_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                 
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            [dice_val, dice_liver_vals, dice_tumour_vals] = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values_DSC.append(dice_val)
            metric_values_liver_DSC.append(dice_liver_vals)
            metric_values_tumour_DSC.append(dice_tumour_vals)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                #HD95_val_best = HD95_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model_100000_epoch_roi_192_192_48_n_sample_6.pth")
                )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 100000
eval_num = 1000
post_label = AsDiscrete(to_onehot=True, n_classes=3)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
global_step = 0
dice_val_best = 0.0
HD95_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values_DSC = []
metric_values_liver_DSC = []
metric_values_tumour_DSC = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
#save the model
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_100000_epoch_roi_192_192_48_n_sample_6.pth")))


# ### Visualization the Results

# In[26]:


plt.figure("train", (12, 6))
plt.subplot(1, 4, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 4, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values_DSC))]
y = metric_values_DSC
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 4, 3)
plt.title("Val Mean Liver Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values_liver_DSC))]
y = metric_values_liver_DSC
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 4, 4)
plt.title("Val Mean Tumour Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values_tumour_DSC))]
y = metric_values_tumour_DSC
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()

print("The best mean DSC liver achieved as ", max(metric_values_liver_DSC), "in the iteration of", x[metric_values_liver_DSC.index(max(metric_values_liver_DSC))])
print("The best mean global DSC achieved as ", max(metric_values_DSC), "in the iteration of", x[metric_values_DSC.index(max(metric_values_DSC))])
print("The best mean DSC tumour achieved as ", max(metric_values_tumour_DSC), "in the iteration of", x[metric_values_tumour_DSC.index(max(metric_values_tumour_DSC))])
print("The least mean loss achieved as ", min(epoch_loss_values), "in the iteration of", x[epoch_loss_values.index(min(epoch_loss_values))])

