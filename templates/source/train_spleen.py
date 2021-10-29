
# coding: utf-8

# ### Project: Deep Learning Course Project - UNETR
# ### Task: Spleen MSD Data Set
# ### Team: Ramtin Mojtahedi - Vignesh Rao
# ### Group#: 28

# ### Coding Part

# In[2]:


# Import required libraries and Packages
import torch.multiprocessing # import multi-processing
torch.multiprocessing.set_sharing_strategy('file_system')# This is used to share memory to provide shared views on the same data in different processes
import os #import OS
import shutil # Offers high-level file processing
import matplotlib.pyplot as plt #Import plot
import numpy as np # import numpy
from tqdm import tqdm # allows you to output a smart progress bar by wrapping around any iterable
import torch# import torch
import glob # used for detecting specific file extention
from typing import Sequence, Tuple, Union
from typing import Sequence, Tuple, Union# define the sequence values and matrices related libraries from MONAI
from monai.utils.misc import ensure_tuple_rep #Returns a copy of tup with dim values by either shortened or duplicated input.
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Install and import MONAI Packages and libraries

# In[2]:


# Import MONAI libraries and packages
get_ipython().system("pip install 'monai[all]' # Install all MONAI libraries")
get_ipython().system('pip install -q "monai-weekly[nibabel, einops]" #Install weekly updated version of MONAI for file extention of nibabel and platform of einops')


# In[3]:


from monai.networks.layers import Norm #import MONAI layer normalization
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric #import MONAI metric of DSC, HD, and ASD
from monai.losses import DiceLoss #import dice loss
from monai.networks.nets import UNet #import developed U-Net by MONAI 
from monai.networks.nets import vit #IMPORT developed ViT transformer by MONAI
from monai.inferers import sliding_window_inference #to maintain the same spatial sizes, the output image will be cropped to the original input size.
from monai.data import CacheDataset, DataLoader, decollate_batch #import related libraries for data loading
#import MONAI transforming functions
from monai.transforms import (
    Compose, #making the tranformation
    CropForegroundd, #crop the image
    #RandFlipd,
    #Orientationd,
    ScaleIntensityRanged, #change the intensity
    #RandRotate90d,
    RandAffined, #random spatial rotation
    Spacingd, #change the image spacing
)


# #### UNETR Model Implementation
# ![2021-10-22_182105.jpg](attachment:2021-10-22_182105.jpg)

# In the U-Net architecture for the encoder side there is no Res block. However as mentioned in the paper's architecture, sequence representations are being extracted, Zi (i ∈{3,6,9,12}), from the transformers using residual blocks and are given to the decoder section using skip connections. 

# In[4]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# #### Transfromer (ViT-B16)

# In[10]:


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


# #### UNET-R Model

# In[11]:


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


# In[12]:


#setting up yhe model
model = UNETR_R(
    input_channels_UNETR=1,# input channel 
    output_channels_UNETR=2, #binary - foreground and background
    img_size_UNETR=(96, 96, 96), #volume sizes
    norm_name='batch', #layer normalization
).to(torch.device("cuda:0")) #check if the cuda and gpu available

loss_function = DiceLoss(to_onehot_y=True, softmax=True) #define the dice loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)#defien the Adam optimizer
dice_metric = DiceMetric(include_background=False, reduction="mean") #defien the average dice
HD95_metric = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95)#deifine the HD with 95% percentile
ASD_metric = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean')#define the avergae surface


# In[13]:


data_directory = "/home/ramtin/Desktop/DL/Spleen" #define the data directory


# In[14]:


# find and sort test images
test_images = sorted(glob.glob(os.path.join(data_directory, "imagesTs", "*.nii.gz")))
test_labels = sorted(glob.glob(os.path.join(data_directory, "labelsTs", "*.nii.gz")))
# find and sort traing images
train_images = sorted(glob.glob(os.path.join(data_directory, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_directory, "labelsTr", "*.nii.gz")))

#assig labels to train and define train and validation images - with the ratio of 80:15:5
data_assigned = [{"image": image_name, "label": label_name}for image_name, label_name in zip(train_images, train_labels)]
train_data, validation_data = data_assigned[:-10], data_assigned[-10:]


# #### Data Augmentation

# In[15]:


#transforming data for data augmentation

train_transforms_augmentation = Compose(
    [
        LoadImaged(keys=["image", "label"]),#load images based on the image and its label
        EnsureChannelFirstd(keys=["image", "label"]),# check if there is an image in the first channel
        # change the resolution based on the paper's recommendation
        Spacingd(keys=["image", "label"], pixdim=(
            1, 1, 1), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
                # use random spatial rotation
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)),
        
        #chage the CT intensity based on the range of the image's intensities
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        
        #crop the foreground of the image
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

#do the same rtansformations on the validation data set
val_transforms_augmentation = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1, 1, 1), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        #crop the foreground
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
)


# In[ ]:


#define the data loader
data_loader = DataLoader(check_ds, batch_size=2)
spleen_image, spleen_label = (data_loader["image"][0][0], data_loader["label"][0][0])#get the image and its label


# #### Training the model

# In[ ]:


#define the batch size
b_size = 2

#apply transformer
train_data_augmented = CacheDataset(
    data=train_images, transform=train_transforms_augmentation,
    cache_rate=1.0, num_workers=4)

# loading to train data
train_loading = DataLoader(train_data_augmented, batch_size=b_size, shuffle=True, num_workers=4)

#apply transformer
validation_data_augmeted = CacheDataset(
    data=validation_data, transform=val_transforms_augmentation, cache_rate=1.0, num_workers=4)

# loading data to validation data set
validation_loading = DataLoader(validation_data_augmeted, batch_size = b_size, num_workers=4)


# In[ ]:


#define validation data and calculate the mentioned metrics
def validation(validation_epoch):
    model.eval()#make a list 
    dice_vals = list() #make a list 
    HD95_vals = list()#make a list 
    ASD_vals = list()#make a list 
    with torch.no_grad(): #disabled gradient calculation
        for step, batch in enumerate(validation_epoch):
            val_inputs, val_labels = (batch["spleen_image"].cuda(), batch["spleen_label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), b_size, model)
            val_labels_list = (val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            #define the evalautaion metrics
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            HD95_metric(y_pred=val_output_convert, y=val_labels_convert)
            ASD_metric(y_pred=val_output_convert, y=val_labels_convert)
            HD95 = HD95_metric.aggregate().item()                       
            dice = dice_metric.aggregate().item()
            ASD = ASD_metric.aggregate().item()
            HD95_vals.append(HD95)
            dice_vals.append(dice)
            ASD_vals.append(ASD)
            #reset the metrics
        HD95_metric.reset()
        dice_metric.reset()
        ASD_metric.reset()
        #make average of the metrics
    mean_dice_val = np.mean(dice_vals)
    mean_HD95_vals = np.mean(HD95_vals)
    mean_ASD_vals = np.mean(ASD_vals)
    #return metrics
    return mean_dice_val, mean_HD95_vals, mean_ASD_vals

#define model's training
def train_model(step, train_loading, dice_val_best, step_best):
    model.train_model()
    epoch_loss = 0
    step = 0
    #shows the steps
#     epoch_iteration = tqdm(
#         train_loader, desc="Training (X / X Steps) (loss_validation=X.X)", dynamic_ncols=True
#     )
    for step, batch in enumerate(epoch_iteration):
        step += 1
        x, y = (batch["spleen_image"].cuda(), batch["spleen_label"].cuda())
        logit_map = model(x)
        loss_validation = loss_function(logit_map, y)
        loss_validation.backward()
        epoch_loss += loss_validation.item()
        optimizer.step()
        optimizer.zero_grad() #no gradient calculation

        if (
            step % eval_num == 0 and step != 0
        ) or step == Iterations:
#             validation_epoch = tqdm(
                
#                 val_loader, desc="(X / X Steps)", dynamic_ncols=True
#             )
            #define the metrics
            [dice_val, HD95_val, ASD_val] = validation(validation_epoch)
            epoch_loss /= step
            #append values to each of the metrics
            epoch_loss_values.append(epoch_loss)
            metric_values_DSC.append(dice_val)
            metric_values_HD95.append(HD95_val)
            metric_values_ASD.append(ASD_val)
            #we calculate the best dice as the most important metric in image segementation
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                step_best = step
                torch.save(
                    model.state_dict(), os.path.join(data_directory, "Best_Spleen_DSC_Model.pth")
                )
        step += 1
    return step, dice_val_best, step_best #return step, dice value and the best step for dice value

#setting the parameters
Iterations = 25000 #number of iterations based on the paper
eval_num = 250 #average is being calculated on every 250 samples
post_label = AsDiscrete(to_onehot=True, n_classes=2)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
step = 0
dice_val_best = 0.0
HD95_val_best = 0.0
step_best = 0
#define the metrics as a metrics
epoch_loss_values = []
metric_values_DSC = []
metric_values_HD95 = []
metric_values_ASD = []
#save the metrics until reaching to the maximum number of iterations
while step < Iterations:
    step, dice_val_best, step_best = train_model(
        step, train_loader, dice_val_best, step_best
    )


# #### Visualize the results

# In[189]:


#load the model
model.load_state_dict(torch.load(os.path.join(data_directory, "Best_Spleen_DSC_Model.pth")))

#define subplots for each metric
fig=plt.figure()
plt.figure("train", (25, 15))
plt.subplot(1, 4, 1)
plt.title("Average Loss", fontsize=20)
x_loss = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y_loss = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x_loss, y_loss)
#--------------------------------
plt.subplot(1, 4, 2)
plt.title("Average DSC", fontsize=20)
x_DSC = [eval_num * (i + 1) for i in range(len(metric_values_DSC))]
y_DSC = metric_values_DSC
plt.xlabel("Iteration")
plt.plot(x_DSC, y_DSC, color="green")
#--------------------------------
plt.subplot(1, 4, 3)
plt.title("Average HD95", fontsize=20)
x_HD95 = [eval_num * (i + 1) for i in range(len(metric_values_HD95))]
y_HD95 = metric_values_HD95
plt.xlabel("Iteration")
plt.plot(x_HD95, y_HD95, color="black")
#--------------------------------
plt.subplot(1, 4, 4)
plt.title("Average ASD", fontsize=20)
x_ASD = [eval_num * (i + 1) for i in range(len(metric_values_ASD))]
y_ASD = metric_values_ASD
plt.xlabel("Iteration")
plt.plot(x_ASD, y_ASD, color="red")
#-------------------------------- Prin the results and the best one with respect to its iteration
print("The best DSC achieved as ", max(y_DSC), "in the iteration of", x_DSC[y_DSC.index(max(y_DSC))])
print("The least HD95 achieved as ", min(y_HD95), "in the iteration of", x_HD95[y_HD95.index(min(y_HD95))])
print("The least loss value achieved as ", min(y_loss), "in the iteration of", x_loss[y_loss.index(min(y_loss))])
print("The best ASD achieved as ", min(y_ASD), "in the iteration of", x_ASD[y_ASD.index(min(y_ASD))])


# In[192]:


#save the metrics values
torch.save(epoch_loss_values, 'loss')
torch.save(metric_values_DSC, 'DSC')
torch.save(metric_values_HD95, 'HD95')
torch.save(metric_values_ASD, 'ASD')


# #### SHow the results on a sample image

# In[181]:


#load the best achieved model
model.load_state_dict(torch.load(
    os.path.join(data_directory, "best_metric_model.pth")))
#to not have gradient calculation
with torch.no_grad():
    for i, validation_loading in enumerate(validation_loading):
        roi_size = (96, 96, 96) # size of the output image based on the paper
        sw_batch_size = 2 # define the batch size
        val_outputs = sliding_window_inference(
            validation_data_augmeted["spleen_image"].to(device), roi_size, sw_batch_size, model
        )
        #plot the images
        plt.subplot(1, 3, 1)
        plt.title(f"Raw_Image_{i}") # show raw CT image
        plt.imshow(val_data["image"][0, 0, :, :, 96], cmap="gray")#gray backgroun for CT image
        plt.subplot(1, 3, 2)
        plt.title(f"Ground_Truth_Image_{i}") #show the ground truth images
        plt.imshow(val_data["label"][0, 0, :, :, 96])
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted_Image_{i}") #show the predicted images
        plt.imshow(torch.argmax(
            val_outputs, dim=1).detach().cpu()[0, :, :, 96])
        plt.show()
        #shows only the four predicted images (out of 10 images)
        if i == 4:
            break

