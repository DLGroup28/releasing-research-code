
# coding: utf-8

# ### Group#: 28 - BRATS UNET-R
# ### Couse: CISC867
# ### Team: Ramtin Mojtahedi - Vignesh Rao

# ## Setup environment

# In[1]:


#Install MONAI libraries and packages
get_ipython().system('pip install -q "monai-weekly[nibabel, tqdm, einops]"# install weekely MONAI')
get_ipython().system('python -c "import matplotlib" || pip install -q matplotlib#install plot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#testing the availability of Cuda
import torch
torch.cuda.is_available()


# ## Setup imports

# In[5]:


# Import libraries and packages
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.losses import DiceCELoss
import torch
import glob # used for detecting specific file extention
import tempfile
from tqdm import tqdm
from monai.transforms import (#import transformers
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
)
#MONAI data loading
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


# ### Setup Transformers and Spacing

# In[8]:


# Define composing to make transformation
transform_train = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureChannelFirstd(keys="image"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),#spacing
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),#size of images 128*128*128
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)
transform_validation = Compose(
    [
        EnsureChannelFirstd(keys="image"),
        LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),#spacing
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"]),
    ]
)


# In[9]:


#loading data
data_dir = os.getcwd() +"/Task01_BrainTumour"

images_train = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))

labels_train = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(images_train, labels_train)
]
files_train, files_validation = data_dicts[:-97], data_dicts[-97:]#80/20 split ratio


# In[10]:


files_train


# In[11]:


#loading and caching data
b_size = 1#batch size
ds_train = CacheDataset(
    data=files_train, transform=transform_train,
    cache_rate=0.0, num_workers=4)

loader_train = DataLoader(ds_train, batch_size=b_size, shuffle=True, num_workers=4)

ds_validation = CacheDataset(
    data=files_validation, transform=transform_validation, cache_rate=0.0, num_workers=4)

loader_validation  = DataLoader(ds_validation, batch_size = 1, num_workers=4)


# ## Model UNETR_R

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
        conv_size : int,
        input_channels: int, # number of input channels, which is euqal to 3
        output_channels: int, # number of output channel, which is euqal to 3
        kernel_size: int, # kernel size. In the ViT paper defined as 3
        stride: int = 1, #default stride value
        normalization_type : str = 'batch', #layer normalization type based on the model's graph
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
        self.normalization1 = get_norm_layer(normalization_type , conv_size , output_channels)#layer normalization
        self.normalization2 = get_norm_layer(normalization_type , conv_size , output_channels)#layer normalization
        self.normalization3 = get_norm_layer(normalization_type , conv_size , output_channels)#layer normalization

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
        conv_size : int, #spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int,#number of channels in the output image
        number_layer: int, #number of Conv layers
        kernel_size: Union[Sequence[int], int],# based on the paper's suggestion
        stride: Union[Sequence[int], int],#defaultstride
        upsample_stride: Union[Sequence[int], int], # Based on the model's graph
        normalization_type : str = 'batch'#layer normalization type based on the model's graph
                 ):
        
        super().__init__()#initialize the class

        # define the transpose conv
        self.transpose_conv_layer  = get_conv_layer(conv_size , input_channels, output_channels, upsample_stride, stride, conv_only=True, is_transposed=True) 
        #list two modules to make them as a block
        self.tr_blocks = nn.ModuleList
        (
         [nn.Sequential(
        get_conv_layer(conv_size ,output_channels,output_channels,upsample_stride,upsample_stride,conv_only=True,is_transposed=True),
        UnetResBlock(spatial_dims=conv_size , in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,stride=stride,normalization_type =normalization_type ))
        for i in range(number_layer) # 12 trasnformers
         ] 
        )
    #network structure
    def forward(self, input_x):
        input_x = self.transpose_conv_layer (input_x)# deconv the value
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
        conv_size : int,#spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int,#number of channels in the output image
        kernel_size: Union[Sequence[int], int],#based on the paper's suggestion
        upsample_stride: Union[Sequence[int], int],# Based on the model's graph
        normalization_type : str = 'batch'#layer normalization type based on the model's graph
    ):
    
        super().__init__()#initialize the class
        
        self.transpose_conv_layer  = get_conv_layer(conv_size , input_channels, output_channels, kernel_size=upsample_stride, conv_only=True,is_transposed=True) 
        self.conv_block = UnetResBlock(conv_size ,output_channels+output_channels,output_channels,kernel_size,stride=1,normalization_type =normalization_type )
    #define architecture - number of skip connections is equal to channels
    def forward(self, input_value, skip_connection):
        output_decoder = self.transpose_conv_layer (input_value)# deconv
        output_decoder = torch.cat((output, skip_connection), dim=1)
        output_decoder = self.conv_block(output_decoder)
        return output_decoder


# In[ ]:


# # This class is for the output of the U-Net architecture
class output(nn.Module):
    #initialize class
    def __init__(
        self, 
        conv_size : int,#spatial size of conv
        input_channels: int, #number of channels in the input image
        output_channels: int, #number of channels in the output image
    ):
        super().__init__()#initialize the class
        self.conv_output = get_conv_layer(conv_size , input_channels, output_channels, kernel_size=1, stride=1, bias=True, conv_only=True)
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


transformer_ViT = ViT(
    in_channels = 1,
    img_size = (96, 96, 96),# patch size images 
    patch_size = (96, 96, 96), # Define the Patch size as advised by the paper
 # this is suggested based on the original and selected paper
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
    normalization_type  #normalization layer type
    """
    def __init__(
        self,
        input_channels_UNETR: int,#number of channels in the input image
        output_channels_UNETR: int,#number of channels in the output image
        img_size_UNETR, #size of the output image
        normalization_type : str='batch', #normalization layer defined as batch normalization
         ):

        super().__init__()# initialize the class
        img_size = ensure_tuple_rep((96, 96, 96), 3) #defined by the paper
        self.patch_size = ensure_tuple_rep(16, 3)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.embedding_size = 768 #embedding size C = 768
        
        #setup the layers of the block as showsn in the above figure
        #encoder section(compression)
        self.transformer_ViT = ViT(input_channels_UNETR,img_size_UNETR,self.patch_size,768,3072,12,12,'perceptron',classification=False,dropout_rate=0.1, spatial_dims=3) 
        self.encoder_3 = Conv_Encod(conv_size =3,input_channels=input_channels_UNETR,output_channels=16,kernel_size=3,stride=1,normalization_type = 'batch')
        self.encoder_6 = Conv_DecConv_Encod(conv_size =3,input_channels=768,output_channels=32,number_layer=2,kernel_size=3,stride=1,upsample_stride=2,normalization_type ='batch')
        self.encoder_9 = Conv_DecConv_Encod(conv_size =3,input_channels=768,output_channels=64,number_layer=1,kernel_size=3,stride=1,upsample_stride=2,normalization_type ='batch')
        self.encoder_12 = Conv_DecConv_Encod(conv_size =3,input_channels=768,output_channels=128,number_layer=0,kernel_size=3,stride=1,upsample_stride=2,normalization_type ='batch')
        #decoder section(expanstion)
        self.decoder_12 = decoder_expansion_blk(conv_size =3,input_channels=768,output_channels=128,kernel_size=3,upsample_stride=2,normalization_type ='batch')
        self.decoder_9 = decoder_expansion_blk(conv_size =3,input_channels=128,output_channels=64,kernel_size=3,upsample_stride=2,normalization_type ='batch')
        self.decoder_6 = decoder_expansion_blk(conv_size =3,input_channels=64,output_channels=32,kernel_size=3,upsample_stride=2,normalization_type ='batch')
        self.decoder_3 = decoder_expansion_blk(conv_size =3,input_channels=32,output_channels=16,kernel_size=3,upsample_stride=2,normalization_type ='batch')
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


max_epochs = 500
val_interval = 1
VAL_AMP = True
device = torch.device("cuda:0")

#Define the model
model = UNETR_R(
    in_channels=4,
    out_channels=3,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    normalization_type ="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)
#setup the model parameters
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
)


# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
#             roi_size=(240, 240, 160),
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


# ## Training the Model

# In[ ]:


#Define epochs and iterations
max_epochs = 500
val_interval = 1
VAL_AMP = True
#******************************#defien metrics
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = [] # metrics for the 3 types of brain tumour data
metric_values_wt = []
metric_values_et = []

#training
total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in loader_train:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in loader_validation :
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()#FOR TC
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()#FOR WT
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()#FOR ET
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()
#shows the details of metrics for labels
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, "best_metric_model.pth"),#save the model
                )
total_time = time.time() - total_start


# In[14]:


print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


# ## Visualization of the Results

# In[15]:

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
y = metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
y = metric_values_wt
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
y = metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.show()


#plot and visualize different metrics
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()



# ## Checking the Results for Brain Tumour Detection

# In[17]:


#loading the model
model.load_state_dict(
    torch.load(os.path.join(root_dir, "best_metric_model.pth"))
)
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = ds_validation[6]["image"].unsqueeze(0).to(device)
    roi_size = (128, 128, 64)
    sw_batch_size = 4
    val_output = inference(val_input)
    val_output = post_trans(val_output[0])
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(ds_validation[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(ds_validation[6]["label"][i, :, :, 70].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 70].detach().cpu())
    plt.show()

