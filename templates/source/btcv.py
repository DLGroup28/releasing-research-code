
# coding: utf-8

# ### Group#: 28 - BTCV UNET-R
# ### Couse: CISC867
# ### Team: Ramtin Mojtahedi - Vignesh Rao

# In[1]:


#testing the availability of Cuda
import torch
torch.cuda.is_available()


# ## Installing libraries and packages

# In[6]:


#Install MONAI libraries and packages
get_ipython().system('pip install -q "monai-weekly[nibabel, tqdm, einops]"# install weekely MONAI')
get_ipython().system('python -c "import matplotlib" || pip install -q matplotlib#install plot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#import libraries
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
#transformations functions
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)
#import packages for data loading
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
print_config()


# In[8]:


#cuda version
torch.version.cuda 


# ## Setup transforms for training and validation_training

# In[10]:


#Define Transformation functions
train_transformation_functions = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),#spacing of the images
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizedIntensity(),
        CropForegroundd(keys=["image", "label"], source_key="image"),#crop the volumentric foreground image
        RandShiftIntensityd(#shift in intensity
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),# make it to tensor to be given to the device
    ]
)
#validation_training tranformers
validation_training_transformation_functions = Compose(
    [
        LoadImaged(keys=["image", "label"]),#load images
        AddChanneld(keys=["image", "label"]),#make sure the channel has images and labels
        Orientationd(keys=["image", "label"], axcodes="RAS"),#orienting the validation_training images
        NormalizedIntensity(),
        CropForegroundd(keys=["image", "label"], source_key="image"),#crop the foreground images
        ToTensord(keys=["image", "label"]),#make it ready to be given to the tensors
    ]
)


# In[11]:


#define the data loader and cache rate for data
data_root = "DL/data/"
split_JSON = "dataset_0.json"
data = data_root + split_JSON
datalist = load_decathlon_datalist(data, True, "training")
validation_training_files = load_decathlon_datalist(data, True, "validation_training")
train_data = CacheDataset(
    data=datalist,
    transform=train_transformation_functions,
    cache_num=32,#number of cach data
    cache_rate=1.0,#cache rate
    num_workers=12,#number of CPUs in caching data
)
#define the training data
training_loader = DataLoader(
    train_data, batch_size=6, shuffle=True, num_workers=12, pin_memory=True
)
validation_training_data = CacheDataset(
    data=validation_training_files, transform=validation_training_transformation_functions, cache_num=12, cache_rate=1.0, num_workers=12
)
validation_training_loader = DataLoader(
    validation_training_data, batch_size=6, shuffle=False, num_workers=12, pin_memory=True
)


# ## Visualization a Sample of Data

# In[15]:


#Visualizing sample of labels
image_number = 1
image_name = os.path.split(validation_training_data[image_number]["image_meta_dict"]["filename_or_obj"])[1]
image = validation_training_data[image_number]["image"]
label = validation_training_data[image_number]["label"]
image_shape = image.shape
label_shape = label.shape
print(f"image shape: {image_shape}, label shape: {label_shape}")
plt.figure("image", (24, 12))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, :, :, slice_map[image_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[image_name]].detach().cpu())
plt.show()


# ### Model UNETR_R

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


# #### ViT Transfomer

# In[ ]:


# from monai.networks.nets.vit import ViT# import ViT architecture from MONAI
# '''
# UNETR is is similar to the popular CNN-based architecture known as UNET. 
# The proposed architecture add transformer to the decoder section. 
# The transfromer and its idea is achieved from the paper "An image is worth 16x16 words: Transformers for image recognition at scale.
# The used transformer called ViT-B16 with the parameters of L = 12 layers, an embedding size of K = 768 and patch size of 16*16*16 that is implemented in MONAI
# The used ViT parameters introduced in MONAI are as follows:

#         in_channels (int) – dimension of input channels.

#         image_size (Union[Sequence[int], int]) – dimension of input image.

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
    in_channels = 1, # this is for BTCV data 
    image_size = (96, 96, 96),# patch size images 
    patch_size = (96, 96, 96), # Define the Patch size as advised by the paper
+ # this is suggested based on the original and selected paper
    pos_embed = 'conv', # In the original paper it is considered inside the convolutional layer
    classification = False, #this is not used for classifcation
    dropout_rate = 0.1 # suggested by the original paper
)


# #### UNETR_R Model

# In[ ]:


#define the class of reproduced UNETR(UNETR_R)
class UNETR_R(nn.Module):
    """
    This class is for the proposed UNET-R model implemented by the previously introduced classes.
    The inputs are as follows:
    input_channels_UNETR #number of channels in the input image(e.g., CT Spleen is 1 and MRI is 4)
    output_channels_UNETR #number of channels in the input image(e.g., CT Spleen is 2 and MRI is 4)
    image_size_UNETR #size of the output image (defined by paper as (96,96,96)
    norm_name #normalization layer type
    """
    def __init__(
        self,
        input_channels_UNETR: int,#number of channels in the input image
        output_channels_UNETR: int,#number of channels in the output image
        image_size_UNETR: (int,int,int), #size of the output image
        norm_name: str='batch', #normalization layer defined as batch normalization
         ):

        super().__init__()# initialize the class
        image_size = ensure_tuple_rep((96, 96, 96), 3) #defined by the paper
        self.patch_size = ensure_tuple_rep(16, 3)
        self.feat_size = tuple(image_d // p_d for image_d, p_d in zip(image_size, self.patch_size))
        self.embedding_size = 768 #embedding size C = 768
        
        #setup the layers of the block as showsn in the above figure
        #encoder section(compression)
        self.transformer_ViT = ViT(input_channels_UNETR,image_size_UNETR,self.patch_size,768,3072,12,12,'perceptron',classification=False,dropout_rate=0.1, spatial_dims=3) 
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


# In[ ]:


#setting up yhe model
model = UNETR_R(
    input_channels_UNETR=1,# input channel 
    output_channels_UNETR=14, #binary - foreground and background
    image_size_UNETR=(128, 128, 128), #volume sizes
    norm_name='batch', #layer normalization
).to(torch.device("cuda:0")) #check if the cuda and gpu available

loss_function = DiceLoss(to_onehot_y=True, softmax=True) #define the dice loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)#defien the Adam optimizer
dice_metric = DiceMetric(include_background=False, reduction="mean") #defien the average dice


# ### Execute a typical PyTorch training process

# In[ ]:


#define validation_training data and calculate the mentioned metrics
def validation_training(validation_training_epoch):
    model.eval()#make a list 
    dice_validationidation = list() #make a list  
    with torch.no_grad(): #disabled gradient calculation
        for step, batch in enumerate(validation_training_epoch):
            validation_inputs, validation_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(validation_inputs, (128, 128, 128), 6, model)
            validation_labels_list = (validation_labels)
            validation_labels_convert = [
                post_label(validation_label_tensor) for validation_label_tensor in validation_labels_list
            ]
            validation_outputs_list = decollate_batch(val_outputs)
            validation_output_convert = [
                post_pred(validation_pred_tensor) for validation_pred_tensor in validation_outputs_list
            ]
            #define the evalautaion metrics
            dice_metric(y_pred=validation_output_convert, y=validation_labels_convert)                      
            dice = dice_metric.aggregate().item()
            dice_validationidation.append(dice)
            #reset the metrics
        HD95_metric.reset()
        dice_metric.reset()
        ASD_metric.reset()
        #make average of the metrics
    mean_dice_validation = np.mean(dice_validationidation)
    mean_HD95_vals = np.mean(HD95_vals)
    mean_ASD_vals = np.mean(ASD_vals)
    #return metrics
    return mean_dice_validation, mean_HD95_vals, mean_ASD_vals

#define model's training
def training_model(step, train_loading, dice_validationidation_best, step_best):
    model.training_model()
    epoch_loss = 0
    step = 0
    #shows the steps
#     epoch_iteration = tqdm(
#         training_loader, desc="Training (X / X Steps) (loss_validation_training=X.X)", dynamic_ncols=True
#     )
    for step, batch in enumerate(epoch_iteration):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss_validation_training = loss_function(logit_map, y)
        loss_validation_training.backward()
        epoch_loss += loss_validation_training.item()
        optimizer.step()
        optimizer.zero_grad() #no gradient calculation

        if (
            step % eval_num == 0 and step != 0
        ) or step == EPOCHS:
#             validation_training_epoch = tqdm(
                
#                 validation_training_loader, desc="(X / X Steps)", dynamic_ncols=True
#             )
            #define the metrics
            [dice_validation, HD95_val, ASD_val] = validation_training(validation_training_epoch)
            epoch_loss /= step
            #append values to each of the metrics
            epoch_loss_values.append(epoch_loss)
            metric_values_DSC.append(dice_validation)
            metric_values_HD95.append(HD95_val)
            metric_values_ASD.append(ASD_val)
            #we calculate the best dice as the most important metric in image segementation
            if dice_validation > dice_validationidation_best:
                dice_validationidation_best = dice_validation
                step_best = step
                torch.save(
                    model.state_dict(), os.path.join(data_rootectory, "Best_BTCV_DSC_Model.pth")
                )
        step += 1
    return step, dice_validationidation_best, step_best #return step, dice value and the best step for dice value

#setting the parameters
EPOCHS = 25000 #number of EPOCHS based on the paper
eval_num = 500 #average is being calculated on every 250 samples
post_label = AsDiscrete(to_onehot=True, n_classes=2)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
step = 0
dice_validationidation_best = 0.0
step_best = 0
#define the metrics as a metrics
epoch_loss_values = []
metric_values_DSC = []
#save the metrics until reaching to the maximum number of EPOCHS
while step < EPOCHS:
    step, dice_validationidation_best, step_best = training_model(
        step, training_loader, dice_validationidation_best, step_best
    )


# In[19]:


print(
    f"train completed, best_metric: {dice_validation_best:.4f} "
    f"at iteration: {global_step_best}"
)


# ### Plot the loss and metric

# In[22]:


plt.figure("train", (18, 12))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()

