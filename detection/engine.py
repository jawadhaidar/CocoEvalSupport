import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform,GeneralizedRCNNTransformMy

import matplotlib.pyplot as plt 
import mmdet 
import mmdet.apis
# adding Folder_2 to the system path
# sys.path.append('/home/jawad/codes/')
#  #TODO: check why import * is causing an error
# from OneClassMethod.helper_functions import inference_real_engine_compatible,inference_real_engine_compatible_withfaster,multiout_2_unoout
sys.path.append('/home/jawad/codes/MaskUno')
 #TODO: check why import * is causing an error
from helper_functions import quantitative



#model_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device="cpu")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device,flag,detectionpipeline,classid):
    '''
    parameters:
    model
    data_loader
    device
    flag : 0 finetunedmaskecnn 1 maskrcnn 2 dynamic with bbx from gt 3dynamic with bbx from fasterrcnn
    classid: cococlass 17 cat 18 dog 2 bicycle
    '''
    model_detection = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device=device)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    #flag=0
    if flag==2 or flag==3 or flag==4: #TODO: fix this
        iou_types.append("segm")
    print(f'iou_types {iou_types}')
    coco_evaluator = CocoEvaluator(coco, iou_types)
    ##################for opyion 4
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if detectionpipeline=="normal":
        detectionmodel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device=device)
        detectionmodel.eval()
        #register hook
        activation_value='roi_heads.mask_head.mask_fcn4' #TODO: object fill automatic
        detectionmodel.roi_heads.mask_head.mask_fcn4.register_forward_hook(get_activation(activation_value))
        #modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_maskrcnnfeatures_opt2_same_notpre_30.pth').to(device=device)
   
    if detectionpipeline=="detectors":

        confg_file="/home/jawad/codes/MaskUno/detectors_v2.py"
        checkpoint_file="/home/jawad/Downloads/detectors_htc_r50_1x_coco-329b1453.pth"
        detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
        for name, param in detectionmodel.named_parameters():
            print(name)
        activation_value='roi_head.mask_head.2.convs.3.conv'
        detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
        detectionmodel.eval()
        #modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_detectorsv2features_opt3_same_thr0.1_fixedcounter_editgt_30epochs.pth').to(device=device)
    
    elif detectionpipeline=="mask" :

        confg_file="/home/jawad/codes/MaskUno/mask_config.py"
        checkpoint_file="/home/jawad/mmdetection/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"
        detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
        activation_value='roi_head.mask_head.convs.3.conv'
        for name, param in detectionmodel.named_parameters():
            print(name)
        detectionmodel.roi_head.mask_head.convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.convs.3.conv'))
        detectionmodel.eval()
        #modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_maskrcnnmmdetfeatures_opt3_same_thr0.1_fixedcounter_editgt_epoch5.pth').to(device=device)

    elif detectionpipeline=="htc" :
        print("htc")

        confg_file="/home/jawad/mmdetection/myconfigs/htc.py"
        checkpoint_file="/home/jawad/Downloads/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth"
        detectionmodel=mmdet.apis.init_detector(confg_file,checkpoint_file,device="cuda")
        activation_value='roi_head.mask_head.2.convs.3.conv'

        for name, param in detectionmodel.named_parameters():
            print(name)
        detectionmodel.roi_head.mask_head[2].convs[3].conv.register_forward_hook(get_activation('roi_head.mask_head.2.convs.3.conv'))
        detectionmodel.eval()
        #modelClass=torch.load('/home/jawad/codes/MaskUno/models/model_cat_htc_features_opt3_same_thr0.1_fixedcounter_editgt_epochs30.pth').to(device=device)
    modelClass=model
        

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        if detectionpipeline=="normal":
            images = list(img.to(device) for img in images)
        original_image_sizes = [img.shape[-2:] for img in images]
        #TODO: the comparison should be at the level of original
        #  
        #trans=GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
        #images,targets=trans(images,targets)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        if flag==-1: #detection
            print("here")
            model_detection.eval()
            out_detection=model_detection(images)
            outputs=multiout_2_unoout(out_detection,classid=classid,mask_bool=False)
        

        if flag==0: #finetuned maskrcnn
            #plot the input image before processing 
            '''
            print("plot the input image before processing in falg 0")
            print(f'image 0 shape {images[0].shape}')
            plt.imshow(images[0].cpu().numpy().transpose(1, 2, 0))
            plt.show()
            '''
            outputs = model(images)
            #plot the image after processing 

  
        if flag==1:#for maskrcnn eval
            outputs = model(images)
            outputs=multiout_2_unoout(outputs,classid=classid,mask_bool=True) #17 cat 18 dog 2 bicycle
            
        if flag==2: # with bbx from gt
            '''
            print("in flag two , resize is needed for thier trans")
            #plot the input image before processing 
            print("plot the input image before processing in falg 2")
            print(f'image 0 shape {images[0].shape}')
            plt.imshow(images[0].cpu().numpy().transpose(1, 2, 0))
            plt.show()
            '''
            
            trans=GeneralizedRCNNTransformMy(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
            images1,targets1=trans(images,targets)
            #print(images1.shape)
            images1=images1.tensors
            
            '''
            print("after")
            print(images1[0].cpu().numpy().transpose(1, 2, 0).shape)
            plt.imshow(images1[0].cpu().numpy().transpose(1, 2, 0))
            plt.show()
            '''
            outputs=inference_real_engine_compatible(model,images1,targets1,projection_type='roia_same',original_image_sizes=original_image_sizes)
            #outputs=inference_real_engine_compatible(model,images,targets,projection_type='roia_same',original_image_sizes=original_image_sizes)


        if flag==3: # with bbx from fasterrcnn
            #trans=GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=32, fixed_size=None)
            #images1,targets1=trans(images,targets)
            outputs=inference_real_engine_compatible_withfaster(model_detection=model_detection,model_mask=model,images=images,targets=targets,classid=classid,projection_type='roia_same',original_image_sizes=original_image_sizes)
        
        if flag==4: #pipeline unified

            outputs=quantitative(images,detectionmodel,activation,classid,modelClass,activation_value,detectionpipeline,device)
            print(f'outputs {outputs}')


        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for t in outputs:
            for k, v in t.items():
                print(f'the v {v}')

        model_time = time.time() - model_time



        #from target take image id
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
