import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from Model.pvt_v2 import pvt_v2_b3
from collections import OrderedDict
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM
 
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default="/dataset/Dset_Jerry/ChestXray14/images/00011355_027.png",
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
 
    parser.add_argument('--method', type=str, default='gradcam++',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
 
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
 
    return args
 
def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0), 
        height, width, tensor.size(2))
 
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
 
if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
        
    """
 
    args = get_args()
    methods = \
        {"gradcam": GradCAM, 
         "scorecam": ScoreCAM, 
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}
 
    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
 
    '''Swin'''
    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    # model.eval()
    # print(model)
    # if args.use_cuda:
    #     model = model.cuda()
    #
    # target_layer = model.layers[-1].blocks[-2].norm1
    # print(target_layer )
    # print(model["norm"])
 
    #print(model.layers[-1])
    # print(model.layers[-1].blocks[-2])
    # print(target_layer)
 
    model = pvt_v2_b3()
    # original saved file with DataParallel
    state_dict = torch.load("/datasets/Dset_Jerry/CXR14/PVT_BCE_32/PVT_2.pkl")  # 模型可以保存为pth文件，也可以为pt文件。
    # create new OrderedDict that does not contain module.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
 
    # load params
    model.load_state_dict(new_state_dict)  # 从新加载这个模型。
    if args.use_cuda:
        model = model.cuda().eval()
 
    #print(model)
    #target_layer = model.block4[-1].norm1
    target_layer =model.norm4
    #print(target_layer)
 
 
 
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")
 
    cam = methods[args.method](model=model, 
                               target_layer=target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)
 
    rgb_img = cv2.imread(args.image_path, 1)
    rgb_img = cv2.resize(rgb_img, (384, 384))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5])
 
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 1
    print(target_category)
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
 
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)
 
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'PVT_{args.method}_cam_00011355_027.jpg', cam_image)