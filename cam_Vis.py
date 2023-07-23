import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
import matplotlib.pyplot as plt
from model import *


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='vis_figures/1.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', default=True,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--resume', type=str, default='experiments/CIFAR10/kd_nasty_resnet18/nasty_resnet18_FixedFC/best_model.tar', 
                        help='path of resume model')
    parser.add_argument('--model_arch', type=str, default='resnet18',
                        help='path of resume model')
    parser.add_argument('--target_layers', type=str, default='model.layer1',
                        help='path of resume model')
    parser.add_argument('--num_class', type=int, default=10,
                        help='num_class in dataset')

    args = parser.parse_args()
    print('- Load checkpoint from {}'.format(args.resume))
    args.model = torch.load(args.resume, map_location=lambda storage, loc: storage)

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def plot_logitsDistri(output_teacher_batch, labels_batch):
    fig, axs = plt.subplots(nrows=len(output_teacher_batch)//8, ncols=8, figsize=(65, 40))
    for i, logits in enumerate(output_teacher_batch):
        row = i // 8
        col = i % 8
        axs[row, col].bar(np.arange(len(logits)), logits.cpu().numpy(), color='blue')  # arguments are passed to np.histogram
        axs[row, col].set_title("Nasty teacher logits distribution")
        axs[row, col].set_xlabel("Logits")
        axs[row, col].set_ylabel("Value")
        axs[row, col].axvline(x=labels_batch[i].cpu().numpy(), color='red')  # add a vertical line at the position of the label
    plt.savefig('nasty_teacher_logits_distbution.png')


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # model = models.resnet50(pretrained=True)
    if args.model_arch == 'resnet18':
        model = ResNet18(num_class=args.num_class)
    else:
        raise "Not support model architecture"
    model.load_state_dict(args.model['state_dict'])

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [eval(args.target_layers)]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb) 

    # Resize the images to the same size
    cam_image = cv2.resize(cam_image, (400, 400))
    gb = cv2.resize(gb, (400, 400))
    cam_gb = cv2.resize(cam_gb, (400, 400))

    # Create a blank image with the same size as the three images
    combined_img = np.zeros((400, 1200, 3), dtype=np.uint8)

    # Copy the three images into the blank image
    combined_img[:, :400, :] = cam_image
    combined_img[:, 400:800, :] = gb
    combined_img[:, 800:, :] = cam_gb
    cv2.imwrite(f'combined_3image_{args.target_layers}.jpg', combined_img)
