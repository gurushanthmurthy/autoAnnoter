from pathlib import Path
import cv2
from utils.general import set_logging
from utils.hubconf import custom
import argparse
import glob
import os
from anot_utils import save_yolo, get_BBoxYOLOv7
from utils.torch_utils import select_device


def annotate():

    path_to_dir = args["dataset"]
    path_or_model = args['model']
    img_size = args['size']
    detect_conf = args['confidence']
    out_dir = args['output_dir']
    image_per_sec = args['img_persec']
    image_prefix = args['img_prefix']
    
    # Initialize
    set_logging()

    # Load dataset
    video = cv2.VideoCapture(path_to_dir)    
    fps = video.get(cv2.CAP_PROP_FPS)
    print('frames per second =',fps)
    
    # Directories
    if video:
        save_dir = Path(out_dir, exist_ok=True) 
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)  # make dir
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True) 
        
        image_path = str(save_dir / 'images')  # img.jpg
        label_path = str(save_dir / 'labels')  # label.txt
        
        # Load YOLOv7 Model (best.pt)
        model = custom(path_or_model=path_or_model)  
        # Used as counter variable
        count  = 0
        # checks whether frames were extracted
        success = 1
        if image_per_sec > 0 :
            time_delta = 1000/image_per_sec
        else :
            time_delta = 0
        t_msec = 0
        while success:
            if time_delta > 0 : 
                video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            # video object calls read
            # function extract frames
            success, img = video.read()

            if success:
                # Saves the frames with frame-count
                file_name = image_prefix + str(count)
                cv2.imwrite(image_path + "/" + file_name + ".jpg", img)
                count += 1
                if time_delta > 0 : 
                    t_msec += time_delta
                



            # img_list = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
            #     glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
            #     glob.glob(os.path.join(path_to_dir, '*.png'))

            # for img in img_list:
                # folder_name, file_name = os.path.split(img)
                # img = cv2.imread(img)
                h, w, c = img.shape
                bbox_list, class_list, confidence = get_BBoxYOLOv7(img, model, detect_conf)
                save_yolo(label_path, file_name, w, h, bbox_list, class_list)

                print(f'Successfully Annotated {file_name}')

        print('YOLOv7-Auto_Annotation Successfully Completed for %s', path_to_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, required=True,
                    help="path to dataset/dir")
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to best.pt (YOLOv7) model")
    ap.add_argument("-s", "--size", type=int, default=640,
                    help="Size of image used to train the model")
    ap.add_argument("-c", "--confidence", type=float, default=0.25,
                    help="Model detection Confidence (0<confidence<1)")
    ap.add_argument("-i", "--img-persec", type=float, default=1.0,)
    ap.add_argument("-o", "--output-dir", type=str, default='train',
                    help="Output folder for annonated images and labels")
    ap.add_argument("-p", "--img-prefix", type=str, default='image_',
                    help="image prefix for labels and images")
    args = vars(ap.parse_args())
    print(args)
    # annotate 
    annotate()
