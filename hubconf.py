import torch.nn
import torchvision.transforms as transforms
from PIL import Image
from ssd_PyT.src import utils, model
from torch.utils.model_zoo import load_url


def nvidia_ssd_pyt(pretrained=True, *args, **kwargs):
    m = model.SSD300()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_SSD_FP32_PyT-1ec96418.pt'
        ckpt = load_url(checkpoint, progress=False)
        m.load_state_dict(ckpt['model'])
    return m


def prepare(img):
    size = (300, 300)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans_val = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # ToTensor(),
        normalize, ])
    return trans_val(img)


if __name__ == '__main__':

    model = nvidia_ssd_pyt()
    model.eval()
    test_im = r'C:\datasets\coco\train2017\000000000049.jpg'
    test_im = r'C:\datasets\coco\train2017\000000001332.jpg'
    test_im = r'C:\datasets\coco\train2017\000000001622.jpg'
    img = Image.open(test_im)
    inp = prepare(img)
    inp = inp.expand(1, 3, 300, 300)
    out = model.forward(inp)
    out_boxes = out[0]  #[1:].transpose(1,0)
    out_labels = out[1] #[1:].transpose(1,0)

    enc = utils.Encoder(utils.dboxes300_coco())

    boxes, labels, confidences = enc.decode_batch(out_boxes, out_labels, max_output=10)[0]

    if False:
        draw = ImageDraw.Draw(img)
        for box, label, conf in zip(boxes, labels, confidences):
            print(category_map[label.item()])
            print(label.item())
            draw.line((box[0]*300, box[1]*300, box[2]*300, box[3]*300), fill=128)
        img.show()
    else:
        import os
        class Args:
            pass
        args = Args()
        args.distributed = False
        args.num_workers = 1
        args.eval_batch_size = 2
        args.data = 'c:/datasets/coco'

        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

        val_coco = utils.COCODetection(val_coco_root, val_annotate, None)

        label_map = val_coco.label_map
        label_info = val_coco.label_info
        utils.draw_patches(img, boxes.detach(), labels, order="ltrb", label_map=label_info)
