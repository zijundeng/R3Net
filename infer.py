import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from model import R3Net

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'R^3Net'

args = {
    'snapshot': '6000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}


def main():
    net = R3Net().cuda()

    print
    'load snapshot \'%s\' for testing' % args['snapshot']
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()

    results = {}

    for name, root in to_test.iteritems():

        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()

        img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
        for idx, img_name in enumerate(img_list):
            print
            'predicting for %s: %d / %d' % (name, idx + 1, len(img_list))
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

            img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
            img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
            prediction = net(img_var)
            prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))

            if args['crf_refine']:
                prediction = crf_refine(np.array(img), prediction)

            gt = np.array(Image.open(os.path.join(root, img_name + '.png')).convert('L'))
            precision, recall, mae = cal_precision_recall_mae(prediction, gt)
            for pidx, pdata in enumerate(zip(precision, recall)):
                p, r = pdata
                precision_record[pidx].update(p)
                recall_record[pidx].update(r)
            mae_record.update(mae)

            if args['save_results']:
                Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                    exp_name, name, args['snapshot']), img_name + '.png'))

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print
    'test results:'
    print
    results


if __name__ == '__main__':
    main()
