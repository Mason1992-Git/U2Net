import onnxruntime
import os
from PIL import Image,ImageDraw
import torch
import torchvision

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    img_path = r"d:\DRIVE\test\images"
    onnx_save_path = r"onnx_saved\u2net_eye.onnx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for i, _name in enumerate(os.listdir(img_path)):
        _img = Image.open(os.path.join(img_path, _name))
        w, h = _img.size
        black = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        max_size = max(w, h)
        ratio = 256 / max_size
        img = _img.resize((int(w * ratio), int(h * ratio)))
        black.paste(img, (0, 0, int(w * ratio), int(h * ratio)))
        input = torch.unsqueeze(transform(black), dim=0)
        input = input.to(device)

        ort_session = onnxruntime.InferenceSession(onnx_save_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        out = ort_session.run(None, ort_inputs)

        out = torch.tensor(out)
        x = out[0].cpu().clone().squeeze(0)
        img1 = torchvision.transforms.ToPILImage()(x)
        img1.show()
