from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import monai
import cc3d

from utils import *

class InferenceRunner:
    def __init__(self) -> None:
        self.model = monai.networks.nets.SegResNetDS(
            spatial_dims=2,
            init_filters=32,
            in_channels=1,
            out_channels=7,
            act='relu',
            norm='batch',
            blocks_down=(1, 2, 2, 2, 4),
            dsdepth=4,
            upsample_mode='deconv'
        )
        self.model.load_state_dict(torch.load('../assets/best_segmentator.ckpt', map_location='cpu'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval().to(self.device)

        self.clip_range = [-150, 300]
        self.shape_divider = 32

    def preprocessing(
            self, 
            init_image: np.ndarray, 
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        model_input = torch.from_numpy(init_image).float()
        model_input = ((model_input - self.clip_range[0]) / \
              (float(self.clip_range[1]) - float(self.clip_range[0]))).clip(0,1)
        model_input, pad = divisible_pad(model_input, self.shape_divider)
        return model_input, pad
    
    def inference(
            self,
            init_image: np.ndarray, 
    ) -> torch.Tensor:
        model_input, pad = self.preprocessing(init_image)
        with torch.no_grad():
            logits = self.model(model_input.unsqueeze(0).unsqueeze(0).to(self.device))
        mask = torch.argmax(logits, dim=1, keepdim=True).float().cpu()
        return self.postprocessing(mask, pad)
    
    def postprocessing(
            self,
            mask: torch.Tensor,
            pad: List[List[int]]
    ) -> np.ndarray:
        pad_x, pad_y = pad

        mask_origin = mask[
            0,
            0,
            pad_x[0]:mask.shape[2] - pad_x[1],
            pad_y[0]:mask.shape[3] - pad_y[1],
        ].numpy().astype('uint8')
        
        labels_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 6,
            4: 7,
            5: 8,
            6: 9
        }
        map_func = np.vectorize(lambda x: labels_mapping.get(x, x))
        mask_mapped = map_func(mask_origin)
        kidney_components, k = cc3d.connected_components((mask_mapped==2).astype('uint8'), return_N=True)

        mask_final = np.where(mask_mapped==2, 0, mask_mapped)
        for i in range(1,k+1):
            comp = (kidney_components==i).astype('uint8')
            kidney_label = single_kidney_localization(comp)
            mask_final += kidney_label*comp
        return mask_final

    def __call__(
        self, 
        init_image: np.ndarray,
    ) -> np.ndarray:
        mask_out = self.inference(init_image)
        return mask_out