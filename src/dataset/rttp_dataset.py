from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from src.model.decoder.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def read_img_to_tensor(name):
    """return image as a normalized float32 tensor in (C,W,H)"""
    img = np.array(Image.open(name))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = 2 * (img_tensor / 255.0) - 1.0
    return img_tensor

def read_gt_img_to_tensor(name):
    img = np.array(Image.open(name))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = img_tensor / 255.0
    return img_tensor

def read_depth_to_tensor(name):
    """return depth as a float16 tensor in (C,W,H) and in cm"""
    depth_tensor = torch.load(name).get('depth')
    return depth_tensor.unsqueeze(dim=0)

def read_calib_to_tensor(name, cam_name):
    """return calib as a dict of float64 tensors"""
    calib = np.load(name, allow_pickle=True).item().get(cam_name)
    calib_tensor = {key: torch.from_numpy(value) for key, value in calib.items()}
    return calib_tensor


class RttpDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        self.data_root = opt.data_root
        if self.phase == 'train':
            self.trial_names = opt.train_trial_names
        elif self.phase == 'val':
            self.trial_names = opt.val_trial_names
        elif self.phase == 'test':
            self.trial_names = opt.test_trial_names

        self.img_path = os.path.join(self.data_root, '%s/%s/color/%s.png') # data_root/trial_names/cam/color/xxx.png
        self.depth_path = os.path.join(self.data_root, '%s/%s/depth/%s.pt') # data_root/trial_names/cam/depth/xxx.pt
        self.calib_path = os.path.join(self.data_root, '%s/calib.npy') # data_root/setup/calib.npy
        self.bg_path = os.path.join(self.data_root, '%s/background/%s/color/00000.png') # data_root/setup/background/cam/color/00000.png
        self.bg_depth_path = os.path.join(self.data_root, '%s/background/%s/depth/00000.pt')  # data_root/setup/background/cam/color/00000.png
        self.sample_list = self.get_sample_list()


    def get_sample_list(self):
        """get a list of [trial_name, frame_num]"""
        list = []
        for trial_name in self.trial_names:
            img_folder = os.path.join(self.data_root, trial_name, self.opt.source_cam_names[0][0], 'color')
            frames = sorted(os.listdir(img_folder))
            for frame in frames:
                list.append([trial_name, frame[:-4]])
        return list

    def load_multiview_data(self, sample_name):
        multiview_data = {'name' : sample_name}
        for cam_name in self.opt.source_cam_names:
            img_tensor, gt_img_tensor, depth_tensor, calib_tensor, bg_tensor, bg_depth_tensor= \
                self.load_single_view_tensor(sample_name, cam_name)

            # NOTE: downsample because image too big for attention matrix
            if self.opt.downscale_factor != 1.0:
                calib_tensor["K_RGB"][:2] /= self.opt.downscale_factor
                img_tensor = F.interpolate(img_tensor[None], scale_factor=1 / self.opt.downscale_factor, mode='bilinear')[0]

            height, width = img_tensor.shape[1:]
            extr = torch.eye(4)
            extr[:3] = torch.cat((calib_tensor["R_to_ref_RGB"].transpose(1, 0),
                              -calib_tensor["R_to_ref_RGB"].transpose(1, 0) @ calib_tensor["T_to_ref_RGB"]), dim=1)
            calib_tensor["K_RGB"][0] /= width
            calib_tensor["K_RGB"][1] /= height
            intr = calib_tensor["K_RGB"]


            # R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            # T = np.array(extr[:3, 3], np.float32)

            # FovX = focal2fov(intr[0, 0], width)
            # FovY = focal2fov(intr[1, 1], height)
            # projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
            # world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
            # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)  # (Rt)^T @ K^T
            # camera_center = world_view_transform.inverse()[3, :3]

            multiview_data[cam_name[1]] = {
                'img'   : img_tensor,
                'gt_img': gt_img_tensor,
                'depth' : depth_tensor,
                'bg'  : bg_tensor,
                'bg_depth': bg_depth_tensor,
                'extr': extr.float(),
                'intr': intr.float(),
                # 'FovX': FovX,
                # 'FovY': FovY,
                'width': width,
                'height': height,
                # 'world_view_transform': world_view_transform,
                # 'full_proj_transform': full_proj_transform,
                # 'camera_center': camera_center
            }
            # multiview_data[cam_name[1]].update(calib_tensor)

        return multiview_data

    def load_single_view_tensor(self, sample_name, cam_name, novel_view=False):
        img_name = self.img_path % (sample_name[0], cam_name[0], sample_name[1])
        calib_name = self.calib_path % (sample_name[0].split('/')[0])
        bg_name = self.bg_path % (sample_name[0].split('/')[0], cam_name[0])
        bg_depth_name = self.bg_depth_path % (sample_name[0].split('/')[0], cam_name[0])

        if novel_view:
            return read_gt_img_to_tensor(img_name), \
                read_calib_to_tensor(calib_name, cam_name[0]), \
                read_gt_img_to_tensor(bg_name), \
                read_depth_to_tensor(bg_depth_name)

        depth_name = self.depth_path % (sample_name[0], cam_name[0], sample_name[1])
        return read_img_to_tensor(img_name), \
            read_gt_img_to_tensor(img_name), \
            read_depth_to_tensor(depth_name), \
            read_calib_to_tensor(calib_name, cam_name[0]), \
            read_gt_img_to_tensor(bg_name), \
            read_depth_to_tensor(bg_depth_name)


    def get_novel_view_tensor(self, sample_name, cam_name):
        img, calib, bg, bg_depth = self.load_single_view_tensor(sample_name, cam_name, novel_view=True)

        # NOTE: downsample because image too big for attention matrix
        if self.opt.downscale_factor != 1.0:
            calib["K_RGB"][:2] /= self.opt.downscale_factor
            img = F.interpolate(img[None], scale_factor=1/self.opt.downscale_factor, mode='bilinear')[0]
        height, width = img.shape[1:]

        extr = torch.eye(4)
        extr[:3] = torch.cat((calib["R_to_ref_RGB"].transpose(1, 0),
                          -calib["R_to_ref_RGB"].transpose(1, 0) @ calib["T_to_ref_RGB"]), dim=1)
        calib["K_RGB"][0] /= width
        calib["K_RGB"][1] /= height
        intr = calib["K_RGB"]
        # R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        # T = np.array(extr[:3, 3], np.float32)
        #
        # FovX = focal2fov(intr[0, 0], width)
        # FovY = focal2fov(intr[1, 1], height)
        # projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        # world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0) # (Rt)^T @ K^T
        # camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'img': img * 2. - 1.0,
            'gt_img': img,
            'bg'  : bg,
            'bg_depth': bg_depth,
            'extr': extr.float(),
            'intr': intr.float(),
            # 'FovX': FovX,
            # 'FovY': FovY,
            'width': width,
            'height': height,
            # 'world_view_transform': world_view_transform,
            # 'full_proj_transform': full_proj_transform,
            # 'camera_center': camera_center
        }
        # novel_view_data.update(calib)
        return novel_view_data


    def get_item(self, index):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]
        dict_tensor = self.load_multiview_data(sample_name)
        # TODO: put novel view choice here instead
        for gt_cam_name in self.opt.gt_cam_names:
            novel_view_data = self.get_novel_view_tensor(sample_name, gt_cam_name)
            dict_tensor.update({
                gt_cam_name[1] : novel_view_data
            })

        return dict_tensor


    def get_test_item(self, index, source_id):
        pass


    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index)
        elif self.phase == 'val':
            return self.get_item(index)
        else:
            return self.get_item(index)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)