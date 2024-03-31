import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm


class Teacher(nn.Module):
    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train, args):
        super().__init__()
        self.solver = solver
        self.generator = generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.bn_loss = args.bn_loss
        self.noise = args.noise
        self.ie_loss = args.ie_loss
        self.act_loss = args.act_loss
        self.w_ie = args.w_ie
        self.w_act = args.w_act
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)
        self.first_time = train
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").to('cuda')
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        if self.bn_loss:
            loss_r_feature_layers = []
            for module in self.solver.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm):
                    loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
            self.loss_r_feature_layers = loss_r_feature_layers

    def sample(self, size, return_scores=False):
        self.solver.eval()
        self.generator.train()
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1)
        self.generator.eval()
        with torch.no_grad():
            x_i = self.generator.sample(size, 'cuda')
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]
        _, y = torch.max(y_hat, dim=1)
        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):
        self.solver.eval()
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]
        _, y = torch.max(y_hat, dim=1)
        return (y, y_hat) if return_label else y_hat

    def generate_scores_pen(self, x):
        self.solver.eval()
        with torch.no_grad():
            y_hat = self.solver.feature_extractor(x)
        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1):
        torch.cuda.empty_cache()
        self.generator.train()
        self.generator.to('cuda')
        for epoch in tqdm(range(epochs)):
            inputs = self.generator.sample(bs, 'cuda')
            self.gen_opt.zero_grad()
            self.solver.zero_grad()
            bn_loss = 0
            # content
            if self.act_loss:
                features_t = self.solver.feature(inputs)
                outputs = self.solver.fc(features_t)[:, :self.num_k]
                loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
                loss += - features_t[-1].abs().mean() * self.w_act
            else:
                outputs = self.solver(inputs)[:, :self.num_k]
                ce_loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
                loss = ce_loss

            if self.ie_loss:
                softmax_o_T = F.softmax(outputs, dim=1).mean(dim=0)
                ie_loss = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum()) * self.w_ie
                loss += ie_loss
            # R_feature loss
            if self.bn_loss:
                for mod in self.loss_r_feature_layers:
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    bn_loss = bn_loss + loss_distr
                loss += bn_loss

            # image prior
            if self.noise:
                inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
                loss_var = self.mse_loss(inputs, inputs_smooth).mean()
                noise_loss = self.di_var_scale * loss_var
                loss += noise_loss
            loss.backward()
            self.gen_opt.step()
        torch.cuda.empty_cache()
        self.generator.eval()


class DeepInversionFeatureHook():
    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        x = module.running_var.data.type(var.type())
        y = module.running_mean.data.type(var.type())
        m1 = torch.log(var**(0.5) / (x + 1e-8)**(0.5)).mean()
        r_feature = m1 - 0.5 * (1.0 - (x + 1e-8 + (y - mean)**2) / var).mean()
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class Gaussiansmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to('cuda')
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)
