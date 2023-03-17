import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import chain
import queue
import collections
import threading
from models import resnetbackbone
from torch.nn.modules.batchnorm import _BatchNorm
try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None



_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])
__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']
class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)#将张量扩展为参数x的大小

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnetbackbone.ResNet101(output_stride, BatchNorm)
    else:
        raise NotImplementedError

class Connect(nn.Module):
    def __init__(self, num_classes, num_neighbor, BatchNorm, reduction=3):
        super(Connect, self).__init__()

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1))

        self.connect_branch = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, num_neighbor, 3, padding=1, dilation=1),
                                            )
        self.se = SELayer(num_neighbor, reduction)

        self.connect_branch_d1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3),
                                               )
        self.se_d1 = SELayer(num_neighbor, reduction)


        self._init_weight()


    def forward(self, input):
        seg = self.seg_branch(input)

        con = self.connect_branch(input)
        con0 = self.se(con)

        con_d1 = self.connect_branch_d1(input)
        con1 = self.se_d1(con_d1)

        return torch.sigmoid(seg), con0, con1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_connect(num_classes, num_neighbor, BatchNorm):
    return Connect(num_classes, num_neighbor, BatchNorm)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            in_inplanes = 256
        else:
            raise NotImplementedError

        self.decoder4 = DecoderBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = DecoderBlock(512, 128, BatchNorm)
        self.decoder2 = DecoderBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = DecoderBlock(128, 64, BatchNorm, inp=True)

        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        self._init_weight()


    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)
def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)
def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)
def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)
# class ASPP(nn.Module):
#     def __init__(self, backbone, output_stride, BatchNorm=nn.BatchNorm2d):
#         super(ASPP, self).__init__()
#         if backbone == 'drn':
#             inplanes = 512
#         elif backbone == 'mobilenet':
#             inplanes = 320
#         else:
#             inplanes = 2048
#         if output_stride == 16:
#             dilations = [1, 6, 12, 18]
#         elif output_stride == 8:
#             dilations = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError

#         self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
#                                              nn.BatchNorm2d(256),
#                                              nn.ReLU())
#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()

#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         return self.dropout(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2. / n))
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channel=2048):
        depth = 256
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net


# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                             stride=1, padding=padding, dilation=dilation, bias=False)
#         self.bn = BatchNorm(planes)
#         self.relu = nn.ReLU()

#         self._init_weight()

#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)

#         return self.relu(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'

        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                                                     track_running_stats=track_running_stats)

        if not self.track_running_stats:
            import warnings
            warnings.warn('track_running_stats=False is not supported by the SynchronizedBatchNorm.')

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class CoANet(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=8, num_classes=4, num_neighbor=9,
                 use_synch_bn=False, freeze_bn=False,**_):
        super(CoANet, self).__init__()

        if use_synch_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.aspp = ASPP(in_channel=2048)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.connect = build_connect(num_classes, num_neighbor, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        e1, e2, e3, e4 = self.backbone(input)
        e4 = self.aspp(e4)
        x = self.decoder(e1, e2, e3, e4)
        seg, connect, connect_d1 = self.connect(x)

        # return seg, connect, connect_d1
        return seg
    

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

























    def get_backbone_params(self):
        return chain(self.parameters())

    def get_decoder_params(self):
        return []

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_2x_lr_params(self):
        modules = [self.aspp, self.decoder, self.connect]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p




