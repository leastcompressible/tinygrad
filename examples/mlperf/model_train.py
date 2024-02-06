from tinygrad import Tensor, Device, TinyJit, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad.nn import optim

from tqdm import trange

def train_resnet():
  from extra.models.resnet import ResNet50
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from extra.lr_scheduler import CosineAnnealingLR

  GPUS = 1
  BS = getenv("BS", 64)
  EPOCHS = getenv("EPOCHS", 60)

  train_steps_per_epoch = getenv("TRAIN_STEPS_PER_EPOCH", (len(get_train_files()) // BS) - 1)

  num_classes = 1000
  model = ResNet50(num_classes)

  optimizer = optim.SGD(nn.state.get_parameters(model), lr=0.001*BS*GPUS, momentum=0.875, weight_decay=1/2**15)
  lr_schedule = CosineAnnealingLR(optimizer, EPOCHS*train_steps_per_epoch)
  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

  @TinyJit
  def train_step(x:Tensor, y:Tensor):
    x = x.permute([0,3,1,2]).cast(dtypes.float32) / 255.0
    x -= input_mean
    x /= input_std
    with Tensor.train():
      probs = model(x)
      loss = probs.sparse_categorical_crossentropy(y)
      loss.backward()
      optimizer.step()
      lr_schedule.step()
      acc = ((probs.argmax(-1) == y).mean()*100)
    return loss.realize(), acc.realize()

  # time BEAM=8 TRAIN_STEPS_PER_EPOCH=1 MODEL=resnet HIP_VISIBLE_DEVICES=3 python examples/mlperf/model_train.py
  for _ in trange(EPOCHS):
    iterator = iter(batch_load_resnet(batch_size=BS, val=False, shuffle=train_steps_per_epoch>1))

    for _ in trange(train_steps_per_epoch):
      x, y, c = next(iterator)
      x = x.to(Device.DEFAULT)
      y = Tensor(y, dtype=dtypes.int32, device=Device.DEFAULT)
      loss, acc = train_step(x, y)
    print(loss.numpy(), acc.numpy())

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


