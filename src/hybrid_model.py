# CycleGAN 的网络本质上是双向的 pix2pix 网络（即两个生成器和两个判别器）。
# 因此，网络设计不需要额外修改，可以直接复用 CycleGAN_Turbo 和 Pix2Pix_Turbo 的网络结构