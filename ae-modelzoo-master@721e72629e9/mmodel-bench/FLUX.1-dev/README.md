## FLUX

1.拉取指定镜像进行测试：docker pull 10.208.17.169:80/lpf_test/mmodel-flux:v1

or 搭建测试环境：需要指定以下包的版本：

```
xfuser                            0.4.3.post2+corex.4.4.0.rc.7.20250924
diffusers                         0.35.1
numpy                             1.26.4
yunchang                          0.6.3.post1
```

### Flux 测试步骤

```
cd mmodel-bench/FLUX.1-dev
bash run_bf16.sh /data3t/ckpt/black-forest-labs/FLUX.1-dev
bash run_int8.sh /data3t/ckpt/black-forest-labs/FLUX.1-dev
```
