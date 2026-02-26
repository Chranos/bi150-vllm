## 基于 ray 多卡部署
```bash
cd ray
## 启动服务
bash run_serve_flux_bf16.sh
## http post 请求
bash run_client_flux.sh

```
## 基于 torchrun 多卡部署

```bash
cd torchrun
## 启动服务
## 权重路径及参数 config/config_flux.json
bash run_serve_flux_bf16.sh
## http post 请求
bash run_client_flux.sh

```