import requests
import base64
import io
import numpy as np
from PIL import Image
import torch
import time
import traceback

# 每个类都是comfyUI界面的一个节点
class XDiTFluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A beautiful landscape with mountains and a lake",
                    "multiline": True
                }),
                "server_url": ("STRING", {
                    "default": "http://localhost:6000"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "xDiT"
    OUTPUT_NODE = False
    
    def generate(self, prompt, server_url, num_inference_steps, height, width, cfg, seed):
        """调用 HTTP 服务生成图片"""
        
        print(f"[xDiT] 开始生成图片...")
        print(f"[xDiT] Prompt: {prompt[:50]}...")
        print(f"[xDiT] Server: {server_url}")
        print(f"[xDiT] Steps: {num_inference_steps}, Size: {width}x{height}")
        
        try:
            start_time = time.time()
            request_data = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "cfg": cfg,
                "seed": seed,
                "save_disk_path": None
            }

            response = requests.post(
                f"{server_url}/generate",
                json=request_data,
                timeout=600
            )
            
            if response.status_code != 200:
                raise response.text
            
            result = response.json()
            
            # base64 图片解码 
            img_data = base64.b64decode(result["output"])
            image = Image.open(io.BytesIO(img_data))
 
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            elapsed_time = time.time() - start_time
            
            print(f"[xDiT] 服务端耗时: {result['elapsed_time']}")
            print(f"[xDiT] 总耗时: {elapsed_time:.2f}s")
            
            return (image_tensor,)
            
        except requests.exceptions.Timeout:
            raise "请求超时。。。"
        except requests.exceptions.ConnectionError:
            error_msg = f"无法连接到 xDiT 服务器 {server_url}。请确保服务器正在运行。"
            raise error_msg
        except Exception as e:
            raise traceback.print_exc()


class XDiTFluxBatchNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "default": "A beautiful landscape\nA futuristic city\nA cute cat",
                    "multiline": True
                }),
                "server_url": ("STRING", {
                    "default": "http://localhost:6000"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_batch"
    CATEGORY = "xDiT"
    OUTPUT_NODE = False
    
    def generate_batch(self, prompts, server_url, num_inference_steps, height, width, cfg, seed):
        """调用 HTTP 服务生成 批量生成图片"""

        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            raise "没有有效的 prompts"
        
        print(f"[xDiT Batch] 开始批量生成 {len(prompt_list)} 张图片...")
        
        images = []
        
        for i, prompt in enumerate(prompt_list):
            print(f"[xDiT Batch] 生成第 {i+1}/{len(prompt_list)} 张: {prompt[:50]}...")
            
            try:
                request_data = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "height": height,
                    "width": width,
                    "cfg": cfg,
                    "seed": seed + i,
                    "save_disk_path": None
                }
                
                response = requests.post(
                    f"{server_url}/generate",
                    json=request_data,
                    timeout=600
                )
                
                if response.status_code != 200:
                    print(f"[xDiT Batch] 第 {i+1} 张生成失败， error ： {response.text}")
                    continue
                
                result = response.json()

                # base64 图片解码 
                img_data = base64.b64decode(result["output"])
                image = Image.open(io.BytesIO(img_data))
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                
                images.append(image_tensor)
                
                print(f"[xDiT Batch] 第 {i+1} 张生成成功，耗时： {result['elapsed_time']}")
                
            except Exception as e:
                print(f"[xDiT Batch] 第 {i+1} 张生成出错: {str(e)}")
                continue
        
        if not images:
            raise "所有图片生成失败"

        batch_tensor = torch.stack(images)
        
        print(f"[xDiT Batch] =批量生成完成! 成功 {len(images)}/{len(prompt_list)} 张")
        
        return (batch_tensor,)


class XDiTServerStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {
                    "default": "http://localhost:6000"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check_status"
    CATEGORY = "xDiT"
    OUTPUT_NODE = True
    
    def check_status(self, server_url):
        
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                status = "服务正常"
                print(f"[xDiT Status] {status}")
                return (status,)
            else:
                status = f"服务器返回异常状态码: {response.status_code}"
                print(f"[xDiT Status] {status}")
                return (status,)
        except requests.exceptions.ConnectionError:
            status = f"无法连接到服务器 {server_url}"
            print(f"[xDiT Status] {status}")
            return (status,)
        except Exception as e:
            status = f"报错: {str(e)}"
            print(f"[xDiT Status] {status}")
            return (status,)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "XDiTFluxNode": XDiTFluxNode,
    "XDiTFluxBatchNode": XDiTFluxBatchNode,
    "XDiTServerStatus": XDiTServerStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XDiTFluxNode": "xDiT Flux (Multi-GPU)",
    "XDiTFluxBatchNode": "xDiT Flux Batch (Multi-GPU)",
    "XDiTServerStatus": "xDiT Server Status",
}


