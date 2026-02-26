#!/bin/bash

# 测试 xDiT 服务器

SERVER_URL=${1:-"http://localhost:6000"}

echo "Testing xDiT Flux HTTP Server at $SERVER_URL"
echo "================================================"

# 1. 健康检查
echo "1. Health Check..."
curl -X GET "$SERVER_URL/health"
echo -e "\n"

# 2. 生成测试图片
echo "2. Generate test image..."
curl -X POST "$SERVER_URL/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a cute rabbit",
           "num_inference_steps": 20,
           "height": 1024,
           "width": 1024,
           "seed": 42,
           "cfg": 3.5,
           "save_disk_path": "results"
         }' \
     | jq '.'

echo -e "\n"
echo "================================================"
echo "Test completed!"


