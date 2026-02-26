curl -X POST "http://0.0.0.0:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "日本雌性车力巨人",
           "num_inference_steps": 20,
           "height": 1080,
           "width": 1920,
           "seed": 42,
           "cfg": 4.0 
         }'

curl -X POST "http://0.0.0.0:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a ragdoll cat sitting in front of a big house",
           "num_inference_steps": 50,
           "height": 1024,
           "width": 1024,
           "seed": 42,
           "cfg": 4.0,
           "save_disk_path": "results"
         }'
