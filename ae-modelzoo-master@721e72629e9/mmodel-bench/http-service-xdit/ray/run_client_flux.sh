curl -X POST "http://0.0.0.0:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a cute rabbit",
           "num_inference_steps": 50,
           "height": 1024,
           "width": 1024,
           "seed": 42,
           "cfg": 3.5, 
           "save_disk_path": "results"
         }'
