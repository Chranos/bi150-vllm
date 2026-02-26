

curl -X POST "http://0.0.0.0:6000/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
           "num_inference_steps": 40,
           "height": 720,
           "width": 1280,
           "num_frames": 8,
           "seed": 42,
           "cfg": 5.0, 
           "save_disk_path": "results"
         }'