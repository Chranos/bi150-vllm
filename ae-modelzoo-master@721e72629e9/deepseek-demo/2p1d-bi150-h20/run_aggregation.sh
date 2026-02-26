python3 -m sglang.srt.disaggregation.mini_lb --prefill http://192.168.0.40:12116 \
http://192.168.0.39:12117 --decode http://192.168.0.42:12350 --host 0.0.0.0 --port 12349 \
--prefill-bootstrap-ports 8116 8117
