''' 启动命令
python benmark_serving_qwenvl.py \
    --url http://localhost:8000 --use_v1_api \
    --tokenizer_path ./Qwen2.5-VL-7B-Instruct \
    -c 1 --input_len 512 --input_num 2 --output_len 512 \
    --trust_remote_code \
    --multi_modal_data /gcsp_test/448.png 
'''

import argparse
import json
import random
import time
import base64
import math
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

import numpy as np
import requests
from attrs import Factory, define
from cattrs import unstructure
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

@define
class Metrics:
    concurrency: int = Factory(int)
    total_request_num: int = Factory(int)
    valid_request_num: int = Factory(int)
    valid_ratio: float = Factory(float)
    rps: float = Factory(float)
    total_tps: float = Factory(float)
    input_throughput: float = Factory(float)
    total_throughput: float = Factory(float)
    input_tokens: list[float] = Factory(list)
    output_tokens: list[float] = Factory(list)
    e2e_latency: list[float] = Factory(list)
    ttft: list[float] = Factory(list)
    itl: list[float] = Factory(list)
    tps_per_user: list[float] = Factory(list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url", type=str, required=True, help="The URL endpoint for the POST request."
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="The number of concurrent clients to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The random seed for reproducibility."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="The path to the tokenizer to be used.",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=128,
        help="The length (in tokens) of each input sample.",
    )
    parser.add_argument(
        "--input_num",
        type=int,
        default=10,
        help="The number of input samples to generate.",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=64,
        help="The maximum length (in tokens) of the output.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="If set, trust remote code when loading the tokenizer.",
    )
    parser.add_argument(
        "--use_v1_api",
        action="store_true",
        default=False,
        help="If set, user openai v1 api rather than generate_stream.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="The file to write the metrics to.",
    )
    parser.add_argument(
        "--first_call_tokens",
        action="store_true",
        default=False,
        help="If set, call tokens api before generate_stream.",
    )
    parser.add_argument(
        "--multi_modal_data",
        type=str,
        default="",
        help="The path to the image file for multi-modal testing.",
    )

    return parser.parse_args()

def token_calculate(image_path):
    # 使用with语句确保图片文件正确关闭
    with Image.open(image_path) as image:
        # 获取图片的原始尺寸
        height = image.height
        width = image.width
        
        # 将高度调整为28的整数倍
        h_bar = round(height / 28) * 28
        # 将宽度调整为28的整数倍
        w_bar = round(width / 28) * 28
        
        # 图像的Token下限：4个Token
        min_pixels = 28 * 28 * 4
        # 图像的Token上限：1280个Token
        max_pixels = 1280 * 28 * 28
            
        # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
        if h_bar * w_bar > max_pixels:
            # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
            beta = math.sqrt((height * width) / max_pixels)
            # 重新计算调整后的高度，确保为28的整数倍
            h_bar = math.floor(height / beta / 28) * 28
            # 重新计算调整后的宽度，确保为28的整数倍
            w_bar = math.floor(width / beta / 28) * 28
        elif h_bar * w_bar < min_pixels:
            # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
            beta = math.sqrt(min_pixels / (height * width))
            # 重新计算调整后的高度，确保为28的整数倍
            h_bar = math.ceil(height * beta / 28) * 28
            # 重新计算调整后的宽度，确保为28的整数倍
            w_bar = math.ceil(width * beta / 28) * 28

        # 计算图像的Token数：总像素除以28 * 28, 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各计1个Token）
        token = int((h_bar * w_bar) / (28 * 28)) +2
        print("====== h_bar: ", h_bar, " w_bar: ", w_bar, " token: ", token)

        return h_bar, w_bar, token

def get_random_input_data(
    input_len: int, input_num: int, tokenizer_path: str, multi_modal_data: str
) -> tuple[list[str], list[int], list[str], list[int], list[int]]:
    """
    Generate random input data based on the input length and the number of samples.

    Args:
        input_len: Number of tokens per input
        input_num: Number of input samples
        tokenizer_path: Path to the tokenizer
        use_v1_api: Use v1 api or generate_stream
        multi_modal_data: Path to the image file for multi-modal testing.

    Returns:
        A tuple containing:
            - prompts: A list of generated prompt texts
            - input_lens: A list of input lengths for each prompt
    """
    prompts: list[str] = []
    text_lens: list[int] = []
    images: list[str] = []
    image_lens: list[int] = []
    total_input_lens: list[int] = []

    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer: Any = (
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            tokenizer_path, trust_remote_code=True
        )
    )
    vocab_size: int = tokenizer.vocab_size
    
    base64_image = ""
    image_token_len = 0
    has_image = bool(multi_modal_data)
    print("====== has_image: ", has_image)

    if has_image:
        with open(multi_modal_data, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    else:
        base64_image = ""


    # 估算图像token长度（这里使用一个固定值作为示例）
    _, _, image_token_len = token_calculate(multi_modal_data)
    #assert image_token_len < input_len

    for _ in range(input_num):
        # 决定是否包含图像
        # has_image = random.random() < 0.5 #image_ratio
        if has_image:
            # 如果包含图像，文本长度需要减少图像token的长度
            #text_token_len = input_len - image_token_len - 512
            text_token_len = input_len + image_token_len
            image = base64_image
            images.append(image)
            image_lens.append(image_token_len)
        else:
            # 如果不包含图像，全部长度用于文本
            text_token_len = input_len
            images.append("")
            image_lens.append(0)

        # 生成随机文本
        candidate_ids: list[int] = [
            random.randint(10, vocab_size - 1) for _ in range(text_token_len)
        ]
        candidate_prompt: str = tokenizer.decode(candidate_ids)
        # if not use_v1_api:
        #     candidate_prompt = input_template_type.format(candidate_prompt)
        candidate_prompt = candidate_prompt[:text_token_len] if len(candidate_prompt) > text_token_len else candidate_prompt
        prompts.append(candidate_prompt)
        text_lens.append(len(candidate_prompt))
        total_input_lens.append(text_lens[-1] + image_lens[-1])
        #print("====== text_lens: ", text_lens, " image_lens: ", image_lens, " total_input_lens: ", total_input_lens)


    return prompts, text_lens, images, image_lens, total_input_lens


def get_output_length(input_num: int, output_len: int) -> list[int]:
    """
    Calculate the output token count for each input based on the specified output length.

    Args:
        input_num: Number of input samples
        output_len: Base output token count

    Returns:
        A list of output token counts for each sample
    """
    return [output_len] * input_num


def post_data_decorator(
    func: Callable[..., Any],
    request_gen: Callable[[str, int, bool, str, bool], tuple[dict[str, Any], str]],
    use_v1_api: bool,
    model_path: str,
    first_call_tokens: bool,
) -> Callable[[str, list[str], int], Any]:
    """
    Decorator function that generates request data using request_gen and then calls func to send the request.

    Args:
        func: A function to send the POST request (e.g., signature: func(url: str, request_data: dict, **kwargs))
        request_gen: A function to generate the request data (signature: request_gen(input_data, max_new_tokens, **kwargs))
        use_v1_api: Use v1 api or generate_stream
        model_path: The model path
        first_call_tokens: If True, call tokens api before generate_stream.

    Returns:
        A decorated function accepting (url, input_data, max_new_tokens) as parameters.
    """

    def wrapper(url: str, input_data: tuple[str, str, int], max_new_tokens: int) -> Any:
        request_data, _ = request_gen(
        input_data[0],  # prompt
        input_data[1],  # base64_image
        input_data[2],  # image_lens
        max_new_tokens,
        use_v1_api,
        model_path,
        first_call_tokens
    )
        result: list[list[float]] = func(
            url, request_data, use_v1_api, first_call_tokens
        )
        return result

    return wrapper


def process_stream_response(
    response: requests.Response, text_extractor: Callable[[dict[str, Any]], str]
) -> list[float]:
    """
    Process the streaming response, extracting token text via text_extractor
    and recording the elapsed time for each token.

    Args:
        response: The streaming HTTP response.
        text_extractor: A callable that extracts token text from a decoded JSON object.

    Returns:
        A list of elapsed times for each response token.
    """
    used_time: list[float] = []
    last_time: float = time.time()
    generated_text: str = ""

    for line in response.iter_lines():
        if line:
            # Remove the "data: " prefix if present.
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                line_str = line_str[6:]
            if line_str == "[DONE]":
                continue
            data: dict[str, Any] = json.loads(line_str)
            token_text = text_extractor(data)
            # print("token_text: ", token_text)
            if token_text:
                generated_text += token_text
            current_time: float = time.time()
            used_time.append(current_time - last_time)
            last_time = current_time
    return used_time


def post_stream_vllm(
    url: str, request_data: dict[str, Any], use_v1_api: bool, first_call_tokens: bool
) -> list[float]:
    """
    Send a POST request in streaming mode and record the elapsed time for each token of the response.

    Depending on the use_v1_api flag, use different endpoints and response formats.

    Args:
        url: The base URL.
        request_data: The request payload.
        use_v1_api: If True, use the v1 API; otherwise use the generate_stream mode.

    Returns:
        A list of elapsed times for each response token.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if use_v1_api:
        # For v1 API mode.
        try:
            response = requests.post(
                f"{url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(request_data),
                stream=True,
            )

            print(f"Response status: {response.status_code}")
            
        except Exception as e:
            raise
        assert response.status_code == 200
        # Define a token extractor for v1 API response.
        # extractor: Callable[[dict[str, Any]], Any] = lambda data: data.get(
        #     "choices", [{}]
        # )[0].get("text", "")

        def chat_extractor(data: dict[str, Any]) -> str:
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                return content if content is not None else ""
            return ""
        extractor: Callable[[dict[str, Any]], Any] = chat_extractor
    else:
        # For generate_stream mode, optionally perform a warmup call.
        if first_call_tokens:
            warmup_resp = requests.post(
                f"{url}/tokens",
                json={"text": request_data["inputs"]},
                headers=headers,
                stream=False,
            )
            assert warmup_resp.status_code == 200
        response = requests.post(
            f"{url}/generate_stream",
            headers=headers,
            data=json.dumps(request_data),
            stream=True,
        )
        assert response.status_code == 200
        # Define a token extractor for generate_stream response.
        extractor = lambda data: data.get("token", {}).get("text", "")

    return process_stream_response(response, extractor)


def gen_vllm_request(
    inputs: str,
    base64_image: str,
    image_lens: int,
    max_new_tokens: int,
    use_v1_api: bool,
    model_path: str,
    first_call_tokens: bool,
) -> tuple[dict[str, Any], str]:
    """
    Generate request data based on the input text and maximum new tokens.

    Args:
        inputs: The input prompt text.
        max_new_tokens: Maximum number of new tokens to generate.
        use_v1_api: If True, get the v1 style request; otherwise get the generate_stream style request.
        model_path: The model path.
        first_call_tokens: If True, call tokens api before generate_stream.

    Returns:
        A tuple containing the request data and the prompt.
    """
    prompt: str = inputs
    if use_v1_api:
        assert (
            first_call_tokens == False
        ), "tokens api only support in generate_stream mode."
        messages = []
        if base64_image: # Check if an image is provided
            content = [
                {
                    "type": "text",
                    "text": inputs if inputs else "Describe this image."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": inputs})

        request_data: dict[str, Any] = {
            "model": model_path,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
            "ignore_eos": True, 
            "logprobs": 1,
            "return_tokens_as_token_ids": True
        }

    
    else:
        request_data = {
            "inputs": inputs,
            "parameters": {
                "details": True,
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.05,
                "temperature": 0.8,
                "top_p": 0.7,
                "top_k": 40,
                "watermark": False,
                "ignore_eos": True,
                "token_healing_top_k": None,
                "token_healing_unmerge_last_token": None,
                "stop": ["<|im_end|>"],
            },
            "stream": True,
            "num_beam": 0,
        }

    return request_data, prompt


def get_printable_time(times: list[float], ori: list[float] | None = None) -> list[str]:
    """
    Convert a list of times to a list of formatted strings.

    Args:
        times: A list of times in seconds

    Returns:
        A list of formatted strings
    """
    if ori == None:
        return [f"{t:.2f}" for t in times]
    else:
        return [f"{t:.2f}/{t/o*100:.1f}%" for t, o in zip(times, ori)]


def dump_to_file(file_name: str, data: Metrics) -> None:
    """
    Dump the metrics to a file in JSON format.

    Args:
        file_name: The name of the file to write to
        data: The metrics to write
    """
    if file_name == "":
        return
    with open(file_name, "w") as f:
        json.dump(unstructure(data), f, indent=4)


def main() -> None:
    """
    Main function that parses arguments and sends test requests.

    Args:
        args: Command-line arguments
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    url: str = args.url
    data_percentiles: list[int] = [0, 25, 50, 75, 100]
    metric_percentiles: list[int] = [50, 90, 99, 100]
    use_v1_api: bool = args.use_v1_api
    model_path: str = args.tokenizer_path
    first_call_tokens: bool = args.first_call_tokens

    post_stream: Callable[[str, list[str], int], Any] = post_data_decorator(
        post_stream_vllm, gen_vllm_request, use_v1_api, model_path, first_call_tokens
    )
    prompts, input_lens, base64_image, image_lens, total_input_len = get_random_input_data(
        args.input_len, args.input_num, model_path, args.multi_modal_data
    )
    max_new_tokens: list[int] = get_output_length(len(total_input_len), args.output_len)

    start_time: float = time.time()
    console = Console()

    def post_wrapper(p: tuple[tuple[str, str, int], int]) -> list[float]:
        return post_stream(url, p[0], p[1])

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Pair each prompt with its corresponding max_new_tokens using a lambda
        
        input_tuples_for_zip = []
        for i in range(len(prompts)):
            input_tuples_for_zip.append((prompts[i], base64_image[i], image_lens[i]))

        results: list[list[float]] = list(
            tqdm(
                executor.map(
                    # lambda p: post_stream(url, p[0], p[1]),
                    post_wrapper,
                    zip(input_tuples_for_zip, max_new_tokens),
                ),
                total=len(prompts),
                desc="Running tests",
            )
        )
    end_time: float = time.time()
    # print(results)

    first_token_time: list[float] = []
    decode_token_time: list[float] = []
    request_time: list[float] = []
    final_output_lens: list[int] = []
    tps_per_user: list[float] = []
    valid_num: int = 0

    # 多模态使用chat/completion接口， 首token为role：assitant信息无实际内容，从第二个token开始计算，设置偏移量为1
    token_offset = 1 

    # Collect statistics only for results with at least two decoded tokens
    for result in results:
        if len(result) > 1:
            first_token_time.append(result[0]+result[1])
            decode_token_time.append(sum(result[2:]) / len(result[2:]))
            request_time.append(sum(result))
            final_output_lens.append(len(result)-1)
            valid_num += 1
            tps_per_user.append(len(result) / sum(result))
    
    # Build 2-row sub-tables for the three percentile metrics
    input_values:list[float] = cast(list[float], list(np.percentile(total_input_len, data_percentiles)))
    output_values:list[float] = cast(list[float], list(np.percentile(final_output_lens, data_percentiles)))  
    e2e_values:list[float] = cast(list[float], list(np.percentile(request_time, data_percentiles)))   
    first_values:list[float] = cast(list[float], list(np.percentile(first_token_time, metric_percentiles) * 1000))
    decode_values:list[float] = cast(list[float], list(np.percentile(decode_token_time, metric_percentiles) * 1000))
    tps_values: list[float] = cast(list[float], list(np.percentile(tps_per_user, metric_percentiles) * 1000))

    first_values.append(sum(first_token_time) * 1000 / len(first_token_time))
    decode_values.append(sum(decode_token_time) * 1000 / len(decode_token_time))
    tps_values.append(sum(tps_per_user) * 1000 / len(tps_per_user))

    metrics = Metrics(
        concurrency=args.concurrency,
        total_request_num=len(results),
        valid_request_num=valid_num,
        valid_ratio=valid_num / len(results),
        rps=valid_num / (end_time - start_time),
        total_tps=sum(final_output_lens) / (end_time - start_time),
        input_throughput=sum(total_input_len) / (end_time - start_time),
        total_throughput=(sum(total_input_len) + sum(final_output_lens))
        / (end_time - start_time),
        input_tokens=input_values,
        output_tokens=output_values,
        e2e_latency=e2e_values,
        ttft=first_values,
        itl=decode_values,
        tps_per_user=tps_values,
    )
    dump_to_file(args.output_file, metrics)

    # Build the main summary table without internal row separators, with a highlighted header,
    # and using white for values instead of magenta.
    summary_table = Table(
        title="Benchmark Summary", show_lines=True, header_style="bold"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Concurrency", str(args.concurrency))
    summary_table.add_row("Total Request Num", str(len(results)))
    summary_table.add_row("Valid Request Num", str(valid_num))
    summary_table.add_row("Valid Ratio", f"{valid_num / len(results):.2f}")
    summary_table.add_row("RPS", f"{valid_num / (end_time - start_time):.2f}")
    summary_table.add_row(
        "Total TPS", f"{sum(final_output_lens) / (end_time - start_time):.2f} token/s"
    )
    summary_table.add_row(
        "Input Throughput",
        f"{(sum(total_input_len)) / (end_time - start_time):.2f} token/s",
    )
    summary_table.add_row(
        "Total Throughput",
        f"{(sum(total_input_len) + sum(final_output_lens)) / (end_time - start_time):.2f} token/s",
    )

    data_table = Table(title="Data Summary", show_lines=True, header_style="bold")
    data_table.add_column("Metric", style="cyan", no_wrap=True)
    data_table.add_column("P0", style="white")
    data_table.add_column("P25", style="white")
    data_table.add_column("P50", style="white")
    data_table.add_column("P75", style="white")
    data_table.add_column("P100", style="white")
    data_table.add_row("Input Tokens", *get_printable_time(input_values))
    data_table.add_row("Output Tokens", *get_printable_time(output_values))
    data_table.add_row("E2E latency(s)", *get_printable_time(e2e_values))

    metric_table = Table(title="Metric Summary", show_lines=True, header_style="bold")
    metric_table.add_column("Metric", style="cyan", no_wrap=True)
    metric_table.add_column("P50", style="white")
    metric_table.add_column("P90", style="white")
    metric_table.add_column("P99", style="white")
    metric_table.add_column("P100", style="white")
    metric_table.add_column("Avg", style="white")
    metric_table.add_row("TTFT(ms)", *get_printable_time(first_values))
    metric_table.add_row("ITL(ms)", *get_printable_time(decode_values))
    metric_table.add_row("TPS PER USER(tokens/s)", *get_printable_time(tps_values))

    console.print(summary_table)
    console.print(data_table)
    console.print(metric_table)


if __name__ == "__main__":
    main()

