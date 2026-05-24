import argparse
import os
import sys

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8081/v1"
DEFAULT_SYSTEM = "You are a helpful assistant."


def pick_model(client: OpenAI) -> str:
    models = [m.id for m in client.models.list().data]
    if not models:
        raise RuntimeError("llama-server returned no models")
    if len(models) == 1:
        return models[0]
    print("Available models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}) {m}")
    while True:
        choice = input(f"Select model [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn chat CLI for llama-server")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMA_API_KEY"),
        help="API key for the llama-server (or set LLAMA_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL),
        help=f"Full base URL of the OpenAI-compatible server (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LLAMA_HOST"),
        help="Shortcut to override just the host (e.g. 'xyz.ai' or 'localhost:8081'); appends /v1",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--system", help="System prompt string")
    group.add_argument("--system-file", help="Path to a file containing the system prompt")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("error: --api-key is required (or set LLAMA_API_KEY)")

    if args.host:
        host = args.host
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        base_url = host.rstrip("/") + "/v1"
    else:
        base_url = args.base_url

    client = OpenAI(
        base_url=base_url,
        api_key=args.api_key,
        default_headers={"User-Agent": "curl/8.7.1"},
    )

    if args.system_file:
        with open(args.system_file) as f:
            system_prompt = f.read()
    else:
        system_prompt = args.system or DEFAULT_SYSTEM

    model = pick_model(client)

    messages = [{"role": "system", "content": system_prompt}]
    print(f"Chatting with BlackSmith ({model}). Type /exit to quit, /reset to clear history.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            messages = messages[:1]
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})

        print("bot> ", end="", flush=True)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        chunks = []
        for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
        print("\n")
        messages.append({"role": "assistant", "content": "".join(chunks)})


if __name__ == "__main__":
    main()
