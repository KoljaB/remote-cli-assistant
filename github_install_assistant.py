from __future__ import annotations

import argparse
from pathlib import Path

from install_assistant.controller import ChatAssistantConfig, ChatAssistantController
from install_assistant.remote_control import run_remote_worker_server
from install_assistant.worker import run_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an AI-assisted CLI controller locally or connect it to a remote "
            "visible CLI window over the local network."
        )
    )
    parser.add_argument(
        "initial_goal",
        nargs="?",
        help="Optional initial goal or instruction for the assistant.",
    )
    parser.add_argument(
        "--goal",
        help="Optional initial goal or instruction for the assistant.",
    )
    parser.add_argument(
        "--planner-model",
        default="gpt-5.4-mini",
        help="Model used for new user instructions and deeper planning turns.",
    )
    parser.add_argument(
        "--loop-model",
        default="gpt-5.4-nano",
        help="Model used for routine follow-up turns after command execution.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional OpenAI-compatible base URL such as http://127.0.0.1:1234/v1 for LM Studio.",
    )
    parser.add_argument(
        "--research-backend",
        choices=["auto", "openai", "searxng", "none"],
        default="auto",
        help="How external web research should be performed.",
    )
    parser.add_argument(
        "--search-api-url",
        help="Base URL for a self-hosted SearXNG instance, for example http://127.0.0.1:8080.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "confirm"],
        default="confirm",
        help="Whether the assistant should execute routine commands automatically or ask before each one.",
    )
    parser.add_argument(
        "--max-auto-steps",
        type=int,
        default=12,
        help="Maximum number of consecutive automatic command executions before the assistant pauses.",
    )
    parser.add_argument(
        "--connect",
        help="Connect to a remote CLI worker in HOST:PORT format.",
    )
    parser.add_argument(
        "--worker-server",
        action="store_true",
        help="Run a remote worker server that opens a visible CLI window on this machine.",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Bind host for --worker-server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --worker-server or --connect.",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args.worker)
    if args.worker_server:
        return run_remote_worker_server(listen_host=args.listen_host, port=args.port)

    initial_goal = args.goal or args.initial_goal
    config = ChatAssistantConfig(
        initial_goal=initial_goal,
        planner_model=args.planner_model,
        loop_model=args.loop_model,
        base_url=args.base_url,
        research_backend=args.research_backend,
        search_api_url=args.search_api_url,
        execution_mode=args.execution_mode,
        connect_target=args.connect,
        max_auto_steps=args.max_auto_steps,
    )
    return ChatAssistantController(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
