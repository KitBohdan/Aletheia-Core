from __future__ import annotations

import argparse
import json
from typing import Sequence

from .configuration import RoboDogSettings, apply_key_path, parse_typed_value
from .robodog.dog_bot_brain import RoboDogBrain


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="vct/config.yaml")
    parser.add_argument("--wav")
    parser.add_argument("--cmd")
    parser.add_argument("--gpio-pin", type=int, default=None)
    parser.add_argument("--simulate", action="store_true")


def _handle_run_command(args: argparse.Namespace) -> None:
    brain = RoboDogBrain(cfg_path=args.config, gpio_pin=args.gpio_pin, simulate=args.simulate)
    if args.wav:
        res = brain.run_once_from_wav(args.wav)
    else:
        res = brain.handle_command(args.cmd or "сидіти")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _handle_config_show(args: argparse.Namespace) -> None:
    settings = RoboDogSettings.load(args.config)
    payload = settings.model_dump(mode="json") if args.as_json else settings.model_dump()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _handle_config_set(args: argparse.Namespace) -> None:
    settings = RoboDogSettings.load(args.config)
    value = parse_typed_value(args.value, args.type)
    key_path = _split_key_path(args.key)
    updated = apply_key_path(settings, key_path, value)
    updated.save(args.config)
    print(f"✅ Updated {args.key} in {args.config}")


def _split_key_path(path: str) -> Sequence[str]:
    return [segment.strip() for segment in path.split(".") if segment.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboDog control CLI")
    _add_common_run_arguments(ap)
    sub = ap.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the RoboDog behavior once")
    _add_common_run_arguments(run_parser)
    run_parser.set_defaults(func=_handle_run_command)

    config_parser = sub.add_parser("config", help="Inspect or modify configuration")
    config_parser.add_argument("--config", default="vct/config.yaml")
    config_sub = config_parser.add_subparsers(dest="config_cmd")

    show_parser = config_sub.add_parser("show", help="Display the current configuration")
    show_parser.add_argument("--as-json", action="store_true")
    show_parser.set_defaults(func=_handle_config_show)

    set_parser = config_sub.add_parser("set", help="Update a configuration value")
    set_parser.add_argument("key", help="Dot separated key path to update (e.g. commands_map.сидіти)")
    set_parser.add_argument("value", help="New value for the key")
    set_parser.add_argument(
        "--type",
        choices=["str", "int", "float", "bool", "json"],
        default="str",
        help="Type of the value for correct parsing",
    )
    set_parser.set_defaults(func=_handle_config_set)

    args = ap.parse_args()
    if args.command is None:
        # Backwards compatibility: behave like "run"
        _handle_run_command(args)
        return

    if not hasattr(args, "func"):
        ap.error("No command specified")
    args.func(args)


if __name__ == "__main__":
    main()
