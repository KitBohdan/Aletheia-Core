import argparse, json
from .robodog.dog_bot_brain import RoboDogBrain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vct/config.yaml")
    ap.add_argument("--wav")
    ap.add_argument("--cmd")
    ap.add_argument("--gpio-pin", type=int, default=None)
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    brain = RoboDogBrain(cfg_path=args.config, gpio_pin=args.gpio_pin, simulate=args.simulate)
    if args.wav:
        res = brain.run_once_from_wav(args.wav)
    else:
        res = brain.handle_command(args.cmd or "сидіти")
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()