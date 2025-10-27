import csv
import os
from typing import Final

OUTPUT_DIR: Final[str] = "data/synthetic"
OUTPUT_PATH: Final[str] = os.path.join(OUTPUT_DIR, "commands_manifest.csv")
COMMANDS: Final[list[tuple[str, str]]] = [
    ("sydity.wav", "сидіти"),
    ("lezhaty.wav", "лежати"),
    ("do_mene.wav", "до мене"),
    ("bark.wav", "голос"),
]


def main() -> None:
    """Write the manifest file to :data:`OUTPUT_PATH`."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "filename", "label"])
        for index, (filename, label) in enumerate(COMMANDS, start=1):
            writer.writerow([index, filename, label])

    print("synthetic manifest generated.")


if __name__ == "__main__":
    main()
