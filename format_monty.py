import re

with open("monty.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
scene_num = 0

for line in lines:
    line = line.rstrip()

    # Scene headers
    if re.match(r"Scene\s+\d+", line):
        scene_num += 1
        out.append(f"\n<SCENE {scene_num}>\n")
        continue

    # Stage directions like [clop clop]
    if re.match(r"\[.*\]", line.strip()):
        sfx = line.strip().strip("[]")
        out.append(f"[SFX: {sfx}]\n")
        continue

    # Speaker lines
    if ":" in line and line.split(":")[0].isupper():
        speaker, text = line.split(":", 1)
        out.append(f"{speaker.strip()}: {text.strip()}\n")
        continue

    # Dialogue continuation or narration
    if line.strip():
        out.append(line.strip() + "\n")
    else:
        out.append("\n")

with open("monty_formatted.txt", "w", encoding="utf-8") as f:
    f.writelines(out)

print("Formatted file written to monty_formatted.txt")
