import csv, os
os.makedirs("data/synthetic", exist_ok=True)
with open("data/synthetic/commands_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id","filename","label"])
    for i,(fname,label) in enumerate([
        ("sydity.wav","сидіти"),("lezhaty.wav","лежати"),("do_mene.wav","до мене"),("bark.wav","голос")
    ],1):
        w.writerow([i,fname,label])
print("synthetic manifest generated.")