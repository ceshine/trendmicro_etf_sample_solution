"""Convert Big-5 to UTF-8 and fix some trailing commas"""
with open("data/tetfp.csv", "rb") as f:
    with open("data/tetfp_fixed.csv", "wb") as f2:
        for line in f.readlines():
            line = line.decode("big5").strip()
            if line.endswith(","):
                f2.write((line[:-1] + "\n").encode("utf8"))
            else:
                f2.write((line + "\n").encode("utf8"))
