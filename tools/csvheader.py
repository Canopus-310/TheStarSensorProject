
import csv,sys
fn="C:/Users/aryan/PycharmProjects/TheStarSensorProject/data/hipparcos_subset.csv"
with open(fn, newline='', encoding='utf-8') as f:
    r=csv.reader(f)
    hdr=next(r)
    print("Header columns:", hdr)
