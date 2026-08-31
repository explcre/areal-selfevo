"""Two greedy scorings of the SAME weights: is the difference text or grading?"""
import json, pathlib, sys
sys.path.insert(0, "/home/ubuntu/areal-selfevo/experiments/bench")
from math_bench import grade

def load(p):
    out = {}
    for line in pathlib.Path(p).open():
        r = json.loads(line)
        if r.get("benchmark") != "math500":
            continue
        out[int(r["idx"])] = r
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(a) & set(b))
same_text = sum(1 for k in shared if a[k]["text"] == b[k]["text"])
same_boxed = sum(1 for k in shared if a[k].get("boxed") == b[k].get("boxed"))
same_grade = sum(1 for k in shared if a[k]["correct"] == b[k]["correct"])
acc_a = sum(a[k]["correct"] for k in shared)/len(shared)
acc_b = sum(b[k]["correct"] for k in shared)/len(shared)
print(f"problems           : {len(shared)}")
print(f"identical text     : {same_text} ({same_text/len(shared):.1%})")
print(f"identical boxed    : {same_boxed} ({same_boxed/len(shared):.1%})")
print(f"identical grade    : {same_grade} ({same_grade/len(shared):.1%})")
print(f"accuracy           : A {acc_a:.4f}   B {acc_b:.4f}   gap {acc_b-acc_a:+.4f}")

# Of the problems graded differently, how many also had different text?
diff = [k for k in shared if a[k]["correct"] != b[k]["correct"]]
dt = sum(1 for k in diff if a[k]["text"] != b[k]["text"])
print(f"\ngraded differently : {len(diff)}; of those, {dt} also had DIFFERENT text")
print(f"same text but different grade: {len(diff)-dt}  <- would be a grader bug")
# Regrade the stored text with the current grader, twice, to test the grader itself.
r1 = {k: grade(a[k]["text"], a[k]["gold"]) for k in shared}
r2 = {k: grade(a[k]["text"], a[k]["gold"]) for k in shared}
print(f"grader self-consistent on identical input: {sum(1 for k in shared if r1[k]==r2[k])}/{len(shared)}")
print(f"stored flag vs fresh regrade of same text: {sum(1 for k in shared if bool(a[k]['correct'])==r1[k])}/{len(shared)} agree")
