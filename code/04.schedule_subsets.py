import pandas as pd

# Load data
df = pd.read_csv("data/labelled_docs.csv", encoding="utf-8")

# Filter to schedule == 1, split by type
sched = df[df["schedule"] == 1]
sched_c = sched[sched["type"] == "c"]
sched_p = sched[sched["type"] == "p"]
print(f"Documents with schedule=1: {len(sched)} (c={len(sched_c)}, p={len(sched_p)})")

# Each subset: 300 docs = 150 c + 150 p
# Shared 100: 50 c + 50 p
# Exclusive per subset 200: 100 c + 100 p
# Total pool: 250 c + 250 p
pool_c = sched_c.sample(n=250, random_state=42)
pool_p = sched_p.sample(n=250, random_state=42)

def make_parts(pool, shared_n, excl_n):
    shared    = pool.iloc[:shared_n].copy()
    excl_a    = pool.iloc[shared_n:shared_n + excl_n].copy()
    excl_b    = pool.iloc[shared_n + excl_n:].copy()
    shared["shared"] = 1
    excl_a["shared"] = 0
    excl_b["shared"] = 0
    return shared, excl_a, excl_b

shared_c, excl_a_c, excl_b_c = make_parts(pool_c, shared_n=50, excl_n=100)
shared_p, excl_a_p, excl_b_p = make_parts(pool_p, shared_n=50, excl_n=100)

subset_a = pd.concat([shared_c, excl_a_c, shared_p, excl_a_p]).reset_index(drop=True)
subset_b = pd.concat([shared_c, excl_b_c, shared_p, excl_b_p]).reset_index(drop=True)

# Verify
overlap = set(subset_a["doc"]) & set(subset_b["doc"])
print(f"Subset A: {len(subset_a)} docs (c={len(subset_a[subset_a['type']=='c'])}, p={len(subset_a[subset_a['type']=='p'])}, shared={subset_a['shared'].sum()})")
print(f"Subset B: {len(subset_b)} docs (c={len(subset_b[subset_b['type']=='c'])}, p={len(subset_b[subset_b['type']=='p'])}, shared={subset_b['shared'].sum()})")
print(f"Actual doc overlap: {len(overlap)}")

subset_a.to_csv("data/schedule_subset_a.csv", index=False)
subset_b.to_csv("data/schedule_subset_b.csv", index=False)
print("Saved: data/schedule_subset_a.csv, data/schedule_subset_b.csv")
