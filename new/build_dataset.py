import pandas as pd
import re

# Loading the real data set
df = pd.read_csv("gene_data.csv")

# Data prview
# df.head()
# print(df["Germline classification"].value_counts())

# Checking for duplicates
s = df[["CDS Mutation", "AA Mutation"]].drop_duplicates().shape
print(s)

# Removing duplicates and checking the distribution of Germline classification
df_unique = df.drop_duplicates(subset=["CDS Mutation", "AA Mutation"])
print(df_unique["Germline classification"].value_counts())

# Loading new data
clinvar = pd.read_csv("clinvar_result.txt", sep="\t", dtype=str)
print(clinvar.shape)
print(clinvar.columns)

# keep only FLT3 rows (sometimes Gene(s) can be 'ETV6|FLT3' etc)
clinvar_flt3 = clinvar[clinvar["Gene(s)"].str.contains(r"\bFLT3\b", na=False)].copy()

# keep only SNVs 
clinvar_flt3 = clinvar_flt3[clinvar_flt3["Variant type"].str.lower().eq("single nucleotide variant")].copy()

# keep clean germline labels
keep = {"Pathogenic", "Likely pathogenic", "Benign", "Likely benign"}
clinvar_flt3 = clinvar_flt3[clinvar_flt3["Germline classification"].isin(keep)].copy()


# print(clinvar_flt3.shape)
# print(clinvar_flt3["Germline classification"].value_counts())
# print(clinvar_flt3[["Name","Variant type","Germline classification"]].head(10))

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C",
    "Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
    "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P",
    "Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"
}

def parse_from_name(name):
    if pd.isna(name):
        return pd.Series([None]*6)

    # c.2979G>A
    mc = re.search(r"c\.(\d+)([ACGT])>([ACGT])", name)
    if mc:
        cds_pos, cds_from, cds_to = int(mc.group(1)), mc.group(2), mc.group(3)
    else:
        cds_pos, cds_from, cds_to = None, None, None

    # p.Ser993= or p.Ala988Pro or p.K663R
    mp = re.search(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3}|=)", name)
    if mp:
        aa_from, aa_pos, aa_to = mp.group(1), int(mp.group(2)), mp.group(3)
        aa_from = AA3_TO_1.get(aa_from, aa_from)  # handles 3-letter
        aa_to   = AA3_TO_1.get(aa_to, aa_to)      # handles 3-letter or "="
    else:
        aa_pos, aa_from, aa_to = None, None, None

    return pd.Series([cds_pos, cds_from, cds_to, aa_pos, aa_from, aa_to])

clinvar_flt3[["cds_pos","cds_from","cds_to","aa_pos","aa_from","aa_to"]] = (
    clinvar_flt3["Name"].apply(parse_from_name)
)

# Checking missing values 
# print(clinvar_flt3[["Name","cds_pos","cds_from","cds_to","aa_pos","aa_from","aa_to"]].head(10))
# print("Missing cds_pos:", clinvar_flt3["cds_pos"].isna().sum())
# print("Missing aa_pos:",  clinvar_flt3["aa_pos"].isna().sum())

clinvar_structured = clinvar_flt3.dropna(
    subset=["cds_pos","cds_from","cds_to","aa_pos","aa_from","aa_to"]
).copy()

label_map = {
    "Likely benign": "Benign",
    "Benign": "Benign",
    "Likely pathogenic": "Pathogenic",
    "Pathogenic": "Pathogenic",
}

clinvar_structured["Germline classification"] = (
    clinvar_structured["Germline classification"].map(label_map)
)

# print(clinvar_structured["Germline classification"].value_counts())

# # Combine with original dataset
# final = pd.concat([df, clinvar_structured], ignore_index=True)

# final = final.drop_duplicates(
#     subset=["cds_pos","cds_from","cds_to",
#             "aa_pos","aa_from","aa_to"]
# )

# print(final.shape)
# print(final["Germline classification"].value_counts())


def parse_from_cds_aa(cds, aa):
    if pd.isna(cds):
        cds_pos, cds_from, cds_to = None, None, None
    else:
        mc = re.search(r"c\.(\d+)([ACGT])>([ACGT])", cds)
        if mc:
            cds_pos, cds_from, cds_to = int(mc.group(1)), mc.group(2), mc.group(3)
        else:
            cds_pos, cds_from, cds_to = None, None, None

    if pd.isna(aa):
        aa_pos, aa_from, aa_to = None, None, None
    else:
        mp = re.search(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3}|=)", aa)
        if mp:
            aa_from, aa_pos, aa_to = mp.group(1), int(mp.group(2)), mp.group(3)
            aa_from = AA3_TO_1.get(aa_from, aa_from)
            aa_to   = AA3_TO_1.get(aa_to, aa_to)
        else:
            aa_pos, aa_from, aa_to = None, None, None

    return pd.Series([cds_pos, cds_from, cds_to, aa_pos, aa_from, aa_to])

df[["cds_pos","cds_from","cds_to",
       "aa_pos","aa_from","aa_to"]] = (
    df.apply(lambda row: parse_from_cds_aa(
        row["CDS Mutation"], 
        row["AA Mutation"]
    ), axis=1)
)

df["Germline classification"] = (
    df["Germline classification"].map(label_map)
)

#dropping duplictates again after parsing
df = df.drop_duplicates(
    subset=["cds_pos","cds_from","cds_to",
            "aa_pos","aa_from","aa_to"]
)

clinvar_structured = clinvar_structured.drop_duplicates(
    subset=["cds_pos","cds_from","cds_to",
            "aa_pos","aa_from","aa_to"])

data = pd.concat(
    [df, clinvar_structured],
    ignore_index=True)
 
data = data.drop_duplicates(
    subset=["cds_pos","cds_from","cds_to",
            "aa_pos","aa_from","aa_to"])

conflicts = (
    data.groupby(["cds_pos","cds_from","cds_to",
                      "aa_pos","aa_from","aa_to"])
    ["Germline classification"]
    .nunique()
)
print("Conflicts:", (conflicts > 1).sum())
print(data.shape)
print(data["Germline classification"].value_counts())


#keep only needed columns
final_cols = [
    "cds_pos","cds_from","cds_to",
    "aa_pos","aa_from","aa_to",
    "Germline classification"
]

data = data[final_cols].copy()

print(data["cds_pos"].isna().sum())
print(data["aa_pos"].isna().sum())

data = data.dropna(subset=["aa_pos"]).copy()

data["cds_pos"] = data["cds_pos"].astype(int)
data["aa_pos"]  = data["aa_pos"].astype(int)

data.to_csv("new_data.csv", index=False)
