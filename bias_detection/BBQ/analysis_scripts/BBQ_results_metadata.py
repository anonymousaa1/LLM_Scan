import os
import json
import pandas as pd
import numpy as np
from glob import glob

# Define directory paths
this_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_dir)

# Function to read JSONL files
def read_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]

# List of unknown strings
unknowns = [
    "unknown", "cannot be determined", "can't be determined", "not answerable",
    "not known", "not enough info", "not enough information", "cannot answer",
    "can't answer", "undetermined"
]

# Get list of JSONL filenames
filenames = glob("../data/*.jsonl")
filenames = ["../data/Gender_identity.jsonl"]

# Read and process each JSONL file
dat = []
for f in filenames:
    temp = read_jsonl(f)
    temp_df = pd.DataFrame(temp)
    
    ans0_info = pd.json_normalize(temp_df['answer_info'].apply(lambda x: x['ans0'])).add_prefix('ans0_')
    ans1_info = pd.json_normalize(temp_df['answer_info'].apply(lambda x: x['ans1'])).add_prefix('ans1_')
    ans2_info = pd.json_normalize(temp_df['answer_info'].apply(lambda x: x['ans2'])).add_prefix('ans2_')
    stereotyped_groups = pd.json_normalize(temp_df['additional_metadata'].apply(lambda x: x['stereotyped_groups'])).rename(columns={0: 'stereotyped_groups'})
    
    temp_df = temp_df.drop(columns=['answer_info', 'additional_metadata'])
    temp_df = pd.concat([temp_df, ans0_info, ans1_info, ans2_info, stereotyped_groups], axis=1)
    dat.append(temp_df)

dat = pd.concat(dat, ignore_index=True)

# Check for missing stereotyped groups
check_missing = dat[dat['stereotyped_groups'] == "list()"]
print("Missing categories:", check_missing['category'].unique())

# Read template files
templates_files = glob("../templates/*.csv")
templates_files = ["../templates/new_templates - Gender_identity.csv"]
st_group_data = []

for file in templates_files:
    if not any(exclude in file for exclude in ["vocab", "_x_", "Filler"]):
        temp = pd.read_csv(file)
        temp2 = temp[['Category', 'Known_stereotyped_groups', 'Q_id', 'Relevant_social_values']].rename(columns={
            "Category": "category",
            "Q_id": "question_index"
        })
        temp2['question_index'] = temp2['question_index'].astype(str)
        st_group_data.append(temp2)

st_group_data = pd.concat(st_group_data, ignore_index=True)

st_group_data2 = st_group_data.replace({
    "GenderIdentity": "Gender_identity",
    "PhysicalAppearance": "Physical_appearance",
    "RaceEthnicity": "Race_ethnicity",
    "Religion ": "Religion",
    "SexualOrientation": "Sexual_orientation",
    "DisabilityStatus": "Disability_status"
})

st_group_data2 = st_group_data2.groupby(['category', 'question_index', 'Known_stereotyped_groups', 'Relevant_social_values']).size().reset_index(name='count').drop(columns=['count'])

dat4 = pd.merge(dat, st_group_data2, on=['category', 'question_index'], how='outer')

# Processing non-intersectional templates
dat_base = dat4[~dat4['category'].str.contains("_x_")].drop(columns=['stereotyped_groups'])
dat_base = dat_base.replace({
    "man": "M", "boy": "M", "woman": "F", "girl": "F",
    "transgender women, transgender men": "trans",
    "transgender men": "trans", "transgender women": "trans",
    "transgender women, transgender men, trans": "trans",
    "low SES": "lowSES", "high SES": "highSES"
})

dat_base = dat_base.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)
dat_base['ProperName'] = np.where(dat_base['question_index'].astype(int) > 25, "name", "label")

dat_base = dat_base.assign(
    ans0_info=dat_base['ans0_info'].str.replace(r"[MF]-", "", regex=True),
    ans1_info=dat_base['ans1_info'].str.replace(r"[MF]-", "", regex=True),
    ans2_info=dat_base['ans2_info'].str.replace(r"[MF]-", "", regex=True)
)

# Create target location columns
dat_base = dat_base.assign(
    target_loc_0=np.where(dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans0_info'], case=False), 1, 0),
    target_loc_1=np.where(dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans1_info'], case=False), 1, 0),
    target_loc_2=np.where(dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans2_info'], case=False), 1, 0)
)

# Correct target location for Age category
dat_base.loc[dat_base['category'] == "Age", 'target_loc_0'] = np.where(
    ((dat_base['Known_stereotyped_groups'] == "nonOld") & (dat_base['ans0_info'] == "nonOld")) | 
    ((dat_base['Known_stereotyped_groups'] == "old") & (dat_base['ans0_info'] == "old")), 1, 0)
dat_base.loc[dat_base['category'] == "Age", 'target_loc_1'] = np.where(
    ((dat_base['Known_stereotyped_groups'] == "nonOld") & (dat_base['ans1_info'] == "nonOld")) | 
    ((dat_base['Known_stereotyped_groups'] == "old") & (dat_base['ans1_info'] == "old")), 1, 0)
dat_base.loc[dat_base['category'] == "Age", 'target_loc_2'] = np.where(
    ((dat_base['Known_stereotyped_groups'] == "nonOld") & (dat_base['ans2_info'] == "nonOld")) | 
    ((dat_base['Known_stereotyped_groups'] == "old") & (dat_base['ans2_info'] == "old")), 1, 0)

# Correct target location for Nationality category
dat_base.loc[dat_base['category'] == "Nationality", 'target_loc_0'] = np.where(
    dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans0_text'], case=False), 1, 0)
dat_base.loc[dat_base['category'] == "Nationality", 'target_loc_1'] = np.where(
    dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans1_text'], case=False), 1, 0)
dat_base.loc[dat_base['category'] == "Nationality", 'target_loc_2'] = np.where(
    dat_base['Known_stereotyped_groups'].str.contains(dat_base['ans2_text'], case=False), 1, 0)

# Correct target location for race category
dat_base.loc[(dat_base['ans0_text'] == "African") & (dat_base['Known_stereotyped_groups'].str.contains("african american", case=False)), 'target_loc_0'] = 0
dat_base.loc[(dat_base['ans1_text'] == "African") & (dat_base['Known_stereotyped_groups'].str.contains("african american", case=False)), 'target_loc_1'] = 0
dat_base.loc[(dat_base['ans2_text'] == "African") & (dat_base['Known_stereotyped_groups'].str.contains("african american", case=False)), 'target_loc_2'] = 0

# Correct target location for identifying trans target
dat_base.loc[(dat_base['Known_stereotyped_groups'] == "trans") & (dat_base['ans0_info'].str.contains("^trans", case=False)), 'target_loc_0'] = 1
dat_base.loc[(dat_base['Known_stereotyped_groups'] == "trans") & (dat_base['ans1_info'].str.contains("^trans", case=False)), 'target_loc_1'] = 1
dat_base.loc[(dat_base['Known_stereotyped_groups'] == "trans") & (dat_base['ans2_info'].str.contains("^trans", case=False)), 'target_loc_2'] = 1

# Assign target location
dat_base['target_loc'] = np.select(
    [dat_base['target_loc_0'] == 1, dat_base['target_loc_1'] == 1, dat_base['target_loc_2'] == 1],
    [0, 1, 2],
    default=np.nan
)

# Correct target location for non-negative examples
dat_base_target_loc_corrected = dat_base.assign(
    new_target_loc=np.where(
        (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 0) & (dat_base['ans1_info'] != "unknown"), 1,
        np.where(
            (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 0) & (dat_base['ans2_info'] != "unknown"), 2,
            np.where(
                (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 1) & (dat_base['ans0_info'] != "unknown"), 0,
                np.where(
                    (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 1) & (dat_base['ans2_info'] != "unknown"), 2,
                    np.where(
                        (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 2) & (dat_base['ans0_info'] != "unknown"), 0,
                        np.where(
                            (dat_base['question_polarity'] == "nonneg") & (dat_base['target_loc'] == 2) & (dat_base['ans1_info'] != "unknown"), 1,
                            dat_base['target_loc']
                        )
                    )
                )
            )
        )
    )
)

# Check for missing target locations after correction
missing_target_loc_count = dat_base_target_loc_corrected[dat_base_target_loc_corrected['new_target_loc'].isna()].shape[0]
print("Number of missing target locations after correction:", missing_target_loc_count)

# Final selection and renaming
dat_base_selected = dat_base_target_loc_corrected.assign(
    full_cond=np.nan,
    Known_stereotyped_race=np.nan,
    Known_stereotyped_var2=np.nan,
    corr_ans_aligns_var2=np.nan,
    corr_ans_aligns_race=np.nan
).rename(columns={"new_target_loc": "target_loc"})

dat_base_selected = dat_base_selected[['category', 'question_index', 'example_id', 'target_loc', 'label_type', 'Known_stereotyped_groups', 'Relevant_social_values']]

# Processing intersectional templates
st_group_data = []

for file in templates_files:
    if "_x_" in file:
        temp = pd.read_csv(file)
        temp2 = temp[['Category', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Q_id', 'Relevant_social_values', 'Proper_nouns_only']].rename(columns={
            "Category": "category",
            "Q_id": "question_index"
        })
        temp2['question_index'] = temp2['question_index'].astype(str)
        st_group_data.append(temp2)

st_group_data = pd.concat(st_group_data, ignore_index=True)

st_group_data2 = st_group_data.replace({
    'Gender_x_race': 'Race_x_gender'
}).groupby(['category', 'question_index', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Relevant_social_values', 'Proper_nouns_only']).size().reset_index(name='count').drop(columns=['count'])

dat4 = pd.merge(dat, st_group_data2, on=['category', 'question_index'], how='outer').dropna(subset=['example_id'])

# Process Race_x_gender
dat_race_x_gender = dat4[dat4['category'] == "Race_x_gender"].drop(columns=['stereotyped_groups'])

dat_race_x_gender = dat_race_x_gender.assign(
    label_type=np.where(dat_race_x_gender['Proper_nouns_only'], "name", "label")
).drop(columns=['Proper_nouns_only'])

dat_race_x_gender = dat_race_x_gender.replace({
    "ans0_info": dat_race_x_gender['ans0_info'].str.replace("M-", "").str.replace("F-", ""),
    "ans1_info": dat_race_x_gender['ans1_info'].str.replace("M-", "").str.replace("F-", ""),
    "ans2_info": dat_race_x_gender['ans2_info'].str.replace("M-", "").str.replace("F-", "")
})

dat_race_x_gender = dat_race_x_gender.assign(
    ans0_gender=np.where(dat_race_x_gender['ans0_info'] == 'unknown', 'unknown', dat_race_x_gender['ans0_info'].str[0]),
    ans0_race=np.where(dat_race_x_gender['ans0_info'] == 'unknown', 'unknown', dat_race_x_gender['ans0_info'].str[2:]),
    ans1_gender=np.where(dat_race_x_gender['ans1_info'] == 'unknown', 'unknown', dat_race_x_gender['ans1_info'].str[0]),
    ans1_race=np.where(dat_race_x_gender['ans1_info'] == 'unknown', 'unknown', dat_race_x_gender['ans1_info'].str[2:]),
    ans2_gender=np.where(dat_race_x_gender['ans2_info'] == 'unknown', 'unknown', dat_race_x_gender['ans2_info'].str[0]),
    ans2_race=np.where(dat_race_x_gender['ans2_info'] == 'unknown', 'unknown', dat_race_x_gender['ans2_info'].str[2:])
)

dat_race_x_gender = dat_race_x_gender.assign(
    corr_ans_race=np.select(
        [dat_race_x_gender['label'] == 0, dat_race_x_gender['label'] == 1, dat_race_x_gender['label'] == 2],
        [dat_race_x_gender['ans0_race'], dat_race_x_gender['ans1_race'], dat_race_x_gender['ans2_race']],
        default=np.nan
    ),
    corr_ans_gender=np.select(
        [dat_race_x_gender['label'] == 0, dat_race_x_gender['label'] == 1, dat_race_x_gender['label'] == 2],
        [dat_race_x_gender['ans0_gender'], dat_race_x_gender['ans1_gender'], dat_race_x_gender['ans2_gender']],
        default=np.nan
    )
)

dat_race_x_gender = dat_race_x_gender.assign(
    corr_ans_aligns_race=dat_race_x_gender['Known_stereotyped_race'].str.contains(dat_race_x_gender['corr_ans_race'], case=False).astype(int),
    corr_ans_aligns_gender=dat_race_x_gender['Known_stereotyped_var2'].str.contains(dat_race_x_gender['corr_ans_gender'], case=False).astype(int),
    race_condition=np.where(
        (dat_race_x_gender['ans0_race'] == dat_race_x_gender['ans1_race']) | 
        (dat_race_x_gender['ans1_race'] == dat_race_x_gender['ans2_race']) | 
        (dat_race_x_gender['ans0_race'] == dat_race_x_gender['ans2_race']), 
        "Match Race", "Mismatch Race"
    ),
    gender_condition=np.where(
        (dat_race_x_gender['ans0_gender'] == dat_race_x_gender['ans1_gender']) | 
        (dat_race_x_gender['ans1_gender'] == dat_race_x_gender['ans2_gender']) | 
        (dat_race_x_gender['ans0_gender'] == dat_race_x_gender['ans2_gender']), 
        "Match Gender", "Mismatch Gender"
    ),
    full_cond=dat_race_x_gender['race_condition'] + "\n " + dat_race_x_gender['gender_condition'],
    target_loc=np.select(
        [
            (dat_race_x_gender['Known_stereotyped_var2'].str[0] == dat_race_x_gender['ans0_gender']) & 
            (dat_race_x_gender['Known_stereotyped_race'].str.contains(dat_race_x_gender['ans0_race'], case=False)),
            (dat_race_x_gender['Known_stereotyped_var2'].str[0] == dat_race_x_gender['ans1_gender']) & 
            (dat_race_x_gender['Known_stereotyped_race'].str.contains(dat_race_x_gender['ans1_race'], case=False)),
            (dat_race_x_gender['Known_stereotyped_var2'].str[0] == dat_race_x_gender['ans2_gender']) & 
            (dat_race_x_gender['Known_stereotyped_race'].str.contains(dat_race_x_gender['ans2_race'], case=False))
        ],
        [0, 1, 2],
        default=np.nan
    )
)

# Correct target location for non-negative examples
dat_race_x_gender_target_loc_corrected = dat_race_x_gender.assign(
    new_target_loc=np.where(
        (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 0) & (dat_race_x_gender['ans1_gender'] != "unknown"), 1,
        np.where(
            (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 0) & (dat_race_x_gender['ans2_gender'] != "unknown"), 2,
            np.where(
                (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 1) & (dat_race_x_gender['ans0_gender'] != "unknown"), 0,
                np.where(
                    (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 1) & (dat_race_x_gender['ans2_gender'] != "unknown"), 2,
                    np.where(
                        (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 2) & (dat_race_x_gender['ans0_gender'] != "unknown"), 0,
                        np.where(
                            (dat_race_x_gender['question_polarity'] == "nonneg") & (dat_race_x_gender['target_loc'] == 2) & (dat_race_x_gender['ans1_gender'] != "unknown"), 1,
                            dat_race_x_gender['target_loc']
                        )
                    )
                )
            )
        )
    )
)

# Check for missing target locations after correction
missing_target_loc_count = dat_race_x_gender_target_loc_corrected[dat_race_x_gender_target_loc_corrected['new_target_loc'].isna()].shape[0]
print("Number of missing target locations after correction:", missing_target_loc_count)

# Final selection and renaming
dat_racegen_selected = dat_race_x_gender_target_loc_corrected[['category', 'question_index', 'example_id', 'new_target_loc', 'label_type', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Relevant_social_values', 'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond']].rename(columns={"new_target_loc": "target_loc"})

# Process Race_x_SES
dat_race_x_ses = dat4[dat4['category'] == "Race_x_SES"].drop(columns=['stereotyped_groups'])

dat_race_x_ses = dat_race_x_ses.assign(
    label_type=np.where(dat_race_x_ses['Proper_nouns_only'], "name", "label")
).drop(columns=['Proper_nouns_only'])

dat_race_x_ses = dat_race_x_ses.replace({
    "ans0_info": dat_race_x_ses['ans0_info'].str.replace("M-", "").str.replace("F-", ""),
    "ans1_info": dat_race_x_ses['ans1_info'].str.replace("M-", "").str.replace("F-", ""),
    "ans2_info": dat_race_x_ses['ans2_info'].str.replace("M-", "").str.replace("F-", "")
})

dat_race_x_ses = dat_race_x_ses.assign(
    ans0_gender=np.where(dat_race_x_ses['ans0_info'] == 'unknown', 'unknown', dat_race_x_ses['ans0_info'].str[:6]),
    ans0_race=np.where(dat_race_x_ses['ans0_info'] == 'unknown', 'unknown', dat_race_x_ses['ans0_info'].str[7:]),
    ans1_gender=np.where(dat_race_x_ses['ans1_info'] == 'unknown', 'unknown', dat_race_x_ses['ans1_info'].str[:6]),
    ans1_race=np.where(dat_race_x_ses['ans1_info'] == 'unknown', 'unknown', dat_race_x_ses['ans1_info'].str[7:]),
    ans2_gender=np.where(dat_race_x_ses['ans2_info'] == 'unknown', 'unknown', dat_race_x_ses['ans2_info'].str[:6]),
    ans2_race=np.where(dat_race_x_ses['ans2_info'] == 'unknown', 'unknown', dat_race_x_ses['ans2_info'].str[7:])
)

dat_race_x_ses = dat_race_x_ses.assign(
    corr_ans_race=np.select(
        [dat_race_x_ses['label'] == 0, dat_race_x_ses['label'] == 1, dat_race_x_ses['label'] == 2],
        [dat_race_x_ses['ans0_race'], dat_race_x_ses['ans1_race'], dat_race_x_ses['ans2_race']],
        default=np.nan
    ),
    corr_ans_gender=np.select(
        [dat_race_x_ses['label'] == 0, dat_race_x_ses['label'] == 1, dat_race_x_ses['label'] == 2],
        [dat_race_x_ses['ans0_gender'], dat_race_x_ses['ans1_gender'], dat_race_x_ses['ans2_gender']],
        default=np.nan
    )
)

dat_race_x_ses = dat_race_x_ses.assign(
    corr_ans_aligns_race=dat_race_x_ses['Known_stereotyped_race'].str.contains(dat_race_x_ses['corr_ans_race'], case=False).astype(int),
    corr_ans_aligns_gender=dat_race_x_ses['Known_stereotyped_var2'].str.contains(dat_race_x_ses['corr_ans_gender'], case=False).astype(int),
    race_condition=np.where(
        (dat_race_x_ses['ans0_race'] == dat_race_x_ses['ans1_race']) | 
        (dat_race_x_ses['ans1_race'] == dat_race_x_ses['ans2_race']) | 
        (dat_race_x_ses['ans0_race'] == dat_race_x_ses['ans2_race']), 
        "Match Race", "Mismatch Race"
    ),
    SES_condition=np.where(
        (dat_race_x_ses['ans0_gender'] == dat_race_x_ses['ans1_gender']) | 
        (dat_race_x_ses['ans1_gender'] == dat_race_x_ses['ans2_gender']) | 
        (dat_race_x_ses['ans0_gender'] == dat_race_x_ses['ans2_gender']), 
        "Match SES", "Mismatch SES"
    ),
    full_cond=dat_race_x_ses['race_condition'] + "\n " + dat_race_x_ses['SES_condition'],
    target_loc=np.select(
        [
            (dat_race_x_ses['Known_stereotyped_var2'].str[:3] == dat_race_x_ses['ans0_gender'].str[:3]) & 
            (dat_race_x_ses['Known_stereotyped_race'].str.contains(dat_race_x_ses['ans0_race'], case=False)),
            (dat_race_x_ses['Known_stereotyped_var2'].str[:3] == dat_race_x_ses['ans1_gender'].str[:3]) & 
            (dat_race_x_ses['Known_stereotyped_race'].str.contains(dat_race_x_ses['ans1_race'], case=False)),
            (dat_race_x_ses['Known_stereotyped_var2'].str[:3] == dat_race_x_ses['ans2_gender'].str[:3]) & 
            (dat_race_x_ses['Known_stereotyped_race'].str.contains(dat_race_x_ses['ans2_race'], case=False))
        ],
        [0, 1, 2],
        default=np.nan
    )
)

# Correct target location for non-negative examples
dat_race_x_ses_target_loc_corrected = dat_race_x_ses.assign(
    new_target_loc=np.where(
        (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 0) & (dat_race_x_ses['ans1_gender'] != "unknown"), 1,
        np.where(
            (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 0) & (dat_race_x_ses['ans2_gender'] != "unknown"), 2,
            np.where(
                (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 1) & (dat_race_x_ses['ans0_gender'] != "unknown"), 0,
                np.where(
                    (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 1) & (dat_race_x_ses['ans2_gender'] != "unknown"), 2,
                    np.where(
                        (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 2) & (dat_race_x_ses['ans0_gender'] != "unknown"), 0,
                        np.where(
                            (dat_race_x_ses['question_polarity'] == "nonneg") & (dat_race_x_ses['target_loc'] == 2) & (dat_race_x_ses['ans1_gender'] != "unknown"), 1,
                            dat_race_x_ses['target_loc']
                        )
                    )
                )
            )
        )
    )
)

# Check for missing target locations after correction
missing_target_loc_count = dat_race_x_ses_target_loc_corrected[dat_race_x_ses_target_loc_corrected['new_target_loc'].isna()].shape[0]
print("Number of missing target locations after correction:", missing_target_loc_count)

# Final selection and renaming
dat_raceses_selected = dat_race_x_ses_target_loc_corrected[['category', 'question_index', 'example_id', 'new_target_loc', 'label_type', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Relevant_social_values', 'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond']].rename(columns={"new_target_loc": "target_loc"})

# Combine intersectional data
dat_ints = pd.concat([dat_racegen_selected, dat_raceses_selected], ignore_index=True)
dat_ints['Known_stereotyped_groups'] = np.nan

dat_base_selected['full_cond'] = np.nan
dat_base_selected['Known_stereotyped_race'] = np.nan
dat_base_selected['Known_stereotyped_var2'] = np.nan
dat_base_selected['corr_ans_aligns_var2'] = np.nan
dat_base_selected['corr_ans_aligns_race'] = np.nan

all_metadata = pd.concat([dat_ints, dat_base_selected], ignore_index=True)

# Save to CSV
all_metadata.to_csv("additional_metadata.csv", index=False)

