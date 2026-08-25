CodeHeader = """from tools import tabtools
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
"""

_MIMIC_IV_SCHEMA = """Dataset profile:
- This is a local, task-oriented MIMIC-IV-derived SQLite database, not the full canonical MIMIC-IV schema.
- subject_id identifies a patient, hadm_id identifies a hospital admission, and stay_id identifies an ICU stay. Treat them as identifiers even when SQLite stores them as REAL.
- No primary keys, foreign keys, uniqueness constraints, or NOT NULL constraints are declared. Filter nulls explicitly and never assume one row per identifier unless the query establishes it.
- The few-shot examples use a small number of deidentified records verified in this local database snapshot so that every example is executable. They are demonstrations, not evaluation cases. Their identifiers and any resulting values are restricted MIMIC-derived data: keep this prompt inside the approved environment and never send it to an unapproved third-party service.

Exact table names and verified columns (static_information is summarized because it has 172 columns):
(1) antibiotic: subject_id, hadm_id, starttime, endtime, drug, admittime, offset_at_st, offset_at_et
(2) bloodcultures: subject_id, hadm_id, charttime, itemid, org_name, admittime, offset_at_ct
(3) bloodpressure: subject_id, hadm_id, stay_id, charttime, sbp, dbp, admittime, offset_at_ct
(4) chart: subject_id, hadm_id, stay_id, charttime, itemid, valuenum, item, admittime, offset_at_ct
(5) gcs: subject_id, hadm_id, stay_id, charttime, gcs_eye, gcs_verbal, gcs_motor, gcs, admittime, offset_at_ct
(6) input_to_patients: subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom, patientweight, item, admittime, offset_at_st, offset_at_et
(7) procedures: subject_id, hadm_id, stay_id, starttime, endtime, itemid, value, valueuom, patientweight, item, admittime, offset_at_st, offset_at_et
(8) rrt_kidneytranplant: subject_id, hadm_id, seq_num, icd_code, icd_version, charttime, admittime, offset_at_ct. The misspelling in the table name is intentional and must be preserved.
(9) static_information: 172 nullable columns. Verified core columns include stay_id, hadm_id, subject_id, intime, outtime, admittime, dischtime, edregtime, edouttime, first_careunit, last_careunit, admission_location, discharge_location, admission_type, insurance, language, marital_status, los, gender, anchor_age, age_at_intime, age_at_admittime, race, deathtype, death_adm, deathtime, CKD, RRT_adm, kidney_transplant_adm, hist_KT_adm, hist_RRT_adm, offset_at_it, offset_at_ot, and offset_at_dht. It also contains CCI_*, EHCI_*, and *_p precomputed fields whose exact meanings must not be inferred from their names.
(10) urine: subject_id, hadm_id, stay_id, endtime, urine, intime, outtime, offset_it, starttime, admittime, offset_at_st, offset_at_et
(11) weight: subject_id, hadm_id, stay_id, charttime, valuenum, admittime, offset_at_ct

Join and time rules:
- Use (subject_id, hadm_id, stay_id) when both tables contain stay_id. Never join clinical events using subject_id alone.
- antibiotic, bloodcultures, and rrt_kidneytranplant do not contain stay_id. Link them to static_information with (subject_id, hadm_id) plus ICU time overlap because one admission may contain multiple ICU stays.
- Point events use charttime and fall in an ICU stay when intime <= charttime < outtime.
- Interval events use the half-open interval [starttime, endtime). Two intervals overlap when event_start < window_end and event_end > window_start.
- In this database snapshot, offset_at_ct, offset_at_st, and offset_at_et are elapsed minutes from admittime to charttime, starttime, and endtime. In static_information, offset_at_it, offset_at_ot, and offset_at_dht are elapsed minutes from admittime to intime, outtime, and deathtime. This was validated against the current DB rather than documented by its builder, so revalidate it after replacing or rebuilding the DB. Prefer named timestamps for joins and boundary logic; never reinterpret these offsets as hours or days or assume they are non-negative.
- urine.offset_it is a separate field whose reference clock and unit have not been established. Do not equate it with static_information.offset_at_it.
- chart, input_to_patients, and procedures already contain curated item labels; antibiotic contains drug and bloodcultures contains org_name. There are no dictionary tables to join.
- chart, bloodpressure, gcs, urine, and weight do not provide measurement-unit columns. Never invent units. amountuom, rateuom, and valueuom may be used only where present.
- Specialized tables may overlap information summarized elsewhere. Do not combine them blindly or double-count events.
- static_information contains precomputed features as well as stay context. Do not present a precomputed field as raw event evidence unless the question explicitly requests it and its meaning is established.
- Cost, general diagnosis lookup, general ICD dictionary lookup, and hospital transfer history are not represented. If a requested concept is outside this schema, use answer = None instead of inventing a table, column, code meaning, or value.
"""

_MIMIC_IV_TOOL_GUIDE = """(1) LoadDB(DBNAME) which loads the database DBNAME and returns the database. The DBNAME can be one of the following: antibiotic, bloodcultures, bloodpressure, chart, gcs, input_to_patients, procedures, rrt_kidneytranplant, static_information, urine, weight.
(2) FilterDB(DATABASE, CONDITIONS), which filters the DATABASE according to the CONDITIONS and returns the filtered database. The CONDITIONS is one single string composed of simple conditions joined with ||, using lowercase column names (e.g., subject_id=10001884||stay_id=37510196).
(3) GetValue(DATABASE, ARGUMENT), which returns a string containing all the values of the column in the DATABASE (if multiple values, separated by ", "). When there is no additional operations on the values, the ARGUMENT is the column_name in demand. If the values need to be returned with certain operations, the ARGUMENT is composed of the column_name and the operation (like gcs, min). Please do not contain " or ' in the argument.
(4) SQLInterpreter(SQL), which interprets the query SQL and returns the result as a list of tuples. Only read-only SELECT or WITH queries are allowed. Prefer it for joins, aggregates, grouping, and time windows; select only the needed columns and rows.
Do not call DataFrame methods directly. Manipulate loaded tables only through LoadDB, FilterDB, and GetValue; use Python built-ins or standard-library functions only for scalar values returned by those helpers."""

RetrKnowledge = (
    """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.

"""
    + _MIMIC_IV_SCHEMA
    + """
Question: what was the minimum gcs score recorded across all icu stays of patient 10002428?
Knowledge:
- We can find every icu stay of patient 10002428 in the static_information database.
- As gcs scores are stored in the gcs database, we can filter it by subject_id and stay_id for each stay and take the minimum.

Question: had any cefepime been given to patient 10005817 in their last hospital visit?
Knowledge:
- We can find the visiting information of patient 10005817 in the static_information database; max(admittime) identifies the last hospital visit.
- As antibiotic administrations are stored in the antibiotic database and its drug column stores the curated label CefePIME, we will check whether any matching record exists for that admission.

Question: what was the name of the procedure that was given two or more times to patient 10003400?
Knowledge:
- As procedures are stored in the procedures database with a curated item label, no dictionary join is needed.
- Counting events per item requires grouping, so a single SQLInterpreter query is the right tool.

Question: {question}
Knowledge:
"""
)

SYSTEM_PROMPT = """For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code."""

EHRAgent_Message_Prompt = (
    """Assume you have knowledge of the following tables in a local MIMIC-IV-derived SQLite database:

"""
    + _MIMIC_IV_SCHEMA
    + """
Write a python code to solve the given question. You can use the following functions:
"""
    + _MIMIC_IV_TOOL_GUIDE
    + """
Use the variable 'answer' to store the answer of the code. Use 1/0 for a well-defined existence question, None when a requested concept or scalar result is unavailable, and a deterministically ordered list for multi-row answers. Here are some examples:
{examples}
(END OF EXAMPLES)
Knowledge:
{knowledge}
Question: {question}
Solution: """
)

CodeDebugger = (
    """Given a question:
{question}
The user have written code with the following functions:
"""
    + _MIMIC_IV_TOOL_GUIDE
    + """
The code is as follows:
{code}

The execution result is:
{error_info}

Please check the code and point out the most possible reason to the error.
"""
)

EHRAgent_4Shots_Knowledge = """Question: what was the minimum gcs score recorded across all icu stays of patient 10002428?
Knowledge:
- We can find every icu stay of patient 10002428 in the static_information database.
- As gcs scores are stored in the gcs database, we can filter it by subject_id and stay_id for each stay and take the minimum.
Solution: # We can find every icu stay of patient 10002428 in the static_information database.
stays_db = LoadDB('static_information')
filtered_stays_db = FilterDB(stays_db, 'subject_id=10002428')
stay_id_list = GetValue(filtered_stays_db, 'stay_id, list')
# As gcs scores are stored in the gcs database, we can filter it by subject_id and stay_id for each stay and take the minimum.
min_gcs = None
for stay_id in stay_id_list:
    gcs_db = LoadDB('gcs')
    filtered_gcs_db = FilterDB(gcs_db, 'subject_id=10002428||stay_id={}'.format(int(float(stay_id))))
    stay_min_gcs = GetValue(filtered_gcs_db, 'gcs, min')
    if min_gcs is None or stay_min_gcs < min_gcs:
        min_gcs = stay_min_gcs
answer = min_gcs

Question: had any cefepime been given to patient 10005817 in their last hospital visit?
Knowledge:
- We can find the visiting information of patient 10005817 in the static_information database; max(admittime) identifies the last hospital visit.
- As antibiotic administrations are stored in the antibiotic database and its drug column stores the curated label CefePIME, we will check whether any matching record exists for that admission.
Solution: # We can find the visiting information of patient 10005817 in the static_information database; max(admittime) identifies the last hospital visit.
stays_db = LoadDB('static_information')
filtered_stays_db = FilterDB(stays_db, 'subject_id=10005817||max(admittime)')
hadm_id = int(float(GetValue(filtered_stays_db, 'hadm_id')))
# As antibiotic administrations are stored in the antibiotic database and its drug column stores the curated label CefePIME, we will check whether any matching record exists for that admission.
antibiotic_db = LoadDB('antibiotic')
filtered_antibiotic_db = FilterDB(antibiotic_db, 'subject_id=10005817||hadm_id={}||drug=CefePIME'.format(hadm_id))
if len(filtered_antibiotic_db) > 0:
    answer = 1
else:
    answer = 0

Question: what was the name of the procedure that was given two or more times to patient 10003400?
Knowledge:
- As procedures are stored in the procedures database with a curated item label, no dictionary join is needed.
- Counting events per item requires grouping, so a single SQLInterpreter query is the right tool.
Solution: answer = SQLInterpreter('select item from procedures where subject_id = 10003400 group by item having count(*) >= 2 order by item')

Question: calculate the length of stay of the first stay of patient 10001884 in the icu.
Knowledge:
- We can find the icu stays of patient 10001884 in the static_information database; min(intime) identifies the first icu stay.
- As we only need the length of stay, we can read intime and outtime and compute the difference.
Solution: from datetime import datetime
# We can find the icu stays of patient 10001884 in the static_information database; min(intime) identifies the first icu stay.
stays_db = LoadDB('static_information')
filtered_stays_db = FilterDB(stays_db, 'subject_id=10001884||min(intime)')
# As we only need the length of stay, we can read intime and outtime and compute the difference.
intime = GetValue(filtered_stays_db, 'intime')
outtime = GetValue(filtered_stays_db, 'outtime')
intime = datetime.strptime(intime, '%Y-%m-%d %H:%M:%S')
outtime = datetime.strptime(outtime, '%Y-%m-%d %H:%M:%S')
length_of_stay = outtime - intime
if length_of_stay.seconds // 3600 > 12:
    answer = length_of_stay.days + 1
else:
    answer = length_of_stay.days
"""
