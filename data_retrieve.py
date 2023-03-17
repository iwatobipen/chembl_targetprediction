#!/usr/bin/env python
# coding: utf-8
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import tables as tb
import json
from tables.atom import ObjectAtom

engine = create_engine('sqlite:///chembl_32/chembl_32_sqlite/chembl_32.db') 


qtext = """
SELECT
  activities.doc_id                    AS doc_id,
  activities.standard_value            AS standard_value,
  molecule_hierarchy.parent_molregno   AS molregno,
  compound_structures.canonical_smiles AS canonical_smiles,
  molecule_dictionary.chembl_id        AS chembl_id,
  target_dictionary.tid                AS tid,
  target_dictionary.chembl_id          AS target_chembl_id,
  protein_classification.pref_name     AS pref_name,
  protein_classification.short_name     AS short_name,
  protein_classification.PROTEIN_CLASS_DESC    AS protein_class,
  protein_classification.class_level     AS class_level
FROM activities
  JOIN assays ON activities.assay_id = assays.assay_id
  JOIN target_dictionary ON assays.tid = target_dictionary.tid
  JOIN target_components ON target_dictionary.tid = target_components.tid
  JOIN component_class ON target_components.component_id = component_class.component_id
  JOIN protein_classification ON component_class.protein_class_id = protein_classification.protein_class_id
  JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
  JOIN molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
  JOIN compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
WHERE activities.standard_units = 'nM' AND
      activities.standard_type IN ('EC50', 'IC50', 'Ki', 'Kd', 'XC50', 'AC50', 'Potency') AND
      activities.data_validity_comment IS NULL AND
      activities.standard_relation IN ('=', '<') AND
      activities.potential_duplicate = 0 AND assays.confidence_score >= 8 AND
      target_dictionary.target_type = 'SINGLE PROTEIN'"""

with engine.begin() as conn:
    res = conn.execute(text(qtext))
    df = pd.DataFrame(res.fetchall())



df.columns = res.keys()
df = df.where((pd.notnull(df)), None)



cls_list=df["protein_class"].to_list()

uniq = list(set(cls_list))




df = df.sort_values(by=['standard_value', 'molregno', 'tid'], ascending=True)
df = df.drop_duplicates(subset=['molregno', 'tid'], keep='first')

df.to_csv('chembl_activity_data.csv', index=False)




def set_active(row):
    active = 0
    if row['standard_value'] <= 1000:
        active = 1
    if "ion channel" in row['protein_class']:
        if row['standard_value'] <= 10000:
            active = 1
    if "kinase" in row['protein_class']:
        if row['standard_value'] > 30:
            active = 0
    if "nuclear receptor" in row['protein_class']:
        if row['standard_value'] > 100:
            active = 0
    if "membrane receptor" in row['protein_class']:
        if row['standard_value'] > 100:
            active = 0
    return active

df['active'] = df.apply(lambda row: set_active(row), axis=1)

# get targets with at least 100 different active molecules
acts = df[df['active'] == 1].groupby(['target_chembl_id']).agg('count')
acts = acts[acts['molregno'] >= 100].reset_index()['target_chembl_id']

# get targets with at least 100 different inactive molecules
inacts = df[df['active'] == 0].groupby(['target_chembl_id']).agg('count')
inacts = inacts[inacts['molregno'] >= 100].reset_index()['target_chembl_id']

# get targets mentioned in at least two docs
docs = df.drop_duplicates(subset=['doc_id', 'target_chembl_id'])
docs = docs.groupby(['target_chembl_id']).agg('count')
docs = docs[docs['doc_id'] >= 2.0].reset_index()['target_chembl_id']



t_keep = set(acts).intersection(set(inacts)).intersection(set(docs))

# get dta for filtered targets
activities = df[df['target_chembl_id'].isin(t_keep)]


ion = pd.unique(activities[activities['protein_class'].str.contains("ion channel",  na=False)]['tid']).shape[0]
kin = pd.unique(activities[activities['protein_class'].str.contains("kinase",  na=False)]['tid']).shape[0]
nuc = pd.unique(activities[activities['protein_class'].str.contains("nuclear receptor",  na=False)]['tid']).shape[0]
gpcr = pd.unique(activities[activities['protein_class'].str.contains("membrane receptor", na=False)]['tid']).shape[0]
print('Number of unique targets: ', len(t_keep))
print('  Ion channel: ', ion)
print('  Kinase: ', kin)
print('  Nuclear receptor: ',  nuc)
print('  GPCR: ', gpcr)
print('  Others: ', len(t_keep) - ion - kin - nuc - gpcr)


# save it to a file
activities.to_csv('chembl_activity_data_filtered.csv', index=False)




def gen_dict(group):
    return {tid: act  for tid, act in zip(group['target_chembl_id'], group['active'])}

print('MULTI TASK DATA PREP')
group = activities.groupby('chembl_id')
temp = pd.DataFrame(group.apply(gen_dict))
mt_df = pd.DataFrame(temp[0].tolist())
mt_df['chembl_id'] = temp.index
mt_df = mt_df.where((pd.notnull(mt_df)), -1)




structs = activities[['chembl_id', 'canonical_smiles']].drop_duplicates(subset='chembl_id')

print('GET MOL')
# drop mols not sanitizing on rdkit
def molchecker(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        return None
    else:
        return 1

#structs['romol'] = structs.apply(lambda row: Chem.MolFromSmiles(row['canonical_smiles']), axis=1)
structs['romol'] = structs.apply(lambda row: molchecker(row['canonical_smiles']), axis=1)
structs = structs.dropna()
del structs['romol']

# add the structures to the final df
mt_df = pd.merge(structs, mt_df, how='inner', on='chembl_id')


# save to csv
mt_df.to_csv('chembl_multi_task_data.csv', index=False)




FP_SIZE = 1024 
RADIUS = 2

def calc_fp(smiles, fp_size, radius):
    """
    calcs morgan fingerprints as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(False)
    Chem.GetSSSR(mol)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

# calc fps
print('CALC FP')
descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in mt_df['canonical_smiles'].values]
descs = np.asarray(descs, dtype=np.float32)

# put all training data in a pytables file
print('SAVE DATA')
with tb.open_file('mt_data.h5', mode='w') as t_file:

    # set compression filter. It will make the file much smaller
    filters = tb.Filters(complib='blosc', complevel=5)

    # save chembl_ids
    tatom = ObjectAtom()
    cids = t_file.create_vlarray(t_file.root, 'chembl_ids', atom=tatom)
    for cid in mt_df['chembl_id'].values:
        cids.append(cid)

    # save fps
    fatom = tb.Atom.from_dtype(descs.dtype)
    fps = t_file.create_carray(t_file.root, 'fps', fatom, descs.shape, filters=filters)
    fps[:] = descs

    del mt_df['chembl_id']
    del mt_df['canonical_smiles']

    # save target chembl ids
    tcids = t_file.create_vlarray(t_file.root, 'target_chembl_ids', atom=tatom)
    for tcid in mt_df.columns.values:
        tcids.append(tcid)

    # save labels
    labs = t_file.create_carray(t_file.root, 'labels', fatom, mt_df.values.shape, filters=filters)
    labs[:] = mt_df.values
    
    # save task weights
    # each task loss will be weighted inversely proportional to its number of data points
    weights = []
    for col in mt_df.columns.values:
        c = mt_df[mt_df[col] >= 0.0].shape[0]
        weights.append(1 / c)
    weights = np.array(weights)
    ws = t_file.create_carray(t_file.root, 'weights', fatom, weights.shape)
    ws[:] = weights




with tb.open_file('mt_data.h5', mode='r') as t_file:
    print(t_file.root.chembl_ids.shape)
    print(t_file.root.target_chembl_ids.shape)
    print(t_file.root.fps.shape)
    print(t_file.root.labels.shape)
    print(t_file.root.weights.shape)
    
    # save targets to a json file
    with open('targets.json', 'w') as f:
        json.dump(t_file.root.target_chembl_ids[:], f)

